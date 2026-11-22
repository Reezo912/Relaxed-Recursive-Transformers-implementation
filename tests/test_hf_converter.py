from __future__ import annotations

import copy

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from rrt import (
    SharedWeightLoRALinear,
    build_layer_groups,
    convert_hf_model_to_relaxed,
    hf_parameter_summary,
)


def _make_tiny_qwen3(num_layers: int = 8) -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    return Qwen3ForCausalLM(cfg)


@pytest.mark.parametrize(
    "strategy,num_layers,num_recursions",
    [
        ("cycle", 8, 2),
        ("sequence", 8, 2),
        ("middle_cycle", 10, 2),
        ("middle_sequence", 10, 2),
    ],
)
def test_hf_qwen3_full_rank_svd_exact_reconstruction(
    strategy: str,
    num_layers: int,
    num_recursions: int,
) -> None:
    """Full-rank SVD conversion of Qwen3ForCausalLM must exactly reproduce logits and loss."""
    original = _make_tiny_qwen3(num_layers=num_layers)
    original.eval()
    converted = copy.deepcopy(original)

    # hidden_size=32 is >= min(out, in) across all projections (q/k/v/o/gate/up/down)
    report = convert_hf_model_to_relaxed(
        converted,
        num_recursions=num_recursions,
        sharing_strategy=strategy,  # type: ignore[arg-type]
        lora_rank=32,
        base_init="mean",
        use_svd=True,
    )
    converted.eval()

    assert report.mean_rel_error_relaxed < 1e-5
    assert report.mean_energy_captured > 0.9999

    x = torch.randint(0, 128, (2, 16))
    with torch.no_grad():
        out_orig = original(input_ids=x, labels=x)
        out_cvt = converted(input_ids=x, labels=x)

    assert torch.allclose(out_orig.logits, out_cvt.logits, atol=1e-4)
    assert torch.allclose(out_orig.loss, out_cvt.loss, atol=1e-4)


def test_hf_qwen3_true_weight_sharing_and_param_reduction() -> None:
    model = _make_tiny_qwen3(num_layers=10)
    before = hf_parameter_summary(model)

    convert_hf_model_to_relaxed(
        model,
        num_recursions=2,
        sharing_strategy="middle_cycle",
        lora_rank=4,
        use_svd=True,
    )
    after = hf_parameter_summary(model)

    groups = build_layer_groups(10, 2, "middle_cycle")
    assert groups[0] == [0]
    assert groups[-1] == [9]
    assert groups[1] == [1, 5]

    # Entry & exit remain plain nn.Linear; tied middle layers share exact same weight storage
    assert not isinstance(model.model.layers[0].self_attn.q_proj, SharedWeightLoRALinear)
    assert not isinstance(model.model.layers[9].self_attn.q_proj, SharedWeightLoRALinear)

    p1 = model.model.layers[1].self_attn.q_proj
    p5 = model.model.layers[5].self_attn.q_proj
    assert isinstance(p1, SharedWeightLoRALinear) and isinstance(p5, SharedWeightLoRALinear)
    assert p1.weight is p5.weight
    assert p1.weight.data_ptr() == p5.weight.data_ptr()
    assert p1.lora_A is not p5.lora_A and p1.lora_B is not p5.lora_B

    # Deduplicated parameter count must drop noticeably vs original
    assert after["lora"] > 0
    assert after["shared_base"] > 0
    assert after["total"] < before["total"]
    # Verify PyTorch's model.parameters() deduplicates the tied weights identically
    torch_unique_params = sum(p.numel() for p in model.parameters())
    assert torch_unique_params == after["total"]


def test_hf_qwen3_gradients_kv_cache_generate_and_grad_checkpointing() -> None:
    model = _make_tiny_qwen3(num_layers=8)
    convert_hf_model_to_relaxed(
        model,
        num_recursions=2,
        sharing_strategy="cycle",
        lora_rank=4,
        use_svd=True,
    )

    # 1. Verify native HF generate() with DynamicCache (KV cache across all virtual layers)
    model.eval()
    prompt = torch.randint(0, 128, (1, 6))
    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=5, do_sample=False, use_cache=True)
    assert generated.shape == (1, 11)

    # 2. Verify training + activation/gradient checkpointing flows gradients to shared weight & distinct LoRAs
    model.train()
    if hasattr(model, "activation_checkpointing_enable"):
        model.activation_checkpointing_enable()
    else:
        model.gradient_checkpointing_enable()
    x = torch.randint(0, 128, (2, 12))
    out = model(input_ids=x, labels=x, use_cache=False)
    out.loss.backward()

    q0 = model.model.layers[0].self_attn.q_proj
    q4 = model.model.layers[4].self_attn.q_proj
    assert isinstance(q0, SharedWeightLoRALinear) and isinstance(q4, SharedWeightLoRALinear)
    assert q0.weight.grad is not None and q0.weight.grad.abs().sum() > 0
    assert q0.lora_A is not None and q0.lora_A.grad is not None and q0.lora_A.grad.abs().sum() > 0
    assert q4.lora_A is not None and q4.lora_A.grad is not None and q4.lora_A.grad.abs().sum() > 0
    assert not torch.allclose(q0.lora_A.grad, q4.lora_A.grad)


def test_hf_qwen3_keep_first_last_and_whitened_svd_exact_and_better_output_error() -> None:
    """keep_first/keep_last must preserve sensitive layers and whitened SVD must reconstruct full-rank exactly."""
    groups = build_layer_groups(12, num_recursions=2, sharing_strategy="middle_sequence", keep_first=3, keep_last=1)
    assert groups == [[0], [1], [2], [3, 4], [5, 6], [7, 8], [9, 10], [11]]

    torch.manual_seed(7)
    original = _make_tiny_qwen3(num_layers=10)
    original.eval()
    calib = [torch.randint(0, 128, (4, 16)) for _ in range(4)]

    # 1. Full-rank whitened SVD must reproduce logits nearly identically
    full_rank_cvt = copy.deepcopy(original)
    convert_hf_model_to_relaxed(
        full_rank_cvt,
        num_recursions=2,
        sharing_strategy="middle_sequence",
        keep_first=3,
        keep_last=1,
        lora_rank=32,
        use_svd=True,
        calibration_batches=calib,
    )
    full_rank_cvt.eval()
    assert not isinstance(full_rank_cvt.model.layers[2].self_attn.q_proj, SharedWeightLoRALinear)
    assert isinstance(full_rank_cvt.model.layers[3].self_attn.q_proj, SharedWeightLoRALinear)
    assert not isinstance(full_rank_cvt.model.layers[9].self_attn.q_proj, SharedWeightLoRALinear)

    x_eval = torch.randint(0, 128, (2, 16))
    with torch.no_grad():
        out_orig = original(input_ids=x_eval).logits
        out_full = full_rank_cvt(input_ids=x_eval).logits
    assert torch.allclose(out_orig, out_full, atol=1e-3)

    # 2. Low-rank whitened SVD with anisotropic inputs must beat naive unweighted SVD in output MSE
    torch.manual_seed(11)
    m_plain = copy.deepcopy(original)
    m_white = copy.deepcopy(original)
    # Inject strong channel scaling in embeddings (mimicking LLM outlier channels)
    with torch.no_grad():
        scale = torch.ones(32)
        scale[:4] = 25.0
        original.model.embed_tokens.weight.mul_(scale)
        m_plain.model.embed_tokens.weight.copy_(original.model.embed_tokens.weight)
        m_white.model.embed_tokens.weight.copy_(original.model.embed_tokens.weight)

    convert_hf_model_to_relaxed(
        m_plain, num_recursions=2, sharing_strategy="middle_sequence", keep_first=3, keep_last=1, lora_rank=4, use_svd=True
    )
    convert_hf_model_to_relaxed(
        m_white,
        num_recursions=2,
        sharing_strategy="middle_sequence",
        keep_first=3,
        keep_last=1,
        lora_rank=4,
        use_svd=True,
        calibration_batches=calib,
    )
    with torch.no_grad():
        ref = original(input_ids=calib[0]).logits
        err_plain = (m_plain(input_ids=calib[0]).logits - ref).pow(2).mean().item()
        err_white = (m_white(input_ids=calib[0]).logits - ref).pow(2).mean().item()
    assert err_white < err_plain

    # 3. Verify condition-bounded ASVD never blows up w_base or unbalances (lora_A, lora_B)
    orig_max_w = max(
        p.detach().float().abs().max().item()
        for n, p in original.named_parameters()
        if "layers." in n and p.ndim == 2
    )
    for mod in m_white.modules():
        if isinstance(mod, SharedWeightLoRALinear) and mod.lora_A is not None and mod.lora_B is not None:
            assert mod.weight.detach().float().abs().max().item() <= orig_max_w + 1e-4
            norm_a = mod.lora_A.detach().float().norm().item()
            norm_b = mod.lora_B.detach().float().norm().item()
            assert abs(norm_a - norm_b) <= 1e-3 * max(norm_a, norm_b, 1e-6)


def test_adaptive_rank_preserves_budget_and_dora_out_scale_flows_gradients() -> None:
    from rrt.hf_converter import compute_adaptive_layer_ranks

    # Check Qwen3-1.7B exact 28-layer layout (keep_first=3, keep_last=1 -> 12 tied pairs)
    groups = build_layer_groups(28, num_recursions=2, sharing_strategy="middle_sequence", keep_first=3, keep_last=1)
    ranks = compute_adaptive_layer_ranks(groups, base_rank=128)
    vals = list(ranks.values())
    assert len(vals) == 12
    assert sum(vals) == 12 * 128  # Exact parameter budget preservation
    assert all(r % 16 == 0 for r in vals)  # Tensor-core aligned
    assert vals[0] > vals[len(vals) // 2] and vals[-1] > vals[len(vals) // 2]  # U-shaped
    assert vals == list(reversed(vals))  # Symmetric

    # Verify parameter count equivalence & DoRA out_scale forward/grad on tiny Qwen3
    torch.manual_seed(21)
    m_uni = _make_tiny_qwen3(num_layers=14)
    m_adp = copy.deepcopy(m_uni)
    convert_hf_model_to_relaxed(
        m_uni, num_recursions=2, sharing_strategy="middle_sequence", keep_first=2, keep_last=2, lora_rank=8, use_svd=True
    )
    convert_hf_model_to_relaxed(
        m_adp,
        num_recursions=2,
        sharing_strategy="middle_sequence",
        keep_first=2,
        keep_last=2,
        lora_rank=8,
        use_svd=True,
        adaptive_rank=True,
        use_dora_scale=True,
    )
    uni_total = hf_parameter_summary(m_uni)["total"]
    adp_total = hf_parameter_summary(m_adp)["total"]
    # Only difference should be the tiny 1D out_scale vectors
    out_scale_params = sum(p.numel() for n, p in m_adp.named_parameters() if "out_scale" in n)
    assert out_scale_params > 0
    assert adp_total == uni_total + out_scale_params

    # Gradients flow into out_scale, lora_A, lora_B, and shared weight
    m_adp.train()
    x = torch.randint(0, 128, (2, 10))
    m_adp(input_ids=x, labels=x).loss.backward()
    q2 = m_adp.model.layers[2].self_attn.q_proj
    assert isinstance(q2, SharedWeightLoRALinear)
    assert q2.out_scale is not None and q2.out_scale.grad is not None and q2.out_scale.grad.abs().sum() > 0
    assert q2.effective_weight().shape == q2.weight.shape

