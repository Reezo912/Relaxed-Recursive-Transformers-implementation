from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from rrt import RRTConfig, RelaxedLinear, RelaxedRecursiveTransformer


def test_relaxed_linear_zero_init_equivalence() -> None:
    torch.manual_seed(0)
    layer = RelaxedLinear(in_features=32, out_features=48, num_recursions=4, lora_rank=8)
    x = torch.randn(2, 5, 32)
    base_expected = F.linear(x, layer.weight, layer.bias)

    for k in range(4):
        out = layer(x, recursion_idx=k)
        assert torch.allclose(out, base_expected, atol=1e-6)
        assert layer.relative_lora_norm(k) == 0.0


def test_relaxed_linear_specialization_and_effective_weight() -> None:
    torch.manual_seed(1)
    layer = RelaxedLinear(in_features=32, out_features=48, num_recursions=3, lora_rank=4, lora_alpha=8.0)
    with torch.no_grad():
        for k in range(3):
            layer.lora_B[k].normal_(mean=0.0, std=0.1 * (k + 1))

    x = torch.randn(2, 6, 32)
    outputs = [layer(x, recursion_idx=k) for k in range(3)]

    for k in range(3):
        eff_w = layer.effective_weight(k)
        assert torch.allclose(outputs[k], F.linear(x, eff_w), atol=1e-5)
        assert layer.relative_lora_norm(k) > 0.0

    # Different recursion steps produce different outputs despite sharing layer.weight
    assert not torch.allclose(outputs[0], outputs[1])
    assert not torch.allclose(outputs[1], outputs[2])


def test_cycle_and_middle_cycle_schedules_and_param_counts() -> None:
    common = dict(vocab_size=64, d_model=64, n_heads=4, max_seq_len=32, num_layers=12)

    standard = RelaxedRecursiveTransformer(RRTConfig(**common, num_recursions=1, lora_rank=0))
    strict = RelaxedRecursiveTransformer(
        RRTConfig(**common, num_recursions=4, sharing_strategy="cycle", lora_rank=0)
    )
    relaxed = RelaxedRecursiveTransformer(
        RRTConfig(**common, num_recursions=4, sharing_strategy="cycle", lora_rank=8)
    )

    assert len(standard.blocks) == 12
    assert len(strict.blocks) == 3
    assert len(relaxed.blocks) == 3
    assert len(relaxed.layer_schedule) == 12
    assert relaxed.layer_schedule == [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3),
    ]

    s_params = standard.parameter_summary()
    r_strict_params = strict.parameter_summary()
    r_relaxed_params = relaxed.parameter_summary()

    assert r_strict_params["lora"] == 0
    assert r_relaxed_params["lora"] > 0
    assert r_strict_params["shared_base"] == r_relaxed_params["shared_base"]
    assert r_strict_params["total"] < r_relaxed_params["total"] < s_params["total"]

    # Middle-cycle: 1 entry + 2 unique middle * 4 loops + 1 exit = 10 virtual layers, 4 unique blocks
    mid = RelaxedRecursiveTransformer(
        RRTConfig(
            vocab_size=64,
            d_model=64,
            n_heads=4,
            max_seq_len=32,
            num_layers=10,
            num_recursions=4,
            sharing_strategy="middle_cycle",
            lora_rank=8,
        )
    )
    assert len(mid.blocks) == 4
    assert len(mid.layer_schedule) == 10
    assert mid.layer_schedule[0] == (0, 0)
    assert mid.layer_schedule[-1] == (3, 0)
    # Entry and exit blocks have no LoRA; middle blocks do
    assert mid.blocks[0].attn.q_proj.lora_rank == 0
    assert mid.blocks[3].attn.q_proj.lora_rank == 0
    assert mid.blocks[1].attn.q_proj.lora_rank == 8
    assert mid.blocks[2].attn.q_proj.lora_rank == 8


def test_strict_vs_relaxed_exact_equivalence_at_init() -> None:
    torch.manual_seed(42)
    base_cfg = dict(
        vocab_size=64,
        d_model=64,
        n_heads=4,
        max_seq_len=32,
        num_layers=8,
        num_recursions=4,
        sharing_strategy="cycle",
    )
    strict = RelaxedRecursiveTransformer(RRTConfig(**base_cfg, lora_rank=0))
    relaxed = RelaxedRecursiveTransformer(RRTConfig(**base_cfg, lora_rank=8))

    # Load all non-LoRA weights from strict into relaxed
    missing, unexpected = relaxed.load_state_dict(strict.state_dict(), strict=False)
    assert unexpected == []
    assert all("lora_" in k for k in missing)

    strict.eval()
    relaxed.eval()
    x = torch.randint(0, 64, (2, 16))
    y = torch.randint(0, 64, (2, 16))

    logits_s, loss_s = strict(x, y)
    logits_r, loss_r = relaxed(x, y)
    assert loss_s is not None and loss_r is not None
    assert torch.allclose(logits_s, logits_r, atol=1e-5)
    assert torch.allclose(loss_s, loss_r, atol=1e-5)


def test_gradient_flow_to_shared_weights_and_per_recursion_lora() -> None:
    torch.manual_seed(7)
    cfg = RRTConfig(
        vocab_size=48,
        d_model=64,
        n_heads=4,
        max_seq_len=24,
        num_layers=6,
        num_recursions=3,
        sharing_strategy="cycle",
        lora_rank=4,
    )
    model = RelaxedRecursiveTransformer(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    x = torch.randint(0, 48, (4, 16))
    y = torch.randint(0, 48, (4, 16))

    # Step 1: B_k == 0 initially, so shared W and B_k receive non-zero distinct grads
    _, loss = model(x, y)
    assert loss is not None
    loss.backward()

    q_proj = model.blocks[0].attn.q_proj
    assert q_proj.weight.grad is not None and q_proj.weight.grad.abs().sum() > 0
    assert q_proj.lora_B is not None and q_proj.lora_A is not None
    for k in range(3):
        grad_b = q_proj.lora_B[k].grad
        assert grad_b is not None and grad_b.abs().sum() > 0
    assert not torch.allclose(q_proj.lora_B[0].grad, q_proj.lora_B[1].grad)

    optimizer.step()
    optimizer.zero_grad()

    # Step 2: Now that B_k != 0, A_k must also receive distinct non-zero gradients
    _, loss2 = model(x, y)
    assert loss2 is not None
    loss2.backward()
    for k in range(3):
        grad_a = q_proj.lora_A[k].grad
        assert grad_a is not None and grad_a.abs().sum() > 0
    assert not torch.allclose(q_proj.lora_A[0].grad, q_proj.lora_A[1].grad)


def test_causal_masking_no_future_leakage() -> None:
    torch.manual_seed(123)
    cfg = RRTConfig(
        vocab_size=32,
        d_model=32,
        n_heads=4,
        max_seq_len=16,
        num_layers=6,
        num_recursions=3,
        lora_rank=4,
    )
    model = RelaxedRecursiveTransformer(cfg)
    model.eval()

    x1 = torch.randint(0, 32, (1, 12))
    x2 = x1.clone()
    x2[0, 6:] = (x2[0, 6:] + 1) % 32  # Perturb future tokens from index 6 onwards

    logits1, _ = model(x1)
    logits2, _ = model(x2)
    assert torch.allclose(logits1[:, :6], logits2[:, :6], atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU not available")
def test_cuda_bf16_forward_backward_and_generate() -> None:
    device = torch.device("cuda")
    cfg = RRTConfig(
        vocab_size=64,
        d_model=128,
        n_heads=4,
        max_seq_len=64,
        num_layers=8,
        num_recursions=4,
        sharing_strategy="cycle",
        lora_rank=8,
    )
    model = RelaxedRecursiveTransformer(cfg).to(device)
    x = torch.randint(0, 64, (4, 32), device=device)
    y = torch.randint(0, 64, (4, 32), device=device)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits, loss = model(x, y)

    assert logits.shape == (4, 32, 64)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()

    prompt = torch.randint(0, 64, (1, 8), device=device)
    generated = model.generate(prompt, max_new_tokens=6, temperature=1.0, top_k=10)
    assert generated.shape == (1, 14)


def test_relaxed_linear_split_vs_merge_exact_forward_and_gradients() -> None:
    """merge-on-the-fly and split LoRA must produce identical outputs and gradients."""
    torch.manual_seed(11)
    lin_split = RelaxedLinear(32, 48, num_recursions=3, lora_rank=8, lora_mode="split")
    lin_merge = RelaxedLinear(32, 48, num_recursions=3, lora_rank=8, lora_mode="merge")
    with torch.no_grad():
        for k in range(3):
            lin_split.lora_B[k].normal_(0.0, 0.1)
    lin_merge.load_state_dict(lin_split.state_dict())

    x1 = torch.randn(4, 10, 32, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    grad_out = torch.randn(4, 10, 48)

    y1 = lin_split(x1, recursion_idx=1)
    y2 = lin_merge(x2, recursion_idx=1)
    assert torch.allclose(y1, y2, atol=1e-5)

    y1.backward(grad_out)
    y2.backward(grad_out)
    assert torch.allclose(x1.grad, x2.grad, atol=1e-5)
    assert torch.allclose(lin_split.weight.grad, lin_merge.weight.grad, atol=1e-5)
    assert torch.allclose(lin_split.lora_A[1].grad, lin_merge.lora_A[1].grad, atol=1e-5)
    assert torch.allclose(lin_split.lora_B[1].grad, lin_merge.lora_B[1].grad, atol=1e-5)


def test_fused_projections_model_forward_backward() -> None:
    torch.manual_seed(17)
    cfg = RRTConfig(
        vocab_size=64,
        d_model=64,
        n_heads=4,
        max_seq_len=32,
        num_layers=8,
        num_recursions=4,
        lora_rank=8,
        lora_mode="auto",
        fuse_projections=True,
    )
    model = RelaxedRecursiveTransformer(cfg)
    assert len(model.blocks[0].relaxed_linears()) == 4  # qkv, o, gate_up, down

    x = torch.randint(0, 64, (2, 16))
    y = torch.randint(0, 64, (2, 16))
    logits, loss = model(x, y)
    assert logits.shape == (2, 16, 64)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert model.blocks[0].attn.qkv_proj.weight.grad is not None
    assert model.blocks[0].ffn.gate_up_proj.lora_B[0].grad is not None
