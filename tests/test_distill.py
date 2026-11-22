from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from rrt.distill import (
    CPUWeightEMA,
    CrossArchHiddenProjector,
    Int8WeightOnlyLinear,
    StochasticBF16Adafactor,
    chunked_kl_ce_backward,
    compute_layerwise_hidden_loss,
    compute_matched_layer_pairs,
    compute_projected_hidden_loss,
    fp32_to_bf16_stochastic,
    quantize_model_int8_weight_only,
)


def test_fp32_to_bf16_stochastic_is_unbiased_and_avoids_stalling() -> None:
    """Stochastic rounding must be unbiased and accumulate sub-ULP updates without stalling."""
    torch.manual_seed(0)
    # 1.0 + 1e-4 is well below bf16 half-ULP (~0.0039), so naive bf16 rounds to 1.0 every time
    val = torch.full((20000,), 1.0005, dtype=torch.float32)
    naive = val.to(torch.bfloat16).float().mean().item()
    stoch = fp32_to_bf16_stochastic(val).float().mean().item()
    assert naive == pytest.approx(1.0, abs=1e-6)
    assert stoch == pytest.approx(1.0005, abs=3e-4)

    # Also test negative values
    neg = torch.full((20000,), -1.001, dtype=torch.float32)
    stoch_neg = fp32_to_bf16_stochastic(neg).float().mean().item()
    assert stoch_neg == pytest.approx(-1.001, abs=3e-4)


@pytest.mark.parametrize("beta1", [0.0, 0.9])
def test_stochastic_bf16_adafactor_reduces_loss_on_bf16_params(beta1: float) -> None:
    """StochasticBF16Adafactor (with and without bf16 momentum) must minimize a quadratic cleanly."""
    torch.manual_seed(42)
    w = nn.Parameter(torch.ones(64, 64, dtype=torch.bfloat16))
    target = torch.full((64, 64), 0.75, dtype=torch.bfloat16)
    opt = StochasticBF16Adafactor([w], lr=1e-3, beta1=beta1, weight_decay=0.0)

    init_loss = F.mse_loss(w.float(), target.float()).item()
    for _ in range(150):
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(w.float(), target.float())
        loss.backward()
        opt.step()
    final_loss = F.mse_loss(w.float(), target.float()).item()
    assert final_loss < init_loss * 0.25


def test_chunked_kl_ce_backward_with_aux_hidden_loss_matches_full_exact() -> None:
    """Single-backward chunked KL+CE + layerwise hidden loss must match full backward exactly."""
    torch.manual_seed(3)
    bsz, seq_len, d_model, vocab = 2, 13, 32, 96
    l1_ref, l2_ref = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
    l1_chk, l2_chk = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
    l1_chk.load_state_dict(l1_ref.state_dict())
    l2_chk.load_state_dict(l2_ref.state_dict())

    head = nn.Linear(d_model, vocab, bias=False)
    head.weight.requires_grad_(False)
    x_in = torch.randn(bsz, seq_len, d_model)
    t_h1 = torch.randn(bsz, seq_len, d_model)
    t_h2 = torch.randn(bsz, seq_len, d_model)
    ids = torch.randint(0, vocab, (bsz, seq_len))

    # Full-tensor reference
    s1_ref = l1_ref(x_in)
    s2_ref = l2_ref(s1_ref)
    aux_ref = 1.5 * compute_layerwise_hidden_loss((s1_ref, s2_ref), (t_h1, t_h2), [0, 1])
    s_log = head(s2_ref[:, :-1]).reshape(-1, vocab)
    t_log = head(t_h2[:, :-1]).reshape(-1, vocab)
    tgts = ids[:, 1:].reshape(-1)
    ce_ref = F.cross_entropy(s_log, tgts)
    kl_ref = F.kl_div(F.log_softmax(s_log / 1.5, dim=-1), F.softmax(t_log / 1.5, dim=-1), reduction="batchmean") * 2.25
    ((0.9 * kl_ref + 0.1 * ce_ref + aux_ref) / 2.0).backward()

    # Chunked single-backward
    s1_chk = l1_chk(x_in)
    s2_chk = l2_chk(s1_chk)
    aux_chk = (1.5 * compute_layerwise_hidden_loss((s1_chk, s2_chk), (t_h1, t_h2), [0, 1])) / 2.0
    chunked_kl_ce_backward(
        student_hidden=s2_chk,
        teacher_hidden=t_h2,
        student_lm_head=head,
        teacher_lm_head=head,
        input_ids=ids,
        kl_weight=0.9,
        ce_weight=0.1,
        temperature=1.5,
        chunk_tokens=5,
        grad_accum=2,
        aux_loss=aux_chk,
    )

    assert torch.allclose(l1_chk.weight.grad, l1_ref.weight.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(l2_chk.weight.grad, l2_ref.weight.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("train_head", [False, True])
def test_chunked_kl_ce_backward_matches_full_tensor_exactly(train_head: bool) -> None:
    """Chunked KL+CE backward must match full-sequence KL+CE gradients to machine precision."""
    torch.manual_seed(0)
    bsz, seq_len, d_model, vocab = 2, 17, 32, 128
    kl_w, ce_w, temp, grad_accum = 0.7, 0.3, 1.5, 2

    s_proj_ref = nn.Linear(d_model, d_model)
    s_proj_chk = nn.Linear(d_model, d_model)
    s_proj_chk.load_state_dict(s_proj_ref.state_dict())

    s_head_ref = nn.Linear(d_model, vocab, bias=False)
    s_head_chk = nn.Linear(d_model, vocab, bias=False)
    s_head_chk.load_state_dict(s_head_ref.state_dict())
    s_head_ref.weight.requires_grad_(train_head)
    s_head_chk.weight.requires_grad_(train_head)

    t_head = nn.Linear(d_model, vocab, bias=False)
    t_head.weight.requires_grad_(False)

    x_in = torch.randn(bsz, seq_len, d_model)
    t_hidden = torch.randn(bsz, seq_len, d_model)
    input_ids = torch.randint(0, vocab, (bsz, seq_len))

    # Full-sequence reference
    hs_ref = s_proj_ref(x_in)
    s_logits = s_head_ref(hs_ref[:, :-1, :]).reshape(-1, vocab)
    t_logits = t_head(t_hidden[:, :-1, :]).reshape(-1, vocab)
    targets = input_ids[:, 1:].reshape(-1)
    ce_ref = F.cross_entropy(s_logits, targets, reduction="mean")
    kl_ref = F.kl_div(
        F.log_softmax(s_logits / temp, dim=-1),
        F.softmax(t_logits / temp, dim=-1),
        reduction="batchmean",
    ) * (temp * temp)
    loss_ref = (kl_w * kl_ref + ce_w * ce_ref) / grad_accum
    loss_ref.backward()

    # Chunked exact implementation (chunk size 5 doesn't evenly divide 32 tokens)
    hs_chk = s_proj_chk(x_in)
    ce_chk, kl_chk = chunked_kl_ce_backward(
        student_hidden=hs_chk,
        teacher_hidden=t_hidden,
        student_lm_head=s_head_chk,
        teacher_lm_head=t_head,
        input_ids=input_ids,
        kl_weight=kl_w,
        ce_weight=ce_w,
        temperature=temp,
        chunk_tokens=5,
        grad_accum=grad_accum,
    )

    assert ce_chk == pytest.approx(ce_ref.item(), rel=1e-5, abs=1e-5)
    assert kl_chk == pytest.approx(kl_ref.item(), rel=1e-5, abs=1e-5)
    assert torch.allclose(s_proj_chk.weight.grad, s_proj_ref.weight.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(s_proj_chk.bias.grad, s_proj_ref.bias.grad, atol=1e-5, rtol=1e-5)
    if train_head:
        assert torch.allclose(s_head_chk.weight.grad, s_head_ref.weight.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("jsd_beta", [0.1, 0.5])
def test_chunked_jsd_matches_full_sequence_reference(jsd_beta: float) -> None:
    torch.manual_seed(77)
    bsz, seq_len, d_model, vocab = 2, 13, 16, 48
    temp, kl_w, ce_w = 1.5, 0.9, 0.1

    s_proj_ref = nn.Linear(d_model, d_model)
    s_proj_chk = nn.Linear(d_model, d_model)
    s_proj_chk.load_state_dict(s_proj_ref.state_dict())

    head = nn.Linear(d_model, vocab, bias=False)
    head.weight.requires_grad_(False)
    t_head = nn.Linear(d_model, vocab, bias=False)
    t_head.weight.requires_grad_(False)

    x_in = torch.randn(bsz, seq_len, d_model)
    t_hidden = torch.randn(bsz, seq_len, d_model)
    input_ids = torch.randint(0, vocab, (bsz, seq_len))

    hs_ref = s_proj_ref(x_in)
    s_logits = head(hs_ref[:, :-1, :]).reshape(-1, vocab)
    t_logits = t_head(t_hidden[:, :-1, :]).reshape(-1, vocab)
    targets = input_ids[:, 1:].reshape(-1)

    ce_ref = F.cross_entropy(s_logits, targets, reduction="mean")
    p_s = F.softmax(s_logits / temp, dim=-1)
    p_t = F.softmax(t_logits / temp, dim=-1)
    m = jsd_beta * p_t + (1.0 - jsd_beta) * p_s
    jsd_norm = 1.0 / (jsd_beta * (1.0 - jsd_beta))
    jsd_ref = (
        jsd_beta * (p_t * (p_t.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum(dim=-1).mean()
        + (1.0 - jsd_beta) * (p_s * (p_s.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum(dim=-1).mean()
    ) * (temp * temp * jsd_norm)
    (kl_w * jsd_ref + ce_w * ce_ref).backward()

    hs_chk = s_proj_chk(x_in)
    ce_chk, jsd_chk = chunked_kl_ce_backward(
        student_hidden=hs_chk,
        teacher_hidden=t_hidden,
        student_lm_head=head,
        teacher_lm_head=t_head,
        input_ids=input_ids,
        kl_weight=kl_w,
        ce_weight=ce_w,
        temperature=temp,
        chunk_tokens=7,
        grad_accum=1,
        kl_type="jsd",
        jsd_beta=jsd_beta,
    )

    assert ce_chk == pytest.approx(ce_ref.item(), rel=1e-5, abs=1e-5)
    assert jsd_chk == pytest.approx(jsd_ref.item(), rel=1e-5, abs=1e-5)
    assert torch.allclose(s_proj_chk.weight.grad, s_proj_ref.weight.grad, atol=1e-5, rtol=1e-5)


def test_int8_weight_only_linear_accuracy_and_memory() -> None:
    torch.manual_seed(9)
    lin = nn.Linear(128, 256, bias=True).to(torch.bfloat16)
    q_lin = Int8WeightOnlyLinear(lin)

    x = torch.randn(4, 16, 128, dtype=torch.bfloat16)
    y_ref = lin(x).float()
    y_q = q_lin(x).float()

    rel_err = (y_ref - y_q).norm() / y_ref.norm().clamp_min(1e-6)
    assert rel_err.item() < 0.015
    assert q_lin.weight_q.dtype == torch.int8
    assert q_lin.weight_q.numel() * q_lin.weight_q.element_size() < lin.weight.numel() * lin.weight.element_size()


def test_quantize_model_int8_weight_only_preserves_logits() -> None:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(11)
    cfg = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    model = Qwen3ForCausalLM(cfg).to(torch.bfloat16).eval()
    ids = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        ref_logits = model(input_ids=ids).logits.float()

    quantize_model_int8_weight_only(model, device="cpu")
    with torch.no_grad():
        q_logits = model(input_ids=ids).logits.float()

    rel_err = (ref_logits - q_logits).norm() / ref_logits.norm().clamp_min(1e-6)
    assert rel_err.item() < 0.02
    assert any(isinstance(m, Int8WeightOnlyLinear) for m in model.modules())
    assert all(not p.requires_grad for p in model.parameters())


def test_chunked_kl_token_mask_matches_masked_reference_exact() -> None:
    torch.manual_seed(88)
    bsz, seq_len, d_model, vocab = 2, 14, 16, 48
    temp, jsd_beta = 1.2, 0.1

    s_proj_ref = nn.Linear(d_model, d_model)
    s_proj_chk = nn.Linear(d_model, d_model)
    s_proj_chk.load_state_dict(s_proj_ref.state_dict())

    head = nn.Linear(d_model, vocab, bias=False)
    head.weight.requires_grad_(False)
    t_head = nn.Linear(d_model, vocab, bias=False)
    t_head.weight.requires_grad_(False)

    x_in = torch.randn(bsz, seq_len, d_model)
    t_hidden = torch.randn(bsz, seq_len, d_model)
    input_ids = torch.randint(0, vocab, (bsz, seq_len))
    # Simulate prompt (False) followed by generated response (True)
    token_mask = torch.zeros((bsz, seq_len), dtype=torch.bool)
    token_mask[0, 5:] = True
    token_mask[1, 8:12] = True

    hs_ref = s_proj_ref(x_in)
    m_flat = token_mask[:, 1:].reshape(-1)
    s_logits = head(hs_ref[:, :-1, :].reshape(-1, d_model)[m_flat])
    t_logits = t_head(t_hidden[:, :-1, :].reshape(-1, d_model)[m_flat])
    p_s = F.softmax(s_logits / temp, dim=-1)
    p_t = F.softmax(t_logits / temp, dim=-1)
    m = jsd_beta * p_t + (1.0 - jsd_beta) * p_s
    jsd_norm = 1.0 / (jsd_beta * (1.0 - jsd_beta))
    jsd_ref = (
        jsd_beta * (p_t * (p_t.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum(dim=-1).mean()
        + (1.0 - jsd_beta) * (p_s * (p_s.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum(dim=-1).mean()
    ) * (temp * temp * jsd_norm)
    jsd_ref.backward()

    hs_chk = s_proj_chk(x_in)
    _, jsd_chk = chunked_kl_ce_backward(
        student_hidden=hs_chk,
        teacher_hidden=t_hidden,
        student_lm_head=head,
        teacher_lm_head=t_head,
        input_ids=input_ids,
        kl_weight=1.0,
        ce_weight=0.0,
        temperature=temp,
        chunk_tokens=4,
        grad_accum=1,
        kl_type="jsd",
        jsd_beta=jsd_beta,
        token_mask=token_mask,
    )
    assert jsd_chk == pytest.approx(jsd_ref.item(), rel=1e-5, abs=1e-5)
    assert torch.allclose(s_proj_chk.weight.grad, s_proj_ref.weight.grad, atol=1e-5, rtol=1e-5)


def test_cross_arch_hidden_projector_maps_layers_and_backpropagates() -> None:
    pairs = compute_matched_layer_pairs(student_layers=28, teacher_layers=36, num_taps=6)
    assert pairs == [(5, 6), (9, 12), (14, 18), (19, 24), (23, 30), (28, 36)]

    torch.manual_seed(5)
    proj = CrossArchHiddenProjector(student_dim=32, teacher_dim=48, num_taps=len(pairs), dtype=torch.float32)
    s_base = nn.Parameter(torch.randn(2, 6, 32))
    s_hiddens = tuple(s_base * (1.0 + 0.01 * i) for i in range(29))
    t_hiddens = tuple(torch.randn(2, 6, 48) for _ in range(37))

    loss = compute_projected_hidden_loss(s_hiddens, t_hiddens, pairs, proj, norm_weight=0.2)
    loss.backward()
    assert s_base.grad is not None and s_base.grad.abs().sum() > 0
    assert all(p.weight.grad is not None and p.weight.grad.abs().sum() > 0 for p in proj.projs)

    # Closed-form ridge initialization must sharply reduce initial cross-arch hidden alignment loss
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(17)
    s_mod = Qwen3ForCausalLM(
        Qwen3Config(vocab_size=64, hidden_size=24, intermediate_size=48, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=6)
    ).eval()
    t_mod = Qwen3ForCausalLM(
        Qwen3Config(vocab_size=64, hidden_size=36, intermediate_size=72, num_hidden_layers=6, num_attention_heads=6, num_key_value_heads=2, head_dim=6)
    ).eval()
    small_pairs = compute_matched_layer_pairs(4, 6, num_taps=3)
    small_proj = CrossArchHiddenProjector(24, 36, len(small_pairs), dtype=torch.float32)
    calib = [torch.randint(0, 64, (4, 16)) for _ in range(3)]
    before, after = small_proj.fit_ridge_(s_mod, t_mod, small_pairs, calib, device="cpu", ridge=0.05)
    assert after < before * 0.5, f"Expected ridge init to slash projector loss, got {before:.4f} -> {after:.4f}"


def test_cpu_weight_ema_averages_deduplicated_params_and_restores() -> None:
    w_shared = nn.Parameter(torch.tensor([1.0, 2.0]))
    m = nn.Module()
    m.register_parameter("a", w_shared)
    m.register_parameter("b", w_shared)  # Tied duplicate must be tracked once
    m.register_parameter("c", nn.Parameter(torch.tensor([10.0, 20.0])))

    ema = CPUWeightEMA(decay=0.0)  # Uniform SWA
    ema.update(m)
    assert len(ema.shadow) == 2

    with torch.no_grad():
        m.a.copy_(torch.tensor([3.0, 4.0]))
        m.c.copy_(torch.tensor([30.0, 40.0]))
    ema.update(m)

    backup = ema.swap_into(m)
    assert torch.allclose(m.a, torch.tensor([2.0, 3.0]))
    assert torch.allclose(m.c, torch.tensor([20.0, 30.0]))

    ema.restore(m, backup)
    assert torch.allclose(m.a, torch.tensor([3.0, 4.0]))
    assert torch.allclose(m.c, torch.tensor([30.0, 40.0]))


