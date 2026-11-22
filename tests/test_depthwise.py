from __future__ import annotations

import pytest
import torch

from rrt import ContinuousDepthwiseEngine, RRTConfig, RelaxedRecursiveTransformer


@pytest.mark.parametrize("fuse_projections", [False, True])
def test_depthwise_engine_full_depth_matches_standard_greedy_decode(fuse_projections: bool) -> None:
    """With exit_threshold > 1.0 (no early exit), ContinuousDepthwiseEngine must match model.forward()."""
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = RRTConfig(
        vocab_size=64,
        d_model=64,
        n_heads=4,
        max_seq_len=48,
        num_layers=8,
        num_recursions=4,
        sharing_strategy="cycle",
        lora_rank=16,
        fuse_projections=fuse_projections,
    )
    model = RelaxedRecursiveTransformer(cfg).to(device)
    # Give LoRA B matrices non-zero weights so every recursion step is genuinely distinct
    with torch.no_grad():
        for blk in model.blocks:
            for lin in blk.relaxed_linears():
                if lin.lora_B is not None:
                    for b_mat in lin.lora_B:
                        b_mat.normal_(0.0, 0.05)
    model.eval()

    prompt = torch.randint(0, 64, (3, 6), device=device)
    max_new = 8

    # Reference greedy decode via full model.forward()
    ref_seq = prompt.clone()
    with torch.no_grad():
        for _ in range(max_new):
            logits, _ = model(ref_seq)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ref_seq = torch.cat([ref_seq, next_tok], dim=1)

    engine = ContinuousDepthwiseEngine(model)
    out_seqs, stats = engine.generate(prompt, max_new_tokens=max_new, exit_threshold=1.01, temperature=0.0)

    assert out_seqs == ref_seq.tolist()
    assert stats.total_tokens_generated == 3 * max_new
    assert stats.total_loop_iterations == max_new * cfg.num_recursions
    assert stats.mean_exit_depth == pytest.approx(float(cfg.num_recursions))
    assert stats.exit_depth_histogram == [0, 0, 0, 3 * max_new]


def test_depthwise_engine_early_exit_reduces_iterations_and_mixes_depths() -> None:
    """With exit_threshold < 1.0, easy tokens exit early and reduce total loop iterations."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = RRTConfig(
        vocab_size=32,
        d_model=64,
        n_heads=4,
        max_seq_len=48,
        num_layers=8,
        num_recursions=4,
        sharing_strategy="cycle",
        lora_rank=8,
        fuse_projections=True,
    )
    model = RelaxedRecursiveTransformer(cfg).to(device)
    model.eval()

    prompt = torch.randint(0, 32, (8, 5), device=device)
    engine = ContinuousDepthwiseEngine(model)

    # Very low threshold triggers early exit on some/all tokens immediately
    seqs, stats = engine.generate(prompt, max_new_tokens=10, exit_threshold=0.04, temperature=0.0)
    assert len(seqs) == 8
    assert all(len(s) == 15 for s in seqs)
    assert stats.mean_exit_depth < cfg.num_recursions
    assert stats.total_loop_iterations < 10 * cfg.num_recursions


def test_multi_exit_training_loss_flows_gradients() -> None:
    torch.manual_seed(0)
    cfg = RRTConfig(
        vocab_size=48,
        d_model=64,
        n_heads=4,
        max_seq_len=32,
        num_layers=8,
        num_recursions=4,
        sharing_strategy="cycle",
        lora_rank=8,
    )
    model = RelaxedRecursiveTransformer(cfg)
    x = torch.randint(0, 48, (4, 16))
    y = torch.randint(0, 48, (4, 16))

    logits, loss = model(x, y, multi_exit_loss=True)
    assert logits.shape == (4, 16, 48)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert model.blocks[0].attn.q_proj.weight.grad is not None
    assert model.blocks[0].attn.q_proj.lora_B[0].grad is not None
    assert model.blocks[0].attn.q_proj.lora_B[3].grad is not None
