from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from rrt import (
    RRTConfig,
    RelaxedRecursiveTransformer,
    convert_standard_to_relaxed,
    decompose_weight_group_svd,
)


@pytest.mark.parametrize("sharing_strategy,num_layers,num_recursions", [
    ("cycle", 8, 4),
    ("middle_cycle", 10, 4),
])
def test_svd_full_rank_exact_reconstruction(
    sharing_strategy: str,
    num_layers: int,
    num_recursions: int,
) -> None:
    """When lora_rank >= min(d_out, d_in), truncated SVD must reconstruct the standard model exactly."""
    torch.manual_seed(0)
    d_model = 32
    d_ff = 64
    src_cfg = RRTConfig(
        vocab_size=48,
        d_model=d_model,
        n_heads=4,
        d_ff=d_ff,
        max_seq_len=24,
        num_layers=num_layers,
        num_recursions=1,
        lora_rank=0,
    )
    source = RelaxedRecursiveTransformer(src_cfg)
    source.eval()

    # min(out, in) across all linear layers is d_model=32, so rank=32 is full rank
    converted, report = convert_standard_to_relaxed(
        source=source,
        num_recursions=num_recursions,
        sharing_strategy=sharing_strategy,  # type: ignore[arg-type]
        lora_rank=d_model,
        base_init="mean",
        use_svd=True,
    )
    converted.eval()

    assert report.mean_rel_error_relaxed < 1e-5
    assert report.mean_energy_captured > 0.9999

    x = torch.randint(0, 48, (2, 16))
    y = torch.randint(0, 48, (2, 16))
    logits_src, loss_src = source(x, y)
    logits_cvt, loss_cvt = converted(x, y)

    assert loss_src is not None and loss_cvt is not None
    assert torch.allclose(logits_src, logits_cvt, atol=1e-4)
    assert torch.allclose(loss_src, loss_cvt, atol=1e-4)


def test_svd_monotonic_error_reduction_and_mean_optimality() -> None:
    torch.manual_seed(1)
    weights = [torch.randn(64, 48) for _ in range(4)]

    # Mean minimizes Frobenius residual compared to picking the first matrix
    _, _, _, err_mean, _, _ = decompose_weight_group_svd(weights, lora_rank=0, scaling=1.0, base_init="mean")
    _, _, _, err_first, _, _ = decompose_weight_group_svd(weights, lora_rank=0, scaling=1.0, base_init="first")
    assert err_mean < err_first

    prev_err = float("inf")
    prev_energy = -1.0
    for rank in (0, 4, 16, 32, 48):
        _, _, _, _, err_relaxed, energy = decompose_weight_group_svd(
            weights,
            lora_rank=rank,
            scaling=1.0,
            base_init="mean",
            use_svd=True,
        )
        assert err_relaxed <= prev_err + 1e-6
        assert energy >= prev_energy - 1e-6
        prev_err = err_relaxed
        prev_energy = energy

    assert prev_err < 1e-5
    assert prev_energy > 0.9999


def test_svd_init_closer_to_source_than_zero_init() -> None:
    torch.manual_seed(42)
    src_cfg = RRTConfig(
        vocab_size=64,
        d_model=64,
        n_heads=4,
        d_ff=128,
        max_seq_len=32,
        num_layers=8,
        num_recursions=1,
        lora_rank=0,
    )
    source = RelaxedRecursiveTransformer(src_cfg)
    source.eval()

    strict, _ = convert_standard_to_relaxed(source, num_recursions=4, lora_rank=0, use_svd=False)
    zero_init, _ = convert_standard_to_relaxed(source, num_recursions=4, lora_rank=16, use_svd=False)
    svd_init, report = convert_standard_to_relaxed(source, num_recursions=4, lora_rank=16, use_svd=True)

    strict.eval()
    zero_init.eval()
    svd_init.eval()

    x = torch.randint(0, 64, (4, 24))
    logits_src, _ = source(x)
    logits_strict, _ = strict(x)
    logits_zero, _ = zero_init(x)
    logits_svd, _ = svd_init(x)

    # Zero-init LoRA equals strict recursive at step 0
    assert torch.allclose(logits_strict, logits_zero, atol=1e-5)

    # SVD-init LoRA is strictly closer to the source standard model
    mse_zero = F.mse_loss(logits_zero, logits_src).item()
    mse_svd = F.mse_loss(logits_svd, logits_src).item()
    assert mse_svd < mse_zero
    assert report.mean_rel_error_relaxed < report.mean_rel_error_strict


def test_middle_cycle_preserves_entry_and_exit_layers_verbatim() -> None:
    torch.manual_seed(9)
    src_cfg = RRTConfig(
        vocab_size=32,
        d_model=32,
        n_heads=4,
        max_seq_len=16,
        num_layers=10,
        num_recursions=1,
        lora_rank=0,
    )
    source = RelaxedRecursiveTransformer(src_cfg)
    converted, _ = convert_standard_to_relaxed(
        source=source,
        num_recursions=4,
        sharing_strategy="middle_cycle",
        lora_rank=8,
        use_svd=True,
    )

    # First block (entry) and last block (exit) must be exact copies of source layers 0 and 9
    assert torch.equal(converted.blocks[0].attn.q_proj.weight, source.blocks[0].attn.q_proj.weight)
    assert torch.equal(converted.blocks[-1].ffn.down_proj.weight, source.blocks[-1].ffn.down_proj.weight)
    assert torch.equal(converted.blocks[0].attn_norms[0].weight, source.blocks[0].attn_norms[0].weight)
    assert torch.equal(converted.blocks[-1].ffn_norms[0].weight, source.blocks[-1].ffn_norms[0].weight)
