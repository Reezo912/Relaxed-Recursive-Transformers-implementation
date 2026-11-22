from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor

from rrt.layers import RelaxedLinear
from rrt.model import RRTConfig, RelaxedRecursiveTransformer

BaseInitMethod = Literal["mean", "first", "middle", "last"]


@dataclass
class MatrixDecompositionStats:
    name: str
    rel_error_strict: float
    rel_error_relaxed: float
    energy_captured: float


@dataclass
class ConversionReport:
    base_init: BaseInitMethod
    use_svd: bool
    lora_rank: int
    sharing_strategy: str
    num_recursions: int
    matrix_stats: list[MatrixDecompositionStats] = field(default_factory=list)

    @property
    def mean_rel_error_strict(self) -> float:
        if not self.matrix_stats:
            return 0.0
        return sum(m.rel_error_strict for m in self.matrix_stats) / len(self.matrix_stats)

    @property
    def mean_rel_error_relaxed(self) -> float:
        if not self.matrix_stats:
            return 0.0
        return sum(m.rel_error_relaxed for m in self.matrix_stats) / len(self.matrix_stats)

    @property
    def mean_energy_captured(self) -> float:
        if not self.matrix_stats:
            return 0.0
        return sum(m.energy_captured for m in self.matrix_stats) / len(self.matrix_stats)


@torch.no_grad()
def compute_base_weight(weights: list[Tensor], method: BaseInitMethod = "mean") -> Tensor:
    """Aggregate a group of tied layer weights into a single shared base matrix."""
    if not weights:
        raise ValueError("weights list must be non-empty")
    if method == "mean":
        return torch.stack([w.float() for w in weights], dim=0).mean(dim=0)
    if method == "first":
        return weights[0].float().clone()
    if method == "middle":
        return weights[len(weights) // 2].float().clone()
    if method == "last":
        return weights[-1].float().clone()
    raise ValueError(f"Unsupported base_init method: {method}")


@torch.no_grad()
def decompose_weight_group_svd(
    weights: list[Tensor],
    lora_rank: int,
    scaling: float,
    base_init: BaseInitMethod = "mean",
    use_svd: bool = True,
) -> tuple[Tensor, list[Tensor], list[Tensor], float, float, float]:
    """Decompose a list of matrices W_0..W_{R-1} into shared W_base + per-step LoRA (A_k, B_k).

    For each step k, computes residual Delta_k = W_k - W_base and its optimal rank-r
    approximation via truncated SVD:
        Delta_k ≈ U[:, :r] @ diag(S[:r]) @ Vh[:r, :] = scaling * (B_k @ A_k)
    splitting sqrt(S) evenly across A_k and B_k for well-conditioned training.

    Returns:
        (W_base, lora_A_list, lora_B_list, mean_rel_err_strict, mean_rel_err_relaxed, mean_energy)
    """
    w_base = compute_base_weight(weights, method=base_init)
    out_features, in_features = w_base.shape

    lora_a_list: list[Tensor] = []
    lora_b_list: list[Tensor] = []
    err_strict_list: list[float] = []
    err_relaxed_list: list[float] = []
    energy_list: list[float] = []

    for w in weights:
        w_f = w.float()
        w_norm = w_f.norm().item()
        residual = w_f - w_base
        strict_err = residual.norm().item() / max(w_norm, 1e-12)
        err_strict_list.append(strict_err)

        if lora_rank == 0:
            err_relaxed_list.append(strict_err)
            energy_list.append(0.0)
            continue

        if not use_svd:
            err_relaxed_list.append(strict_err)
            energy_list.append(0.0)
            continue

        if scaling <= 0.0:
            raise ValueError(f"scaling must be > 0 when lora_rank > 0, got {scaling}")

        u, s, vh = torch.linalg.svd(residual, full_matrices=False)
        r_eff = min(lora_rank, s.numel())

        s_sq_total = s.pow(2).sum().item()
        s_sq_kept = s[:r_eff].pow(2).sum().item()
        energy = (s_sq_kept / s_sq_total) if s_sq_total > 1e-16 else 1.0
        energy_list.append(energy)

        inv_sqrt_scale = 1.0 / (scaling**0.5)
        sqrt_s = s[:r_eff].sqrt() * inv_sqrt_scale

        b_mat = torch.zeros((out_features, lora_rank), dtype=torch.float32, device=w_f.device)
        a_mat = torch.zeros((lora_rank, in_features), dtype=torch.float32, device=w_f.device)
        b_mat[:, :r_eff] = u[:, :r_eff] * sqrt_s.unsqueeze(0)
        a_mat[:r_eff, :] = sqrt_s.unsqueeze(1) * vh[:r_eff, :]

        reconstructed = w_base + scaling * (b_mat @ a_mat)
        relaxed_err = (w_f - reconstructed).norm().item() / max(w_norm, 1e-12)
        err_relaxed_list.append(relaxed_err)

        lora_a_list.append(a_mat)
        lora_b_list.append(b_mat)

    n = len(weights)
    return (
        w_base,
        lora_a_list,
        lora_b_list,
        sum(err_strict_list) / n,
        sum(err_relaxed_list) / n,
        sum(energy_list) / n,
    )


def _named_relaxed_linears(block: torch.nn.Module) -> list[tuple[str, RelaxedLinear]]:
    return [(name, mod) for name, mod in block.named_modules() if isinstance(mod, RelaxedLinear)]


@torch.no_grad()
def convert_standard_to_relaxed(
    source: RelaxedRecursiveTransformer,
    num_recursions: int,
    sharing_strategy: Literal["cycle", "middle_cycle"] = "cycle",
    lora_rank: int = 16,
    lora_alpha: float | None = None,
    lora_dropout: float = 0.0,
    base_init: BaseInitMethod = "mean",
    use_svd: bool = True,
    share_norms: bool = False,
) -> tuple[RelaxedRecursiveTransformer, ConversionReport]:
    """Convert a trained standard Transformer into a Relaxed Recursive Transformer.

    Implements the weight-tying + truncated-SVD LoRA initialization from
    Bae et al. (2024), 'Relaxed Recursive Transformers'.
    """
    if source.config.num_recursions != 1:
        raise ValueError(
            f"source model must be a standard (non-recursive) Transformer with num_recursions=1, "
            f"got num_recursions={source.config.num_recursions}"
        )

    target_cfg = RRTConfig(
        vocab_size=source.config.vocab_size,
        d_model=source.config.d_model,
        n_heads=source.config.n_heads,
        d_ff=source.config.d_ff,
        max_seq_len=source.config.max_seq_len,
        num_layers=source.config.num_layers,
        num_recursions=num_recursions,
        sharing_strategy=sharing_strategy,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_mode=source.config.lora_mode,
        fuse_projections=source.config.fuse_projections,
        dropout=source.config.dropout,
        share_norms=share_norms,
        tie_embeddings=source.config.tie_embeddings,
        rope_theta=source.config.rope_theta,
    )

    device = source.tok_emb.weight.device
    dtype = source.tok_emb.weight.dtype
    target = RelaxedRecursiveTransformer(target_cfg).to(device=device, dtype=dtype)

    # Copy embeddings, final norm, and lm_head
    target.tok_emb.weight.copy_(source.tok_emb.weight)
    target.final_norm.weight.copy_(source.final_norm.weight)
    if not target_cfg.tie_embeddings:
        target.lm_head.weight.copy_(source.lm_head.weight)

    # Group source virtual layer indices by target physical block_idx
    groups: dict[int, list[tuple[int, int]]] = {i: [] for i in range(len(target.blocks))}
    for virtual_layer, (block_idx, recursion_idx) in enumerate(target.layer_schedule):
        groups[block_idx].append((virtual_layer, recursion_idx))

    report = ConversionReport(
        base_init=base_init,
        use_svd=use_svd,
        lora_rank=lora_rank,
        sharing_strategy=sharing_strategy,
        num_recursions=num_recursions,
    )

    for block_idx, entries in groups.items():
        # Sort by recursion_idx to align 0..R-1 cleanly
        entries.sort(key=lambda pair: pair[1])
        src_blocks = [source.blocks[v_layer] for v_layer, _ in entries]
        tgt_block = target.blocks[block_idx]

        # Copy RMSNorms (exact per-depth copy when share_norms=False, else mean)
        if share_norms:
            mean_attn_norm = torch.stack([b.attn_norms[0].weight.float() for b in src_blocks], dim=0).mean(dim=0)
            mean_ffn_norm = torch.stack([b.ffn_norms[0].weight.float() for b in src_blocks], dim=0).mean(dim=0)
            tgt_block.attn_norms[0].weight.copy_(mean_attn_norm.to(dtype=dtype))
            tgt_block.ffn_norms[0].weight.copy_(mean_ffn_norm.to(dtype=dtype))
        else:
            for src_b, (_, rec_idx) in zip(src_blocks, entries):
                tgt_block.attn_norms[rec_idx].weight.copy_(src_b.attn_norms[0].weight)
                tgt_block.ffn_norms[rec_idx].weight.copy_(src_b.ffn_norms[0].weight)

        # Copy or decompose each linear projection
        tgt_linears = _named_relaxed_linears(tgt_block)
        src_linears_per_step = [_named_relaxed_linears(b) for b in src_blocks]

        for proj_i, (proj_name, tgt_lin) in enumerate(tgt_linears):
            src_weights = [step_linears[proj_i][1].weight for step_linears in src_linears_per_step]

            if len(src_weights) == 1:
                # Dedicated unshared block (e.g., entry/exit layer in middle_cycle)
                tgt_lin.weight.copy_(src_weights[0])
                continue

            w_base, lora_as, lora_bs, err_s, err_r, energy = decompose_weight_group_svd(
                weights=src_weights,
                lora_rank=tgt_lin.lora_rank,
                scaling=tgt_lin.scaling,
                base_init=base_init,
                use_svd=use_svd,
            )
            tgt_lin.weight.copy_(w_base.to(dtype=dtype))

            if use_svd and tgt_lin.lora_A is not None and tgt_lin.lora_B is not None:
                for rec_idx, (a_mat, b_mat) in enumerate(zip(lora_as, lora_bs)):
                    tgt_lin.lora_A[rec_idx].copy_(a_mat.to(dtype=dtype))
                    tgt_lin.lora_B[rec_idx].copy_(b_mat.to(dtype=dtype))

            report.matrix_stats.append(
                MatrixDecompositionStats(
                    name=f"block{block_idx}.{proj_name}",
                    rel_error_strict=err_s,
                    rel_error_relaxed=err_r,
                    energy_captured=energy,
                )
            )

    return target, report
