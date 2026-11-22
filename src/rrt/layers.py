from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

LoRAMode = Literal["auto", "split", "merge"]


def lora_merge_token_threshold(in_features: int, out_features: int, factor: float = 1.25) -> int:
    """Analytical token-count threshold M where merge-on-the-fly beats split LoRA.

    Split LoRA costs 2 * M * r * (D_in + D_out) extra FLOPs; merging B @ A costs
    2 * D_in * D_out * r FLOPs. They break even at M = D_in * D_out / (D_in + D_out).
    A slight 1.25x safety factor accounts for writing the temporary W_eff matrix.
    """
    return max(1, int(factor * (in_features * out_features) / (in_features + out_features)))


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        normed = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normed * self.weight.float()).to(x.dtype)


class RelaxedLinear(nn.Module):
    """Linear layer with a shared base weight and per-recursion LoRA adapters.

    At recursion step k, computes:
        y = x @ W^T + (alpha / r) * (x @ A_k^T) @ B_k^T

    When lora_rank == 0, reduces to a strictly shared linear layer.
    When B_k is zero-initialized, the relaxed layer starts out identical to
    the strict recursive layer and progressively specializes per depth.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_recursions: int = 1,
        lora_rank: int = 0,
        lora_alpha: float | None = None,
        lora_dropout: float = 0.0,
        bias: bool = False,
        lora_mode: LoRAMode = "auto",
    ) -> None:
        super().__init__()
        if num_recursions < 1:
            raise ValueError(f"num_recursions must be >= 1, got {num_recursions}")
        if lora_rank < 0:
            raise ValueError(f"lora_rank must be >= 0, got {lora_rank}")
        if lora_mode not in ("auto", "split", "merge"):
            raise ValueError(f"Unsupported lora_mode: {lora_mode}")

        self.in_features = in_features
        self.out_features = out_features
        self.num_recursions = num_recursions
        self.lora_rank = lora_rank
        self.lora_alpha = float(lora_rank) if (lora_alpha is None and lora_rank > 0) else float(lora_alpha or 1.0)
        self.scaling = self.lora_alpha / lora_rank if lora_rank > 0 else 0.0
        self.lora_dropout_p = lora_dropout
        self.lora_mode: LoRAMode = lora_mode
        self._merge_threshold = lora_merge_token_threshold(in_features, out_features)

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        self.dropout = nn.Dropout(lora_dropout) if (lora_rank > 0 and lora_dropout > 0.0) else nn.Identity()

        if lora_rank > 0:
            self.lora_A = nn.ParameterList(
                [nn.Parameter(torch.empty(lora_rank, in_features)) for _ in range(num_recursions)]
            )
            self.lora_B = nn.ParameterList(
                [nn.Parameter(torch.empty(out_features, lora_rank)) for _ in range(num_recursions)]
            )
        else:
            self.lora_A = None
            self.lora_B = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

        if self.lora_A is not None and self.lora_B is not None:
            for a_mat, b_mat in zip(self.lora_A, self.lora_B):
                nn.init.kaiming_uniform_(a_mat, a=math.sqrt(5))
                nn.init.zeros_(b_mat)

    def _should_merge(self, num_tokens: int) -> bool:
        if self.lora_dropout_p > 0.0 and self.training:
            return False
        if self.lora_mode == "merge":
            return True
        if self.lora_mode == "split":
            return False
        return num_tokens >= self._merge_threshold

    def forward(self, x: Tensor, recursion_idx: int = 0) -> Tensor:
        if not (0 <= recursion_idx < self.num_recursions):
            raise IndexError(
                f"recursion_idx={recursion_idx} out of range for num_recursions={self.num_recursions}"
            )

        if self.lora_A is None or self.lora_B is None:
            return F.linear(x, self.weight, self.bias)

        num_tokens = x.numel() // self.in_features
        if self._should_merge(num_tokens):
            w_eff = self.weight + self.scaling * (self.lora_B[recursion_idx] @ self.lora_A[recursion_idx])
            return F.linear(x, w_eff, self.bias)

        out = F.linear(x, self.weight, self.bias)
        delta = F.linear(
            F.linear(self.dropout(x), self.lora_A[recursion_idx]),
            self.lora_B[recursion_idx],
        )
        return out + self.scaling * delta

    @torch.no_grad()
    def effective_weight(self, recursion_idx: int = 0) -> Tensor:
        """Materialize W + (alpha / r) * (B_k @ A_k) for inspection or merging."""
        if not (0 <= recursion_idx < self.num_recursions):
            raise IndexError(
                f"recursion_idx={recursion_idx} out of range for num_recursions={self.num_recursions}"
            )
        if self.lora_A is None or self.lora_B is None:
            return self.weight.detach().clone()
        delta = self.lora_B[recursion_idx] @ self.lora_A[recursion_idx]
        return self.weight.detach() + self.scaling * delta

    @torch.no_grad()
    def relative_lora_norm(self, recursion_idx: int = 0) -> float:
        """Ratio ||scaling * B_k @ A_k||_F / ||W||_F for measuring depth specialization."""
        if self.lora_A is None or self.lora_B is None:
            return 0.0
        base_norm = self.weight.float().norm().item()
        if base_norm == 0.0:
            return 0.0
        delta_norm = (self.scaling * (self.lora_B[recursion_idx].float() @ self.lora_A[recursion_idx].float())).norm().item()
        return delta_norm / base_norm


def precompute_rope_freqs(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """Precompute cosine and sine tables for Rotary Position Embeddings (RoPE)."""
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(positions, freqs)
    return angles.cos(), angles.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE to tensor x of shape (batch, n_heads, seq_len, head_dim)."""
    seq_len = x.size(2)
    half_dim = x.size(-1) // 2

    cos_t = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D/2)
    sin_t = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    x_f = x.float()
    x1 = x_f[..., :half_dim]
    x2 = x_f[..., half_dim:]

    rotated = torch.cat((x1 * cos_t - x2 * sin_t, x1 * sin_t + x2 * cos_t), dim=-1)
    return rotated.to(x.dtype)
