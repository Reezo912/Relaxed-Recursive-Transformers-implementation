from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from rrt.kernels import TritonRMSNorm, triton_swiglu
from rrt.layers import LoRAMode, RMSNorm, RelaxedLinear, apply_rope, precompute_rope_freqs


@dataclass
class RRTConfig:
    vocab_size: int = 256
    d_model: int = 256
    n_heads: int = 4
    d_ff: int | None = None
    max_seq_len: int = 256
    num_layers: int = 12
    num_recursions: int = 4
    sharing_strategy: Literal["cycle", "middle_cycle"] = "cycle"
    lora_rank: int = 16
    lora_alpha: float | None = None
    lora_dropout: float = 0.0
    lora_mode: LoRAMode = "auto"
    fuse_projections: bool = False
    use_triton_kernels: bool = False
    dropout: float = 0.0
    share_norms: bool = False
    tie_embeddings: bool = True
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("head_dim (d_model // n_heads) must be even for RoPE")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if self.num_recursions < 1:
            raise ValueError(f"num_recursions must be >= 1, got {self.num_recursions}")
        if self.sharing_strategy not in ("cycle", "middle_cycle"):
            raise ValueError(f"Unsupported sharing_strategy: {self.sharing_strategy}")

        if self.d_ff is None:
            # Standard SwiGLU hidden width (~8/3 * d_model, rounded to multiple of 16)
            raw_ff = int(8 * self.d_model / 3)
            self.d_ff = ((raw_ff + 15) // 16) * 16

        if self.num_recursions > 1:
            if self.sharing_strategy == "cycle":
                if self.num_layers % self.num_recursions != 0:
                    raise ValueError(
                        f"For 'cycle' sharing, num_layers ({self.num_layers}) must be "
                        f"divisible by num_recursions ({self.num_recursions})"
                    )
            elif self.sharing_strategy == "middle_cycle":
                if self.num_layers < 2 + self.num_recursions or (self.num_layers - 2) % self.num_recursions != 0:
                    raise ValueError(
                        f"For 'middle_cycle' sharing, (num_layers - 2) must be a positive multiple "
                        f"of num_recursions; got num_layers={self.num_layers}, num_recursions={self.num_recursions}"
                    )


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_recursions: int,
        lora_rank: int,
        lora_alpha: float | None,
        lora_dropout: float,
        attn_dropout: float,
        lora_mode: LoRAMode = "auto",
        fuse_projections: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attn_dropout = attn_dropout
        self.fuse_projections = fuse_projections

        linear_kwargs = dict(
            num_recursions=num_recursions,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=False,
            lora_mode=lora_mode,
        )
        if fuse_projections:
            self.qkv_proj = RelaxedLinear(d_model, 3 * d_model, **linear_kwargs)
        else:
            self.q_proj = RelaxedLinear(d_model, d_model, **linear_kwargs)
            self.k_proj = RelaxedLinear(d_model, d_model, **linear_kwargs)
            self.v_proj = RelaxedLinear(d_model, d_model, **linear_kwargs)
        self.o_proj = RelaxedLinear(d_model, d_model, **linear_kwargs)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, recursion_idx: int = 0) -> Tensor:
        bsz, seq_len, _ = x.shape

        if self.fuse_projections:
            qkv = self.qkv_proj(x, recursion_idx)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        else:
            q = self.q_proj(x, recursion_idx).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x, recursion_idx).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x, recursion_idx).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        drop_p = self.attn_dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.o_proj(out, recursion_idx)


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_recursions: int,
        lora_rank: int,
        lora_alpha: float | None,
        lora_dropout: float,
        lora_mode: LoRAMode = "auto",
        fuse_projections: bool = False,
        use_triton_kernels: bool = False,
    ) -> None:
        super().__init__()
        self.fuse_projections = fuse_projections
        self.use_triton_kernels = use_triton_kernels
        linear_kwargs = dict(
            num_recursions=num_recursions,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=False,
            lora_mode=lora_mode,
        )
        if fuse_projections:
            self.gate_up_proj = RelaxedLinear(d_model, 2 * d_ff, **linear_kwargs)
        else:
            self.gate_proj = RelaxedLinear(d_model, d_ff, **linear_kwargs)
            self.up_proj = RelaxedLinear(d_model, d_ff, **linear_kwargs)
        self.down_proj = RelaxedLinear(d_ff, d_model, **linear_kwargs)

    def forward(self, x: Tensor, recursion_idx: int = 0) -> Tensor:
        if self.fuse_projections:
            gate, up = self.gate_up_proj(x, recursion_idx).chunk(2, dim=-1)
        else:
            gate = self.gate_proj(x, recursion_idx)
            up = self.up_proj(x, recursion_idx)

        if self.use_triton_kernels:
            gated = triton_swiglu(gate, up)
        else:
            gated = F.silu(gate) * up
        return self.down_proj(gated, recursion_idx)


class RecursiveTransformerBlock(nn.Module):
    """A single unique Transformer block that can be invoked across multiple recursion steps."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_recursions: int,
        lora_rank: int,
        lora_alpha: float | None,
        lora_dropout: float,
        dropout: float,
        share_norms: bool,
        lora_mode: LoRAMode = "auto",
        fuse_projections: bool = False,
        use_triton_kernels: bool = False,
    ) -> None:
        super().__init__()
        self.num_recursions = num_recursions
        self.share_norms = share_norms
        n_norms = 1 if share_norms else num_recursions
        norm_cls = TritonRMSNorm if use_triton_kernels else RMSNorm

        self.attn_norms = nn.ModuleList([norm_cls(d_model) for _ in range(n_norms)])
        self.ffn_norms = nn.ModuleList([norm_cls(d_model) for _ in range(n_norms)])

        self.attn = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            num_recursions=num_recursions,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            attn_dropout=dropout,
            lora_mode=lora_mode,
            fuse_projections=fuse_projections,
        )
        self.ffn = SwiGLUFFN(
            d_model=d_model,
            d_ff=d_ff,
            num_recursions=num_recursions,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_mode=lora_mode,
            fuse_projections=fuse_projections,
            use_triton_kernels=use_triton_kernels,
        )
        self.resid_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, recursion_idx: int = 0) -> Tensor:
        norm_idx = 0 if self.share_norms else recursion_idx
        h = x + self.resid_dropout(self.attn(self.attn_norms[norm_idx](x), cos, sin, recursion_idx))
        h = h + self.resid_dropout(self.ffn(self.ffn_norms[norm_idx](h), recursion_idx))
        return h

    def relaxed_linears(self) -> list[RelaxedLinear]:
        return [m for m in self.modules() if isinstance(m, RelaxedLinear)]


class RelaxedRecursiveTransformer(nn.Module):
    """Decoder-only Transformer supporting standard, strict recursive, and relaxed recursive modes."""

    def __init__(self, config: RRTConfig) -> None:
        super().__init__()
        self.config = config
        assert config.d_ff is not None

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

        blocks: list[RecursiveTransformerBlock] = []
        schedule: list[tuple[int, int]] = []

        def make_block(num_recursions: int, lora_rank: int) -> RecursiveTransformerBlock:
            return RecursiveTransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,  # type: ignore[arg-type]
                num_recursions=num_recursions,
                lora_rank=lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                dropout=config.dropout,
                share_norms=config.share_norms,
                lora_mode=config.lora_mode,
                fuse_projections=config.fuse_projections,
                use_triton_kernels=config.use_triton_kernels,
            )

        if config.num_recursions == 1:
            # Standard Transformer: every layer is unique, no recursion or LoRA needed.
            for layer_idx in range(config.num_layers):
                blocks.append(make_block(num_recursions=1, lora_rank=0))
                schedule.append((layer_idx, 0))
        elif config.sharing_strategy == "cycle":
            unique_layers = config.num_layers // config.num_recursions
            for _ in range(unique_layers):
                blocks.append(make_block(num_recursions=config.num_recursions, lora_rank=config.lora_rank))
            for r in range(config.num_recursions):
                for u in range(unique_layers):
                    schedule.append((u, r))
        else:
            # middle_cycle: dedicated first and last layer; middle block looped R times.
            unique_mid = (config.num_layers - 2) // config.num_recursions
            blocks.append(make_block(num_recursions=1, lora_rank=0))  # entry block
            for _ in range(unique_mid):
                blocks.append(make_block(num_recursions=config.num_recursions, lora_rank=config.lora_rank))
            blocks.append(make_block(num_recursions=1, lora_rank=0))  # exit block

            schedule.append((0, 0))
            for r in range(config.num_recursions):
                for u in range(unique_mid):
                    schedule.append((1 + u, r))
            schedule.append((1 + unique_mid, 0))

        self.blocks = nn.ModuleList(blocks)
        self.layer_schedule = schedule

        self.final_norm = TritonRMSNorm(config.d_model) if config.use_triton_kernels else RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        head_dim = config.d_model // config.n_heads
        cos, sin = precompute_rope_freqs(head_dim, config.max_seq_len, theta=config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._init_weights()

    def _init_weights(self) -> None:
        # GPT-style scaled init for base weights while preserving LoRA zero-init on B matrices.
        std = 0.02
        resid_std = std / (2.0 * self.config.num_layers) ** 0.5

        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=std)
        if not self.config.tie_embeddings:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=std)

        for block in self.blocks:
            for name, module in block.named_modules():
                if isinstance(module, RelaxedLinear):
                    target_std = resid_std if name.endswith(("o_proj", "down_proj")) else std
                    nn.init.normal_(module.weight, mean=0.0, std=target_std)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                    if module.lora_A is not None and module.lora_B is not None:
                        for a_mat, b_mat in zip(module.lora_A, module.lora_B):
                            nn.init.kaiming_uniform_(a_mat, a=5**0.5)
                            nn.init.zeros_(b_mat)

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        multi_exit_loss: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        _, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}")

        x = self.emb_dropout(self.tok_emb(input_ids))
        collect_exits = (
            multi_exit_loss
            and targets is not None
            and self.config.sharing_strategy == "cycle"
            and self.config.num_recursions > 1
        )
        unique_layers = self.config.num_layers // self.config.num_recursions if collect_exits else 0
        exit_losses: list[Tensor] = []

        for virtual_idx, (block_idx, recursion_idx) in enumerate(self.layer_schedule):
            x = self.blocks[block_idx](x, self.rope_cos, self.rope_sin, recursion_idx=recursion_idx)
            if collect_exits and (virtual_idx + 1) % unique_layers == 0 and (virtual_idx + 1) < self.config.num_layers:
                aux_logits = self.lm_head(self.final_norm(x))
                exit_losses.append(F.cross_entropy(aux_logits.view(-1, aux_logits.size(-1)), targets.view(-1)))

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            final_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            if exit_losses:
                # Weight deeper exits slightly higher while ensuring early exits produce calibrated logits
                all_losses = [*exit_losses, final_loss]
                weights = [float(i + 1) for i in range(len(all_losses))]
                loss = sum(w * l for w, l in zip(weights, all_losses)) / sum(weights)
            else:
                loss = final_loss
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Tensor:
        self.eval()
        out = input_ids
        for _ in range(max_new_tokens):
            ctx = out[:, -self.config.max_seq_len :]
            logits, _ = self(ctx)
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits = next_logits.masked_fill(next_logits < v[:, [-1]], float("-inf"))
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            out = torch.cat([out, next_token], dim=1)
        return out

    def parameter_summary(self) -> dict[str, int]:
        """Count unique parameters split by role (deduplicating tied weights)."""
        seen: set[int] = set()
        counts = {"embeddings": 0, "shared_base": 0, "lora": 0, "norms": 0, "total": 0}

        for name, param in self.named_parameters():
            pid = id(param)
            if pid in seen:
                continue
            seen.add(pid)
            n = param.numel()
            counts["total"] += n
            if "tok_emb" in name or "lm_head" in name:
                counts["embeddings"] += n
            elif "lora_A" in name or "lora_B" in name:
                counts["lora"] += n
            elif "norm" in name:
                counts["norms"] += n
            else:
                counts["shared_base"] += n
        return counts

    @torch.no_grad()
    def lora_relaxation_stats(self) -> list[dict[str, float | int]]:
        """Report mean relative LoRA norm per virtual layer to inspect depth specialization."""
        stats: list[dict[str, float | int]] = []
        for virtual_layer, (block_idx, recursion_idx) in enumerate(self.layer_schedule):
            block = self.blocks[block_idx]
            ratios = [lin.relative_lora_norm(recursion_idx) for lin in block.relaxed_linears()]
            mean_ratio = sum(ratios) / len(ratios) if ratios else 0.0
            stats.append(
                {
                    "virtual_layer": virtual_layer,
                    "block_idx": block_idx,
                    "recursion_idx": recursion_idx,
                    "mean_rel_lora_norm": mean_ratio,
                }
            )
        return stats
