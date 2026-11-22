from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from rrt.kernels import (
    RecursionGroupMetadata,
    prepare_recursion_groups,
    triton_grouped_relaxed_linear,
    triton_swiglu,
)
from rrt.layers import RelaxedLinear
from rrt.model import RecursiveTransformerBlock, RelaxedRecursiveTransformer


@dataclass
class DepthwiseStats:
    total_tokens_generated: int
    total_loop_iterations: int
    elapsed_sec: float
    exit_depth_histogram: list[int]  # counts for exiting at depth 1..R

    @property
    def mean_exit_depth(self) -> float:
        total = sum(self.exit_depth_histogram)
        if total == 0:
            return 0.0
        return sum((i + 1) * c for i, c in enumerate(self.exit_depth_histogram)) / total

    @property
    def tokens_per_sec(self) -> float:
        return self.total_tokens_generated / max(self.elapsed_sec, 1e-6)


class _PackedRelaxedLinear:
    """Packs ParameterList (lora_A, lora_B) into contiguous (R, ...) tensors for Triton SGMV."""

    def __init__(self, linear: RelaxedLinear) -> None:
        self.weight = linear.weight.detach()
        self.bias = linear.bias.detach() if linear.bias is not None else None
        self.scaling = linear.scaling
        self.has_lora = linear.lora_A is not None and linear.lora_B is not None
        if self.has_lora:
            assert linear.lora_A is not None and linear.lora_B is not None
            self.lora_a = torch.stack([p.detach() for p in linear.lora_A], dim=0).contiguous()
            self.lora_b = torch.stack([p.detach() for p in linear.lora_B], dim=0).contiguous()
        else:
            self.lora_a = None
            self.lora_b = None

    def forward(self, x: Tensor, rec_ids: Tensor, meta: RecursionGroupMetadata | None) -> Tensor:
        if not self.has_lora:
            return F.linear(x, self.weight, self.bias)
        assert self.lora_a is not None and self.lora_b is not None
        return triton_grouped_relaxed_linear(
            x=x,
            weight=self.weight,
            lora_a=self.lora_a,
            lora_b=self.lora_b,
            rec_ids=rec_ids,
            scaling=self.scaling,
            bias=self.bias,
            group_meta=meta,
        )


class _PackedBlock:
    """Pre-packed view of a RecursiveTransformerBlock for zero-overhead depth-wise batching."""

    def __init__(self, block: RecursiveTransformerBlock) -> None:
        self.share_norms = block.share_norms
        self.n_heads = block.attn.n_heads
        self.head_dim = block.attn.head_dim
        self.d_model = block.attn.d_model
        self.fuse_attn = block.attn.fuse_projections
        self.fuse_ffn = block.ffn.fuse_projections
        self.eps = block.attn_norms[0].eps

        self.attn_norm_weights = torch.stack([n.weight.detach() for n in block.attn_norms], dim=0).contiguous()
        self.ffn_norm_weights = torch.stack([n.weight.detach() for n in block.ffn_norms], dim=0).contiguous()

        if self.fuse_attn:
            self.qkv_proj = _PackedRelaxedLinear(block.attn.qkv_proj)
        else:
            self.q_proj = _PackedRelaxedLinear(block.attn.q_proj)
            self.k_proj = _PackedRelaxedLinear(block.attn.k_proj)
            self.v_proj = _PackedRelaxedLinear(block.attn.v_proj)
        self.o_proj = _PackedRelaxedLinear(block.attn.o_proj)

        if self.fuse_ffn:
            self.gate_up_proj = _PackedRelaxedLinear(block.ffn.gate_up_proj)
        else:
            self.gate_proj = _PackedRelaxedLinear(block.ffn.gate_proj)
            self.up_proj = _PackedRelaxedLinear(block.ffn.up_proj)
        self.down_proj = _PackedRelaxedLinear(block.ffn.down_proj)


def _batched_rmsnorm(x: Tensor, norm_weights: Tensor, rec_ids: Tensor, share_norms: bool, eps: float) -> Tensor:
    """Apply per-slot RMSNorm where each token uses the norm weight for its recursion depth."""
    w = norm_weights[0].unsqueeze(0) if share_norms else norm_weights[rec_ids]
    x_f = x.float()
    normed = x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (normed * w.float()).to(x.dtype)


def _apply_rope_per_pos(x: Tensor, cos: Tensor, sin: Tensor, positions: Tensor) -> Tensor:
    """Apply RoPE to single-token states x of shape (B, H, head_dim) at per-slot positions (B,)."""
    half = x.size(-1) // 2
    cos_p = cos[positions].unsqueeze(1)  # (B, 1, D/2)
    sin_p = sin[positions].unsqueeze(1)
    x_f = x.float()
    x1, x2 = x_f[..., :half], x_f[..., half:]
    return torch.cat((x1 * cos_p - x2 * sin_p, x1 * sin_p + x2 * cos_p), dim=-1).to(x.dtype)


class ContinuousDepthwiseEngine:
    """Continuous Depth-Wise Batching inference engine (Bae et al., 2024, Section 4).

    All active batch slots execute each shared Transformer block together in a single
    GPU batch even when slots are at different recursion depths k_i in {0..R-1}:
      - Shared base weights W run as one dense GEMM across all slots.
      - Per-depth LoRA adapters (A_{k_i}, B_{k_i}) run via our Triton SGMV kernel
        reusing RecursionGroupMetadata across every projection in the block.
      - When a slot exits early at depth k < R-1 (confidence >= exit_threshold), its
        KV state is copied to deeper virtual layers (k+1..R-1), its next token is
        emitted immediately, and that slot starts depth 0 for the next token on the
        very next iteration (eliminating idle bubbles).
    """

    def __init__(self, model: RelaxedRecursiveTransformer) -> None:
        if model.config.sharing_strategy != "cycle":
            raise ValueError(
                f"ContinuousDepthwiseEngine currently requires sharing_strategy='cycle', "
                f"got '{model.config.sharing_strategy}'"
            )
        self.model = model
        self.config = model.config
        self.num_recursions = model.config.num_recursions
        self.num_unique_blocks = len(model.blocks)
        self.num_virtual_layers = model.config.num_layers
        self.device = model.tok_emb.weight.device
        self.dtype = model.tok_emb.weight.dtype

        self.packed_blocks = [_PackedBlock(b) for b in model.blocks]

    def _step_one_recursion_iteration(
        self,
        h: Tensor,
        slot_rec: Tensor,
        slot_pos: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
    ) -> Tensor:
        """Advance every active slot by 1 full pass through the U shared blocks."""
        bsz = h.shape[0]
        batch_idx = torch.arange(bsz, device=self.device)
        max_len = k_cache.shape[3]
        pos_range = torch.arange(max_len, device=self.device).unsqueeze(0)  # (1, T_max)
        # Causal mask per slot: slot s can attend to positions <= slot_pos[s]
        attn_mask = (pos_range <= slot_pos.unsqueeze(1)).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T_max)

        # Compute Triton SGMV group metadata ONCE and reuse across all U blocks & projections
        group_meta = (
            prepare_recursion_groups(slot_rec, num_recursions=self.num_recursions)
            if h.is_cuda and self.config.lora_rank > 0
            else None
        )

        cos = self.model.rope_cos
        sin = self.model.rope_sin

        for u, blk in enumerate(self.packed_blocks):
            v_layer = slot_rec * self.num_unique_blocks + u  # (B,)

            # 1. Attention
            h_norm = _batched_rmsnorm(h, blk.attn_norm_weights, slot_rec, blk.share_norms, blk.eps)
            if blk.fuse_attn:
                q, k, v = blk.qkv_proj.forward(h_norm, slot_rec, group_meta).chunk(3, dim=-1)
            else:
                q = blk.q_proj.forward(h_norm, slot_rec, group_meta)
                k = blk.k_proj.forward(h_norm, slot_rec, group_meta)
                v = blk.v_proj.forward(h_norm, slot_rec, group_meta)

            q = _apply_rope_per_pos(q.view(bsz, blk.n_heads, blk.head_dim), cos, sin, slot_pos)
            k = _apply_rope_per_pos(k.view(bsz, blk.n_heads, blk.head_dim), cos, sin, slot_pos)
            v = v.view(bsz, blk.n_heads, blk.head_dim)

            # Write new K, V into each slot's virtual layer cache at slot_pos
            k_cache[v_layer, batch_idx, :, slot_pos, :] = k
            v_cache[v_layer, batch_idx, :, slot_pos, :] = v

            # Gather KV for each slot's current virtual layer (B, H, T_max, head_dim)
            k_all = k_cache[v_layer, batch_idx]
            v_all = v_cache[v_layer, batch_idx]

            # Single-query attention against cached prefix
            attn_out = F.scaled_dot_product_attention(
                q.unsqueeze(2),  # (B, H, 1, head_dim)
                k_all,
                v_all,
                attn_mask=attn_mask,
                is_causal=False,
            ).squeeze(2)
            attn_out = attn_out.reshape(bsz, blk.d_model)
            h = h + blk.o_proj.forward(attn_out, slot_rec, group_meta)

            # 2. SwiGLU FFN
            h_ffn = _batched_rmsnorm(h, blk.ffn_norm_weights, slot_rec, blk.share_norms, blk.eps)
            if blk.fuse_ffn:
                gate, up = blk.gate_up_proj.forward(h_ffn, slot_rec, group_meta).chunk(2, dim=-1)
            else:
                gate = blk.gate_proj.forward(h_ffn, slot_rec, group_meta)
                up = blk.up_proj.forward(h_ffn, slot_rec, group_meta)

            gated = triton_swiglu(gate, up) if h.is_cuda else (F.silu(gate) * up)
            h = h + blk.down_proj.forward(gated, slot_rec, group_meta)

        return h

    def _prefill_prompts(
        self,
        prompt_ids: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
    ) -> Tensor:
        """Prefill prompt prefix [:, :-1] at full depth into KV cache and return last-token hidden state."""
        bsz, prompt_len = prompt_ids.shape
        if prompt_len <= 1:
            return self.model.tok_emb(prompt_ids[:, -1])

        prefix = prompt_ids[:, :-1]  # (B, T_pre)
        t_pre = prefix.shape[1]
        x = self.model.tok_emb(prefix)
        cos = self.model.rope_cos
        sin = self.model.rope_sin

        for v_layer, (block_idx, rec_idx) in enumerate(self.model.layer_schedule):
            blk = self.model.blocks[block_idx]
            norm_idx = 0 if blk.share_norms else rec_idx

            h_norm = blk.attn_norms[norm_idx](x)
            if blk.attn.fuse_projections:
                q, k, v = blk.attn.qkv_proj(h_norm, rec_idx).chunk(3, dim=-1)
            else:
                q = blk.attn.q_proj(h_norm, rec_idx)
                k = blk.attn.k_proj(h_norm, rec_idx)
                v = blk.attn.v_proj(h_norm, rec_idx)

            q = q.view(bsz, t_pre, blk.attn.n_heads, blk.attn.head_dim).transpose(1, 2)
            k = k.view(bsz, t_pre, blk.attn.n_heads, blk.attn.head_dim).transpose(1, 2)
            v = v.view(bsz, t_pre, blk.attn.n_heads, blk.attn.head_dim).transpose(1, 2)

            half = blk.attn.head_dim // 2
            cos_t = cos[:t_pre].unsqueeze(0).unsqueeze(0)
            sin_t = sin[:t_pre].unsqueeze(0).unsqueeze(0)
            q_f, k_f = q.float(), k.float()
            q = torch.cat((q_f[..., :half] * cos_t - q_f[..., half:] * sin_t, q_f[..., :half] * sin_t + q_f[..., half:] * cos_t), dim=-1).to(x.dtype)
            k = torch.cat((k_f[..., :half] * cos_t - k_f[..., half:] * sin_t, k_f[..., :half] * sin_t + k_f[..., half:] * cos_t), dim=-1).to(x.dtype)

            k_cache[v_layer, :, :, :t_pre, :] = k
            v_cache[v_layer, :, :, :t_pre, :] = v

            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, t_pre, blk.attn.d_model)
            x = x + blk.attn.o_proj(attn_out, rec_idx)
            x = x + blk.ffn(blk.ffn_norms[norm_idx](x), rec_idx)

        return self.model.tok_emb(prompt_ids[:, -1])

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Tensor,
        max_new_tokens: int,
        exit_threshold: float = 1.01,
        min_exit_recursion: int = 0,
        temperature: float = 0.0,
        top_k: int | None = None,
    ) -> tuple[list[list[int]], DepthwiseStats]:
        """Generate tokens using Continuous Depth-Wise Batching with Early-Exit.

        Args:
            prompt_ids: (B, T_prompt) initial token ids on the model's device.
            max_new_tokens: number of new tokens to generate per sequence in the batch.
            exit_threshold: softmax max-probability threshold for early exit.
                Values > 1.0 disable early exit (full depth R for every token).
            min_exit_recursion: earliest 0-based recursion step allowed to exit (0..R-1).
            temperature: 0.0 for greedy decoding; > 0 for sampling.
            top_k: optional top-k filtering when temperature > 0.

        Returns:
            (generated_sequences, DepthwiseStats) where each sequence includes prompt + generated tokens.
        """
        self.model.eval()
        bsz, prompt_len = prompt_ids.shape
        total_len = prompt_len + max_new_tokens
        if total_len > self.config.max_seq_len:
            raise ValueError(
                f"prompt_len ({prompt_len}) + max_new_tokens ({max_new_tokens}) exceeds "
                f"max_seq_len ({self.config.max_seq_len})"
            )

        n_heads = self.config.n_heads
        head_dim = self.config.d_model // n_heads
        k_cache = torch.zeros(
            (self.num_virtual_layers, bsz, n_heads, total_len, head_dim),
            device=self.device,
            dtype=self.dtype,
        )
        v_cache = torch.zeros_like(k_cache)

        h = self._prefill_prompts(prompt_ids, k_cache, v_cache)  # (B, D)
        slot_rec = torch.zeros((bsz,), device=self.device, dtype=torch.long)
        slot_pos = torch.full((bsz,), prompt_len - 1, device=self.device, dtype=torch.long)
        tokens_generated = torch.zeros((bsz,), device=self.device, dtype=torch.long)
        done = torch.zeros((bsz,), device=self.device, dtype=torch.bool)

        out_tokens = torch.zeros((bsz, max_new_tokens), device=self.device, dtype=torch.long)
        exit_hist_t = torch.zeros((self.num_recursions,), device=self.device, dtype=torch.long)
        iterations = 0
        u_count = self.num_unique_blocks

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        while not bool(done.all()):
            iterations += 1
            h = self._step_one_recursion_iteration(h, slot_rec, slot_pos, k_cache, v_cache)

            logits = self.model.lm_head(self.model.final_norm(h))  # (B, V)
            probs = F.softmax(logits.float(), dim=-1)
            max_conf = probs.max(dim=-1).values

            must_exit = slot_rec >= (self.num_recursions - 1)
            can_early_exit = (slot_rec >= min_exit_recursion) & (max_conf >= exit_threshold)
            exited = (~done) & (must_exit | can_early_exit)

            if bool(exited.any()):
                exit_indices = exited.nonzero(as_tuple=False).squeeze(-1)
                exit_logits = logits[exit_indices]
                if temperature <= 0.0:
                    next_tokens = exit_logits.argmax(dim=-1)
                else:
                    scaled = exit_logits.float() / temperature
                    if top_k is not None:
                        v_top, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                        scaled = scaled.masked_fill(scaled < v_top[:, [-1]], float("-inf"))
                    next_tokens = torch.multinomial(F.softmax(scaled, dim=-1), num_samples=1).squeeze(-1)

                exit_recs = slot_rec[exit_indices]
                exit_hist_t += torch.bincount(exit_recs, minlength=self.num_recursions)
                out_tokens[exit_indices, tokens_generated[exit_indices]] = next_tokens

                # Vectorized Bae et al. (2024) Section 4.1 KV-cache copying for early exits
                if bool((exit_recs < self.num_recursions - 1).any()):
                    for r_exit in range(self.num_recursions - 1):
                        r_mask = exited & (slot_rec == r_exit)
                        if not bool(r_mask.any()):
                            continue
                        r_slots = r_mask.nonzero(as_tuple=False).squeeze(-1)
                        r_pos = slot_pos[r_slots]
                        src_slice = slice(r_exit * u_count, (r_exit + 1) * u_count)
                        src_k = k_cache[src_slice, r_slots, :, r_pos, :]  # (U, n_r, H, D)
                        src_v = v_cache[src_slice, r_slots, :, r_pos, :]
                        for deeper_r in range(r_exit + 1, self.num_recursions):
                            dst_slice = slice(deeper_r * u_count, (deeper_r + 1) * u_count)
                            k_cache[dst_slice, r_slots, :, r_pos, :] = src_k
                            v_cache[dst_slice, r_slots, :, r_pos, :] = src_v

                tokens_generated[exit_indices] += 1
                newly_done = tokens_generated >= max_new_tokens
                done = done | newly_done

                # Immediately refill continuing exited slots with their next token at depth 0
                cont_mask = ~newly_done[exit_indices]
                continuing = exit_indices[cont_mask]
                if continuing.numel() > 0:
                    slot_pos[continuing] += 1
                    slot_rec[continuing] = 0
                    h[continuing] = self.model.tok_emb(next_tokens[cont_mask])

            # Advance non-exited, active slots to the next recursion step
            continuing_deeper = (~done) & (~exited)
            slot_rec[continuing_deeper] += 1

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        full_seqs = torch.cat([prompt_ids, out_tokens], dim=1).tolist()
        stats = DepthwiseStats(
            total_tokens_generated=bsz * max_new_tokens,
            total_loop_iterations=iterations,
            elapsed_sec=elapsed,
            exit_depth_histogram=exit_hist_t.tolist(),
        )
        return full_seqs, stats
