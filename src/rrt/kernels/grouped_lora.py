from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor


@dataclass(frozen=True)
class RecursionGroupMetadata:
    """GPU-resident permutation and segment boundaries for continuous depth-wise batching.

    Can be computed once per Transformer block from ``rec_ids`` and reused across all
    projections (qkv, o, gate_up, down) with zero CPU-GPU synchronization.
    """

    perm: Tensor  # (M,) int32 token indices sorted by recursion depth
    starts: Tensor  # (R,) int32 start offset in perm for each recursion k
    ends: Tensor  # (R,) int32 end offset (exclusive) in perm for each recursion k
    num_recursions: int
    num_tokens: int


@torch.no_grad()
def prepare_recursion_groups(rec_ids: Tensor, num_recursions: int) -> RecursionGroupMetadata:
    """Group arbitrary unsorted recursion indices 0..R-1 completely on GPU (zero CPU sync)."""
    rec_1d = rec_ids.reshape(-1)
    # argsort groups tokens by recursion_idx; bincount+cumsum gives segment bounds on GPU
    perm = torch.argsort(rec_1d, stable=False).to(torch.int32)
    counts = torch.bincount(rec_1d.to(torch.int64), minlength=num_recursions).to(torch.int32)
    ends = torch.cumsum(counts, dim=0).to(torch.int32)
    starts = ends - counts
    return RecursionGroupMetadata(
        perm=perm,
        starts=starts,
        ends=ends,
        num_recursions=num_recursions,
        num_tokens=rec_1d.numel(),
    )


@triton.jit
def _grouped_lora_shrink_kernel(
    X_ptr,
    A_ptr,
    H_ptr,
    PERM_ptr,
    STARTS_ptr,
    ENDS_ptr,
    D_IN,
    RANK,
    stride_xm,
    stride_xk,
    stride_ar,
    stride_arow,
    stride_ak,
    stride_hm,
    stride_hr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    """Segmented LoRA shrink: H[tok, :] = X[tok, :] @ A[k]^T for all tokens at depth k."""
    pid_m = tl.program_id(0)
    k_rec = tl.program_id(1)

    seg_start = tl.load(STARTS_ptr + k_rec)
    seg_end = tl.load(ENDS_ptr + k_rec)
    tile_start = seg_start + pid_m * BLOCK_M
    if tile_start >= seg_end:
        return

    offs_seg = tile_start + tl.arange(0, BLOCK_M)
    m_mask = offs_seg < seg_end
    tok_idx = tl.load(PERM_ptr + offs_seg, mask=m_mask, other=0).to(tl.int64)

    offs_r = tl.arange(0, BLOCK_R)
    r_mask = offs_r < RANK

    acc = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
    a_rec_base = A_ptr + k_rec * stride_ar

    for k_off in range(0, D_IN, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        k_mask = offs_k < D_IN

        x_ptrs = X_ptr + tok_idx[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_tile = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # Load A[k, r, k_in] transposed to (BLOCK_K, BLOCK_R) for Tensor Core tl.dot
        a_ptrs = a_rec_base + offs_r[None, :] * stride_arow + offs_k[:, None] * stride_ak
        a_t_tile = tl.load(a_ptrs, mask=k_mask[:, None] & r_mask[None, :], other=0.0)

        acc += tl.dot(x_tile, a_t_tile, input_precision=INPUT_PRECISION)

    h_ptrs = H_ptr + tok_idx[:, None] * stride_hm + offs_r[None, :] * stride_hr
    tl.store(h_ptrs, acc, mask=m_mask[:, None] & r_mask[None, :])


@triton.jit
def _grouped_lora_expand_add_kernel(
    H_ptr,
    B_ptr,
    Y_BASE_ptr,
    Y_OUT_ptr,
    PERM_ptr,
    STARTS_ptr,
    ENDS_ptr,
    D_OUT,
    RANK,
    scaling,
    stride_hm,
    stride_hr,
    stride_br,
    stride_bn,
    stride_bk,
    stride_ybm,
    stride_ybn,
    stride_yom,
    stride_yon,
    ADD_BASE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    """Segmented LoRA expand + fused epilogue add: Y_out[tok] = Y_base[tok] + scaling * (H[tok] @ B[k]^T)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    k_rec = tl.program_id(2)

    seg_start = tl.load(STARTS_ptr + k_rec)
    seg_end = tl.load(ENDS_ptr + k_rec)
    tile_start = seg_start + pid_m * BLOCK_M
    if tile_start >= seg_end:
        return

    offs_seg = tile_start + tl.arange(0, BLOCK_M)
    m_mask = offs_seg < seg_end
    tok_idx = tl.load(PERM_ptr + offs_seg, mask=m_mask, other=0).to(tl.int64)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < D_OUT

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    b_rec_base = B_ptr + k_rec * stride_br

    for r_off in range(0, RANK, BLOCK_R):
        offs_r = r_off + tl.arange(0, BLOCK_R)
        r_mask = offs_r < RANK

        h_ptrs = H_ptr + tok_idx[:, None] * stride_hm + offs_r[None, :] * stride_hr
        h_tile = tl.load(h_ptrs, mask=m_mask[:, None] & r_mask[None, :], other=0.0)

        # Load B[k, n, r] transposed to (BLOCK_R, BLOCK_N) for Tensor Core tl.dot
        b_ptrs = b_rec_base + offs_n[None, :] * stride_bn + offs_r[:, None] * stride_bk
        b_t_tile = tl.load(b_ptrs, mask=r_mask[:, None] & n_mask[None, :], other=0.0)

        acc += tl.dot(h_tile, b_t_tile, input_precision=INPUT_PRECISION)

    res = acc * scaling
    if ADD_BASE:
        yb_ptrs = Y_BASE_ptr + tok_idx[:, None] * stride_ybm + offs_n[None, :] * stride_ybn
        y_base = tl.load(yb_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)
        res = y_base + res

    yo_ptrs = Y_OUT_ptr + tok_idx[:, None] * stride_yom + offs_n[None, :] * stride_yon
    tl.store(yo_ptrs, res, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _grouped_lora_weight_grad_kernel(
    U_ptr,
    V_ptr,
    OUT_ptr,
    PERM_ptr,
    STARTS_ptr,
    ENDS_ptr,
    N_OUT,
    N_IN,
    scale,
    stride_um,
    stride_uo,
    stride_vm,
    stride_vi,
    stride_or,
    stride_oo,
    stride_oi,
    BLOCK_M: tl.constexpr,
    BLOCK_O: tl.constexpr,
    BLOCK_I: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    """Segmented outer-product reduction: OUT[k] = scale * sum_{tok in k} U[tok]^T @ V[tok]."""
    pid_o = tl.program_id(0)
    pid_i = tl.program_id(1)
    k_rec = tl.program_id(2)

    seg_start = tl.load(STARTS_ptr + k_rec)
    seg_end = tl.load(ENDS_ptr + k_rec)

    offs_o = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    o_mask = offs_o < N_OUT
    i_mask = offs_i < N_IN

    acc = tl.zeros((BLOCK_O, BLOCK_I), dtype=tl.float32)

    cur = seg_start
    while cur < seg_end:
        offs_seg = cur + tl.arange(0, BLOCK_M)
        m_mask = offs_seg < seg_end
        tok_idx = tl.load(PERM_ptr + offs_seg, mask=m_mask, other=0).to(tl.int64)

        u_ptrs = U_ptr + tok_idx[None, :] * stride_um + offs_o[:, None] * stride_uo
        u_t_tile = tl.load(u_ptrs, mask=o_mask[:, None] & m_mask[None, :], other=0.0)

        v_ptrs = V_ptr + tok_idx[:, None] * stride_vm + offs_i[None, :] * stride_vi
        v_tile = tl.load(v_ptrs, mask=m_mask[:, None] & i_mask[None, :], other=0.0)

        acc += tl.dot(u_t_tile, v_tile, input_precision=INPUT_PRECISION)
        cur += BLOCK_M

    out_ptrs = OUT_ptr + k_rec * stride_or + offs_o[:, None] * stride_oo + offs_i[None, :] * stride_oi
    tl.store(out_ptrs, acc * scale, mask=o_mask[:, None] & i_mask[None, :])


def _launch_shrink(x: Tensor, a: Tensor, meta: RecursionGroupMetadata) -> Tensor:
    m, d_in = x.shape
    num_rec, rank, _ = a.shape
    h = torch.empty((m, rank), device=x.device, dtype=x.dtype)

    block_m = 32 if m <= 256 else 64
    block_k = 64
    block_r = max(16, triton.next_power_of_2(rank))
    if block_r > 128:
        raise ValueError(f"triton grouped LoRA currently supports rank <= 128, got {rank}")

    input_precision = "ieee" if x.dtype == torch.float32 else "tf32"
    grid = (triton.cdiv(m, block_m), num_rec)
    _grouped_lora_shrink_kernel[grid](
        x,
        a,
        h,
        meta.perm,
        meta.starts,
        meta.ends,
        d_in,
        rank,
        x.stride(0),
        x.stride(1),
        a.stride(0),
        a.stride(1),
        a.stride(2),
        h.stride(0),
        h.stride(1),
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        BLOCK_R=block_r,
        INPUT_PRECISION=input_precision,
        num_warps=4,
    )
    return h


def _launch_expand(
    h: Tensor,
    b: Tensor,
    meta: RecursionGroupMetadata,
    scaling: float,
    y_base: Tensor | None = None,
    inplace: bool = False,
) -> Tensor:
    m, rank = h.shape
    num_rec, d_out, _ = b.shape
    if y_base is not None and inplace:
        y_out = y_base
    else:
        y_out = torch.empty((m, d_out), device=h.device, dtype=h.dtype)

    add_base = y_base is not None
    y_base_arg = y_base if y_base is not None else y_out

    block_m = 32 if m <= 256 else 64
    block_n = 64
    block_r = min(64, max(16, triton.next_power_of_2(rank)))
    input_precision = "ieee" if h.dtype == torch.float32 else "tf32"

    grid = (triton.cdiv(m, block_m), triton.cdiv(d_out, block_n), num_rec)
    _grouped_lora_expand_add_kernel[grid](
        h,
        b,
        y_base_arg,
        y_out,
        meta.perm,
        meta.starts,
        meta.ends,
        d_out,
        rank,
        scaling,
        h.stride(0),
        h.stride(1),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        y_base_arg.stride(0),
        y_base_arg.stride(1),
        y_out.stride(0),
        y_out.stride(1),
        ADD_BASE=add_base,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_R=block_r,
        INPUT_PRECISION=input_precision,
        num_warps=4,
    )
    return y_out


def _launch_weight_grad(u: Tensor, v: Tensor, meta: RecursionGroupMetadata, scale: float) -> Tensor:
    n_out = u.shape[1]
    n_in = v.shape[1]
    out = torch.empty((meta.num_recursions, n_out, n_in), device=u.device, dtype=u.dtype)

    block_m = 64
    block_o = min(64, max(16, triton.next_power_of_2(n_out)))
    block_i = min(64, max(16, triton.next_power_of_2(n_in)))
    input_precision = "ieee" if u.dtype == torch.float32 else "tf32"
    grid = (triton.cdiv(n_out, block_o), triton.cdiv(n_in, block_i), meta.num_recursions)

    _grouped_lora_weight_grad_kernel[grid](
        u,
        v,
        out,
        meta.perm,
        meta.starts,
        meta.ends,
        n_out,
        n_in,
        scale,
        u.stride(0),
        u.stride(1),
        v.stride(0),
        v.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_M=block_m,
        BLOCK_O=block_o,
        BLOCK_I=block_i,
        INPUT_PRECISION=input_precision,
        num_warps=4,
    )
    return out


class _TritonGroupedLoRAAddFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        y_base: Tensor,
        x: Tensor,
        lora_a: Tensor,
        lora_b: Tensor,
        meta: RecursionGroupMetadata,
        scaling: float,
    ) -> Tensor:
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        yb_2d = y_base.reshape(-1, y_base.shape[-1]).contiguous()
        a_c = lora_a.contiguous()
        b_c = lora_b.contiguous()

        h = _launch_shrink(x_2d, a_c, meta)
        # In inference (no grad needed), we can safely reuse yb_2d only if caller permits;
        # always write to fresh y_out here to keep autograd semantics clean and side-effect-free.
        y_out = _launch_expand(h, b_c, meta, scaling=scaling, y_base=yb_2d, inplace=False)

        if any(ctx.needs_input_grad):
            ctx.save_for_backward(x_2d, a_c, b_c, h)
            ctx.meta = meta
            ctx.scaling = scaling
            ctx.x_shape = x.shape
        return y_out.view(y_base.shape)

    @staticmethod
    def backward(ctx, dy: Tensor) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None, None, None]:
        x_2d, a_c, b_c, h = ctx.saved_tensors
        meta: RecursionGroupMetadata = ctx.meta
        scaling: float = ctx.scaling
        dy_2d = dy.reshape(-1, dy.shape[-1]).contiguous()

        dy_base = dy if ctx.needs_input_grad[0] else None

        # dH[tok] = scaling * (dY[tok] @ B[k])  <-- B[k]^T has shape (r, D_out)
        need_dh = ctx.needs_input_grad[1] or ctx.needs_input_grad[2]
        dh = None
        if need_dh:
            b_t = b_c.transpose(1, 2).contiguous()  # (R, r, D_out)
            dh = _launch_shrink(dy_2d, b_t, meta)

        dx = None
        if ctx.needs_input_grad[1]:
            assert dh is not None
            # dX[tok] = scaling * (dH[tok] @ A[k]) <-- A[k]^T has shape (D_in, r)
            a_t = a_c.transpose(1, 2).contiguous()  # (R, D_in, r)
            dx = _launch_expand(dh, a_t, meta, scaling=scaling, y_base=None, inplace=False).view(ctx.x_shape)

        da = None
        if ctx.needs_input_grad[2]:
            assert dh is not None
            # dA[k] = scaling * sum_{tok in k} dH[tok]^T @ X[tok]
            da = _launch_weight_grad(dh, x_2d, meta, scale=scaling)

        db = None
        if ctx.needs_input_grad[3]:
            # dB[k] = scaling * sum_{tok in k} dY[tok]^T @ H[tok]
            db = _launch_weight_grad(dy_2d, h, meta, scale=scaling)

        return dy_base, dx, da, db, None, None


def triton_grouped_lora_add(
    y_base: Tensor,
    x: Tensor,
    lora_a: Tensor,
    lora_b: Tensor,
    rec_ids: Tensor | None = None,
    scaling: float = 1.0,
    group_meta: RecursionGroupMetadata | None = None,
) -> Tensor:
    """Compute y_base[i] + scaling * (x[i] @ A[k_i]^T) @ B[k_i]^T for mixed recursion depths k_i.

    Supports both training (autograd forward + backward) and fast inference, with zero
    CPU-GPU synchronization on CUDA and eager fallback on CPU.
    """
    num_rec = lora_a.shape[0]
    if not x.is_cuda:
        if rec_ids is None:
            raise ValueError("rec_ids must be provided on CPU")
        x_2d = x.reshape(-1, x.shape[-1])
        out = y_base.reshape(-1, y_base.shape[-1]).clone()
        rec_1d = rec_ids.reshape(-1)
        for k in range(num_rec):
            mask = rec_1d == k
            if mask.any():
                out[mask] = out[mask] + scaling * F.linear(F.linear(x_2d[mask], lora_a[k]), lora_b[k])
        return out.view(y_base.shape)

    if group_meta is None:
        if rec_ids is None:
            raise ValueError("Either rec_ids or group_meta must be provided")
        group_meta = prepare_recursion_groups(rec_ids, num_recursions=num_rec)

    return _TritonGroupedLoRAAddFn.apply(y_base, x, lora_a, lora_b, group_meta, scaling)


def triton_grouped_relaxed_linear(
    x: Tensor,
    weight: Tensor,
    lora_a: Tensor,
    lora_b: Tensor,
    rec_ids: Tensor | None = None,
    scaling: float = 1.0,
    bias: Tensor | None = None,
    group_meta: RecursionGroupMetadata | None = None,
) -> Tensor:
    """Full continuous depth-wise batching RelaxedLinear: 1 shared base GEMM + Triton SGMV LoRA."""
    y_base = F.linear(x, weight, bias)
    return triton_grouped_lora_add(
        y_base=y_base,
        x=x,
        lora_a=lora_a,
        lora_b=lora_b,
        rec_ids=rec_ids,
        scaling=scaling,
        group_meta=group_meta,
    )
