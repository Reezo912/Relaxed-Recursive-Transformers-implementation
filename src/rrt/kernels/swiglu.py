from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _swiglu_fwd_2d_kernel(
    G_ptr,
    U_ptr,
    Y_ptr,
    stride_g_row,
    stride_u_row,
    stride_y_row,
    N_COLS,
    BLOCK_N: tl.constexpr,
):
    """2D row-strided Fused SwiGLU forward: reads strided chunk views with zero copies."""
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < N_COLS

    g = tl.load(G_ptr + row * stride_g_row + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(U_ptr + row * stride_u_row + cols, mask=mask, other=0.0).to(tl.float32)

    sig = tl.sigmoid(g)
    y = g * sig * u
    tl.store(Y_ptr + row * stride_y_row + cols, y, mask=mask)


@triton.jit
def _swiglu_bwd_2d_kernel(
    G_ptr,
    U_ptr,
    DY_ptr,
    DG_ptr,
    DU_ptr,
    stride_g_row,
    stride_u_row,
    stride_dy_row,
    stride_dg_row,
    stride_du_row,
    N_COLS,
    BLOCK_N: tl.constexpr,
):
    """Single-pass Fused SwiGLU backward: recomputes sigmoid(g) in registers and writes dG, dU."""
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < N_COLS

    g = tl.load(G_ptr + row * stride_g_row + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(U_ptr + row * stride_u_row + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY_ptr + row * stride_dy_row + cols, mask=mask, other=0.0).to(tl.float32)

    sig = tl.sigmoid(g)
    silu_g = g * sig
    du = dy * silu_g
    dg = dy * u * sig * (1.0 + g * (1.0 - sig))

    tl.store(DG_ptr + row * stride_dg_row + cols, dg, mask=mask)
    tl.store(DU_ptr + row * stride_du_row + cols, du, mask=mask)


def _as_2d_last_contiguous(t: Tensor) -> Tensor:
    """View tensor as 2D (M, D) preserving row strides if last dim is contiguous (e.g. chunk views)."""
    n_cols = t.shape[-1]
    if t.stride(-1) != 1:
        t = t.contiguous()
    # Try zero-copy view first; fall back to contiguous if leading dims are non-uniformly strided
    try:
        return t.view(-1, n_cols)
    except RuntimeError:
        return t.contiguous().view(-1, n_cols)


class _TritonSwiGLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: Tensor, up: Tensor) -> Tensor:
        if gate.shape != up.shape:
            raise ValueError(f"gate shape {gate.shape} != up shape {up.shape}")
        orig_shape = gate.shape
        g_2d = _as_2d_last_contiguous(gate)
        u_2d = _as_2d_last_contiguous(up)
        n_rows, n_cols = g_2d.shape

        out_2d = torch.empty((n_rows, n_cols), device=gate.device, dtype=gate.dtype)
        block_n = 1024 if n_cols >= 1024 else triton.next_power_of_2(max(128, n_cols))
        grid = (n_rows, triton.cdiv(n_cols, block_n))

        _swiglu_fwd_2d_kernel[grid](
            g_2d,
            u_2d,
            out_2d,
            g_2d.stride(0),
            u_2d.stride(0),
            out_2d.stride(0),
            n_cols,
            BLOCK_N=block_n,
            num_warps=4,
        )

        if any(ctx.needs_input_grad):
            ctx.save_for_backward(g_2d, u_2d)
            ctx.block_n = block_n
        return out_2d.view(orig_shape)

    @staticmethod
    def backward(ctx, dy: Tensor) -> tuple[Tensor, Tensor]:
        g_2d, u_2d = ctx.saved_tensors
        orig_shape = dy.shape
        n_rows, n_cols = g_2d.shape
        dy_2d = _as_2d_last_contiguous(dy)

        dg_2d = torch.empty((n_rows, n_cols), device=g_2d.device, dtype=g_2d.dtype)
        du_2d = torch.empty((n_rows, n_cols), device=u_2d.device, dtype=u_2d.dtype)

        grid = (n_rows, triton.cdiv(n_cols, ctx.block_n))
        _swiglu_bwd_2d_kernel[grid](
            g_2d,
            u_2d,
            dy_2d,
            dg_2d,
            du_2d,
            g_2d.stride(0),
            u_2d.stride(0),
            dy_2d.stride(0),
            dg_2d.stride(0),
            du_2d.stride(0),
            n_cols,
            BLOCK_N=ctx.block_n,
            num_warps=4,
        )
        return dg_2d.view(orig_shape), du_2d.view(orig_shape)


def triton_swiglu(gate: Tensor, up: Tensor) -> Tensor:
    """Fused SwiGLU (silu(gate) * up) backed by Triton on CUDA, with eager fallback on CPU."""
    if not gate.is_cuda:
        return F.silu(gate) * up
    return _TritonSwiGLUFn.apply(gate, up)
