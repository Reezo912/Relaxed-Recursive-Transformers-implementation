from __future__ import annotations

import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch import Tensor


def _choose_num_warps(block_size: int) -> int:
    if block_size <= 1024:
        return 4
    if block_size <= 2048:
        return 8
    if block_size <= 8192:
        return 16
    return 32


@triton.jit
def _rmsnorm_fwd_kernel(
    X_ptr,
    W_ptr,
    Y_ptr,
    RSTD_ptr,
    stride_x_row,
    stride_y_row,
    N_COLS,
    eps,
    STORE_RSTD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """One Triton program per token row: 1 read of X, 1 read of W, 1 write of Y (+ optional scalar rstd)."""
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N_COLS

    x_ptrs = X_ptr + row_idx * stride_x_row + cols
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    mean_sq = tl.sum(x * x, axis=0) / N_COLS
    rstd = tl.math.rsqrt(mean_sq + eps)

    y = x * rstd * w
    if STORE_RSTD:
        tl.store(RSTD_ptr + row_idx, rstd)
    tl.store(Y_ptr + row_idx * stride_y_row + cols, y, mask=mask)


@triton.jit
def _rmsnorm_bwd_kernel(
    X_ptr,
    W_ptr,
    DY_ptr,
    RSTD_ptr,
    DX_ptr,
    PARTIAL_DW_ptr,
    stride_x_row,
    stride_dy_row,
    stride_dx_row,
    stride_pdw_row,
    N_ROWS,
    N_COLS,
    NUM_PROGS,
    BLOCK_SIZE: tl.constexpr,
):
    """Persistent strided backward kernel: keeps W in registers, writes DX per row,
    accumulates partial DW in fp32 registers with zero atomics (deterministic & contention-free).
    """
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N_COLS

    # Load W once into registers for all rows handled by this program
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    dw_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for row_idx in range(pid, N_ROWS, NUM_PROGS):
        x = tl.load(X_ptr + row_idx * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY_ptr + row_idx * stride_dy_row + cols, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(RSTD_ptr + row_idx).to(tl.float32)

        x_hat = x * rstd
        dw_acc += dy * x_hat

        m = dy * w
        dot = tl.sum(m * x_hat, axis=0) / N_COLS
        dx = rstd * (m - x_hat * dot)
        tl.store(DX_ptr + row_idx * stride_dx_row + cols, dx, mask=mask)

    tl.store(PARTIAL_DW_ptr + pid * stride_pdw_row + cols, dw_acc, mask=mask)


class _TritonRMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, eps: float) -> Tensor:
        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x_2d = x.reshape(-1, n_cols).contiguous()
        n_rows = x_2d.shape[0]

        block_size = triton.next_power_of_2(n_cols)
        if block_size > 65536:
            raise ValueError(f"Triton RMSNorm supports hidden dims <= 65536, got {n_cols}")

        y_2d = torch.empty_like(x_2d)
        needs_bwd = any(ctx.needs_input_grad)
        rstd = torch.empty((n_rows,), device=x.device, dtype=torch.float32) if needs_bwd else y_2d

        num_warps = _choose_num_warps(block_size)
        _rmsnorm_fwd_kernel[(n_rows,)](
            x_2d,
            weight,
            y_2d,
            rstd,
            x_2d.stride(0),
            y_2d.stride(0),
            n_cols,
            eps,
            STORE_RSTD=needs_bwd,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )

        if needs_bwd:
            ctx.save_for_backward(x_2d, weight, rstd)
            ctx.block_size = block_size
            ctx.num_warps = num_warps
        return y_2d.reshape(orig_shape)

    @staticmethod
    def backward(ctx, dy: Tensor) -> tuple[Tensor, Tensor, None]:
        x_2d, weight, rstd = ctx.saved_tensors
        orig_shape = dy.shape
        n_rows, n_cols = x_2d.shape
        dy_2d = dy.reshape(n_rows, n_cols).contiguous()

        dx_2d = torch.empty_like(x_2d)
        # 2 programs per SM gives good occupancy while keeping partial_dw tiny
        num_sms = torch.cuda.get_device_properties(x_2d.device).multi_processor_count
        num_progs = min(n_rows, num_sms * 2)
        partial_dw = torch.empty((num_progs, n_cols), device=x_2d.device, dtype=torch.float32)

        _rmsnorm_bwd_kernel[(num_progs,)](
            x_2d,
            weight,
            dy_2d,
            rstd,
            dx_2d,
            partial_dw,
            x_2d.stride(0),
            dy_2d.stride(0),
            dx_2d.stride(0),
            partial_dw.stride(0),
            n_rows,
            n_cols,
            num_progs,
            BLOCK_SIZE=ctx.block_size,
            num_warps=ctx.num_warps,
        )

        dw = partial_dw.sum(dim=0).to(weight.dtype)
        return dx_2d.reshape(orig_shape), dw, None


def triton_rmsnorm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    """Functional Fused RMSNorm backed by Triton on CUDA, with eager fallback on CPU."""
    if not x.is_cuda:
        normed = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
        return (normed * weight.float()).to(x.dtype)
    return _TritonRMSNormFn.apply(x, weight, eps)


class TritonRMSNorm(nn.Module):
    """Drop-in Triton-accelerated replacement for RMSNorm."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return triton_rmsnorm(x, self.weight, self.eps)
