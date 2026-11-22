from __future__ import annotations

import pytest
import torch

from rrt.kernels import (
    TritonRMSNorm,
    prepare_recursion_groups,
    triton_grouped_lora_add,
    triton_grouped_relaxed_linear,
    triton_rmsnorm,
    triton_swiglu,
)
from rrt.layers import RMSNorm
from rrt.model import RecursiveTransformerBlock, precompute_rope_freqs
import torch.nn.functional as F

CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="Triton kernels require CUDA")
@pytest.mark.parametrize("shape", [(16, 128), (64, 768), (4, 64, 1024), (2, 128, 1536), (32, 2048)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_triton_rmsnorm_forward_and_backward_match_reference(
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    d_model = shape[-1]

    ref = RMSNorm(d_model, eps=1e-6).to(device=device, dtype=dtype)
    tri = TritonRMSNorm(d_model, eps=1e-6).to(device=device, dtype=dtype)
    with torch.no_grad():
        ref.weight.uniform_(0.5, 1.5)
        tri.weight.copy_(ref.weight)

    x_ref = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    x_tri = x_ref.detach().clone().requires_grad_(True)
    dy = torch.randn(shape, device=device, dtype=dtype)

    y_ref = ref(x_ref)
    y_tri = tri(x_tri)

    atol = 1e-5 if dtype == torch.float32 else 2e-2
    rtol = 1e-4 if dtype == torch.float32 else 2e-2
    assert torch.allclose(y_tri, y_ref, atol=atol, rtol=rtol)

    y_ref.backward(dy)
    y_tri.backward(dy)

    assert x_tri.grad is not None and x_ref.grad is not None
    assert tri.weight.grad is not None and ref.weight.grad is not None
    assert torch.allclose(x_tri.grad, x_ref.grad, atol=atol, rtol=rtol)
    # dw sums over M rows, so slightly looser absolute tolerance for low-precision dtypes
    dw_atol = 1e-4 if dtype == torch.float32 else 5e-2
    assert torch.allclose(tri.weight.grad, ref.weight.grad, atol=dw_atol, rtol=rtol)


def test_triton_rmsnorm_cpu_fallback() -> None:
    torch.manual_seed(1)
    x = torch.randn(4, 16, 64, requires_grad=True)
    w = torch.ones(64, requires_grad=True)
    y = triton_rmsnorm(x, w)
    ref = RMSNorm(64)(x)
    assert torch.allclose(y, ref, atol=1e-6)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="Triton kernels require CUDA")
@pytest.mark.parametrize("shape", [(32, 3072), (4, 64, 1536), (17, 777)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("chunked_view", [False, True])
def test_triton_swiglu_forward_and_backward(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    chunked_view: bool,
) -> None:
    torch.manual_seed(2)
    device = torch.device("cuda")
    dy = torch.randn(shape, device=device, dtype=dtype)

    if chunked_view:
        # Exercise non-contiguous strided views from gate_up.chunk(2, dim=-1)
        gu_shape = (*shape[:-1], shape[-1] * 2)
        gu_ref = torch.randn(gu_shape, device=device, dtype=dtype, requires_grad=True)
        gu_tri = gu_ref.detach().clone().requires_grad_(True)
        g_ref, u_ref = gu_ref.chunk(2, dim=-1)
        g_tri, u_tri = gu_tri.chunk(2, dim=-1)
        y_ref = F.silu(g_ref) * u_ref
        y_tri = triton_swiglu(g_tri, u_tri)
        y_ref.backward(dy)
        y_tri.backward(dy)
        grad_pairs = [(gu_tri.grad, gu_ref.grad)]
    else:
        g_ref = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
        u_ref = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
        g_tri = g_ref.detach().clone().requires_grad_(True)
        u_tri = u_ref.detach().clone().requires_grad_(True)
        y_ref = F.silu(g_ref) * u_ref
        y_tri = triton_swiglu(g_tri, u_tri)
        y_ref.backward(dy)
        y_tri.backward(dy)
        grad_pairs = [(g_tri.grad, g_ref.grad), (u_tri.grad, u_ref.grad)]

    atol = 1e-5 if dtype == torch.float32 else 2e-2
    rtol = 1e-4 if dtype == torch.float32 else 2e-2
    assert torch.allclose(y_tri, y_ref, atol=atol, rtol=rtol)
    for g_t, g_r in grad_pairs:
        assert g_t is not None and g_r is not None
        assert torch.allclose(g_t, g_r, atol=atol, rtol=rtol)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="Triton kernels require CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("pattern", ["balanced", "unbalanced_with_empty", "3d_batch"])
def test_triton_grouped_lora_forward_and_backward(dtype: torch.dtype, pattern: str) -> None:
    """Verify Triton SGMV grouped LoRA matches naive per-recursion loop in forward and backward."""
    torch.manual_seed(7)
    device = torch.device("cuda")
    num_rec, rank, d_in, d_out = 4, 32, 128, 192
    scaling = 1.5

    if pattern == "balanced":
        x_shape = (96, d_in)
        rec_ids = torch.randint(0, num_rec, (96,), device=device)
    elif pattern == "unbalanced_with_empty":
        x_shape = (73, d_in)
        # Recursion 2 intentionally has 0 tokens!
        choices = torch.tensor([0, 1, 3], device=device)
        rec_ids = choices[torch.randint(0, 3, (73,), device=device)]
    else:
        x_shape = (4, 24, d_in)
        rec_ids = torch.randint(0, num_rec, (4, 24), device=device)

    w_ref = torch.randn((d_out, d_in), device=device, dtype=dtype) * 0.05
    a_ref = torch.randn((num_rec, rank, d_in), device=device, dtype=dtype) * 0.05
    b_ref = torch.randn((num_rec, d_out, rank), device=device, dtype=dtype) * 0.05
    x_data = torch.randn(x_shape, device=device, dtype=dtype)
    dy = torch.randn((*x_shape[:-1], d_out), device=device, dtype=dtype)

    # Reference naive PyTorch implementation
    x_r = x_data.clone().requires_grad_(True)
    w_r = w_ref.clone().requires_grad_(True)
    a_r = a_ref.clone().requires_grad_(True)
    b_r = b_ref.clone().requires_grad_(True)

    y_base_r = F.linear(x_r, w_r)
    x_r_2d = x_r.reshape(-1, d_in)
    rec_1d = rec_ids.reshape(-1)
    delta_r = torch.zeros((x_r_2d.shape[0], d_out), device=device, dtype=dtype)
    for k in range(num_rec):
        mask = rec_1d == k
        if mask.any():
            delta_r[mask] = scaling * F.linear(F.linear(x_r_2d[mask], a_r[k]), b_r[k])
    y_r = y_base_r + delta_r.view(*x_shape[:-1], d_out)
    y_r.backward(dy)

    # Custom Triton grouped SGMV implementation (with reusable metadata)
    x_t = x_data.clone().requires_grad_(True)
    w_t = w_ref.clone().requires_grad_(True)
    a_t = a_ref.clone().requires_grad_(True)
    b_t = b_ref.clone().requires_grad_(True)

    meta = prepare_recursion_groups(rec_ids, num_recursions=num_rec)
    y_t = triton_grouped_relaxed_linear(x_t, w_t, a_t, b_t, scaling=scaling, group_meta=meta)
    y_t.backward(dy)

    atol = 1e-4 if dtype == torch.float32 else 3e-2
    rtol = 1e-4 if dtype == torch.float32 else 3e-2
    assert torch.allclose(y_t, y_r, atol=atol, rtol=rtol)
    assert torch.allclose(x_t.grad, x_r.grad, atol=atol, rtol=rtol)
    assert torch.allclose(w_t.grad, w_r.grad, atol=atol, rtol=rtol)
    # Weight grads sum across tokens & hidden dims (values ~O(10)), so verify relative L2 error too
    w_atol = 1e-4 if dtype == torch.float32 else 0.15
    assert torch.allclose(a_t.grad, a_r.grad, atol=w_atol, rtol=rtol)
    assert torch.allclose(b_t.grad, b_r.grad, atol=w_atol, rtol=rtol)
    rel_l2_a = (a_t.grad.float() - a_r.grad.float()).norm() / a_r.grad.float().norm().clamp_min(1e-8)
    rel_l2_b = (b_t.grad.float() - b_r.grad.float()).norm() / b_r.grad.float().norm().clamp_min(1e-8)
    max_rel = 1e-4 if dtype == torch.float32 else 1e-2
    assert rel_l2_a.item() < max_rel
    assert rel_l2_b.item() < max_rel


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="Triton kernels require CUDA")
def test_transformer_block_with_triton_kernels_matches_pytorch() -> None:
    torch.manual_seed(99)
    device, dtype = torch.device("cuda"), torch.bfloat16
    d_model, n_heads, d_ff = 256, 4, 512
    cos, sin = precompute_rope_freqs(d_model // n_heads, 64, device=device)

    blk_ref = RecursiveTransformerBlock(
        d_model, n_heads, d_ff, 4, 16, 16.0, 0.0, 0.0, False,
        lora_mode="auto", fuse_projections=True, use_triton_kernels=False,
    ).to(device=device, dtype=dtype)
    blk_tri = RecursiveTransformerBlock(
        d_model, n_heads, d_ff, 4, 16, 16.0, 0.0, 0.0, False,
        lora_mode="auto", fuse_projections=True, use_triton_kernels=True,
    ).to(device=device, dtype=dtype)
    blk_tri.load_state_dict(blk_ref.state_dict())

    x_ref = torch.randn((2, 32, d_model), device=device, dtype=dtype, requires_grad=True)
    x_tri = x_ref.detach().clone().requires_grad_(True)
    dy = torch.randn_like(x_ref)

    y_ref = blk_ref(x_ref, cos, sin, recursion_idx=2)
    y_tri = blk_tri(x_tri, cos, sin, recursion_idx=2)
    assert torch.allclose(y_tri, y_ref, atol=3e-2, rtol=3e-2)

    y_ref.backward(dy)
    y_tri.backward(dy)
    assert torch.allclose(x_tri.grad, x_ref.grad, atol=5e-2, rtol=5e-2)
