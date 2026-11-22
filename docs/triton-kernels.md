# Custom Triton Kernels & Structural Fusion

`src/rrt/kernels/` contains three custom GPU kernels written in **OpenAI Triton**, paired with structural QKV and Gate+Up projection fusion in `RecursiveTransformerBlock` (`src/rrt/model.py`). All kernels automatically fall back to exact PyTorch implementations on CPU.

## 1. Fused RMSNorm (`src/rrt/kernels/rmsnorm.py`)

`TritonRMSNorm` fuses variance reduction, reciprocal square-root normalization, and affine weight scaling into a single HBM pass:

- **Forward:** Single-pass row kernel accumulating $\sum_j x_j^2$ in `fp32` registers before emitting `bf16`/`fp16`/`fp32` outputs and caching `rstd` per row.
- **Backward (persistent, zero global atomics):** Computes `dx` per row while accumulating partial `dw` gradients into per-SM persistent buffers, followed by a tiny deterministic reduction pass. Eliminating `tl.atomic_add` contention on `dw` yields **7.40×** speedup over eager PyTorch and **1.57×** over `torch.compile`.

## 2. Zero-Copy 2D Strided SwiGLU (`src/rrt/kernels/swiglu.py`)

When `fuse_projections=True`, the MLP computes `gate_up = W_gu(x)` and splits it via `.chunk(2, dim=-1)`, producing **non-contiguous 2D strided views** `(M, D_ff)` with `stride(0) = 2 * D_ff`.

- `triton_swiglu` indexes both tensors using explicit 2D row/column strides (`stride_gm`, `stride_gn`, `stride_um`, `stride_un`), eliminating intermediate `.contiguous()` memory copies in both forward and backward passes.
- Uses `fp32` sigmoid math inside the kernel (`gate * sigmoid(gate) * up`) for numerical parity with PyTorch while staying memory-bandwidth bound (**1.44×** vs eager).

## 3. Grouped / Segmented LoRA SGMV (`src/rrt/kernels/grouped_lora.py`)

During **Continuous Depth-Wise Batching**, tokens in the same decode batch can be at different recursion depths $k \in \{0, \dots, R-1\}$. Running a Python loop over $R$ masks launches $2R$ small GEMMs with severe kernel-launch overhead.

`triton_grouped_relaxed_linear()` implements a two-stage **Segmented Gather-Matrix-Vector / Matrix-Matrix (SGMV)** kernel:

1. **`prepare_recursion_groups(recursion_ids, num_recursions)`**: Sorts token indices by recursion depth once per decode step and builds compact tile-to-group metadata (`sorted_idx`, `group_offsets`, `tile_group_ids`, `tile_m_starts`). Reusing this metadata across `QKV`, `O`, `Gate+Up`, and `Down` projections amortizes sorting cost to near zero.
2. **Stage 1 (Shrink, `D_in -> r`)**: Gathers active tokens by `sorted_idx`, applies their assigned $A_k$, and writes `(M, r)` intermediate activations.
3. **Stage 2 (Expand, `r -> D_out`)**: Multiplies by $B_k$, scales by $\alpha / r$, and scatters/adds directly onto the shared base GEMM output $y_{\text{base}} = x W^\top$.
4. **Full Backward Pass**: Custom autograd computes `dx`, `dW`, `dA`, and `dB` cleanly across unbalanced and even empty recursion groups.

## 4. Benchmarks (NVIDIA RTX 4080 16 GB, `bfloat16`)

Reproduce all numbers locally with:

```bash
TRITON_DEFAULT_FP_FUSION=1 uv run python scripts/benchmark_kernels.py
```

### Individual Kernels

| Kernel | Shape | Eager PyTorch | `torch.compile` | **Custom Triton** | Speedup |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `TritonRMSNorm` (Fwd+Bwd) | `(4096, 1024)` | `507.9 µs` | `107.5 µs` | **`68.6 µs`** | **7.40×** vs Eager (**1.57×** vs compile) |
| `triton_swiglu` (Fwd+Bwd, zero-copy `.chunk`) | `(4096, 6144)` | `1539.3 µs` | — | **`1070.1 µs`** | **1.44×** vs Eager |

### Grouped LoRA SGMV ($R=4, r=64$, reused metadata)

| Shape `(M, D_in -> D_out)` | Shared Base GEMM | PyTorch Loop | **Triton SGMV** | Speedup vs PyTorch |
| :--- | ---: | ---: | ---: | ---: |
| `(64, 1024 -> 1024)` | `14.3 µs` | `278.8 µs` | **`25.6 µs`** | **10.89×** |
| `(256, 1024 -> 1024)` | `11.3 µs` | `283.5 µs` | **`24.6 µs`** | **11.54×** |
| `(1024, 1024 -> 3072)` | `80.8 µs` | `405.7 µs` | **`106.5 µs`** | **3.81×** |

### Full Recursive Transformer Block (`8 × 256 = 2048` tokens, `d=1024, d_ff=3072, r=64`)

| Configuration | Forward | Forward + Backward | Speedup (Fwd / F+B) |
| :--- | ---: | ---: | ---: |
| `relaxed_split_eager` | `1244.2 µs` | `3786.8 µs` | `1.00×` / `1.00×` |
| `relaxed_fused_triton` (no compile) | **`1002.7 µs`** | **`3094.5 µs`** | **1.24×** / **1.22×** |
| `relaxed_fused_auto_compile` | **`822.3 µs`** | **`2725.9 µs`** | **1.51×** / **1.39×** |
