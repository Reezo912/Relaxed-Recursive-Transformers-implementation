from __future__ import annotations

import torch
import torch.nn.functional as F
import triton

from rrt.kernels import (
    TritonRMSNorm,
    prepare_recursion_groups,
    triton_grouped_relaxed_linear,
    triton_swiglu,
)
from rrt.layers import RMSNorm, RelaxedLinear
from rrt.model import RecursiveTransformerBlock, precompute_rope_freqs

RTX4080_PEAK_BW_GBS = 716.8  # GB/s theoretical GDDR6X bandwidth (256-bit @ 22.4 Gbps)


def bench_us(fn, warmup: int = 25, rep: int = 100) -> float:
    """Return median GPU kernel time in microseconds using Triton's CUDA-event benchmarker."""
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")
    return float(ms) * 1000.0


def benchmark_rmsnorm(device: torch.device, dtype: torch.dtype) -> None:
    print("\n=== 1. RMSNorm: Eager vs torch.compile vs Custom Triton (bf16) ===")
    print(
        f"{'Shape (M, D)':<15} {'Eager Fwd':>10} {'Comp Fwd':>10} {'Triton Fwd':>11} "
        f"{'Triton GB/s':>12} | {'Eager F+B':>10} {'Comp F+B':>10} {'Triton F+B':>11} {'vs Eager':>9}"
    )
    print("-" * 108)

    for d_model in (1024, 2048):
        norm = RMSNorm(d_model).to(device=device, dtype=dtype)
        norm_compiled = torch.compile(norm)
        norm_triton = TritonRMSNorm(d_model).to(device=device, dtype=dtype)
        for m in (512, 2048, 4096):
            x = torch.randn((m, d_model), device=device, dtype=dtype, requires_grad=True)
            dy = torch.randn_like(x)

            # Warmup compile & triton (fwd + bwd)
            for mod in (norm_compiled, norm_triton):
                y = mod(x)
                y.backward(dy)
                x.grad = None
                mod.zero_grad(set_to_none=True)
            torch.cuda.synchronize()

            with torch.no_grad():
                t_e_f = bench_us(lambda: norm(x), warmup=50, rep=150)
                t_c_f = bench_us(lambda: norm_compiled(x), warmup=50, rep=150)
                t_t_f = bench_us(lambda: norm_triton(x), warmup=50, rep=150)

            def fb(mod: torch.nn.Module) -> None:
                x.grad = None
                mod.zero_grad(set_to_none=True)
                mod(x).backward(dy)

            t_e_fb = bench_us(lambda: fb(norm), warmup=50, rep=150)
            t_c_fb = bench_us(lambda: fb(norm_compiled), warmup=50, rep=150)
            t_t_fb = bench_us(lambda: fb(norm_triton), warmup=50, rep=150)

            bytes_moved = (2 * m * d_model + d_model) * x.element_size()
            bw_tri = bytes_moved / (t_t_f * 1e-6) / 1e9
            print(
                f"{f'({m}, {d_model})':<15} {t_e_f:>9.1f}u {t_c_f:>9.1f}u {t_t_f:>10.1f}u "
                f"{bw_tri:>12.1f} | {t_e_fb:>9.1f}u {t_c_fb:>9.1f}u {t_t_fb:>10.1f}u {t_e_fb / t_t_fb:>8.2f}x"
            )


def benchmark_swiglu(device: torch.device, dtype: torch.dtype) -> None:
    print("\n=== 2. SwiGLU: Eager vs torch.compile vs Custom Triton (bf16, chunked zero-copy view) ===")
    print(
        f"{'Shape (M, D_ff)':<16} {'Eager Fwd':>10} {'Comp Fwd':>10} {'Triton Fwd':>11} "
        f"{'Triton GB/s':>12} | {'Eager F+B':>10} {'Comp F+B':>10} {'Triton F+B':>11} {'vs Eager':>9}"
    )
    print("-" * 109)

    def swiglu_fn(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return F.silu(g) * u

    swiglu_compiled = torch.compile(swiglu_fn)

    for d_ff in (3072, 6144):
        for m in (512, 2048, 4096):
            # Simulate realistic fused gate_up output split with .chunk(2, dim=-1)
            gu = torch.randn((m, 2 * d_ff), device=device, dtype=dtype, requires_grad=True)
            dy = torch.randn((m, d_ff), device=device, dtype=dtype)

            for fn in (swiglu_compiled, triton_swiglu):
                g, u = gu.chunk(2, dim=-1)
                fn(g, u).backward(dy)
                gu.grad = None
            torch.cuda.synchronize()

            with torch.no_grad():
                g_ng, u_ng = gu.chunk(2, dim=-1)
                t_e_f = bench_us(lambda: swiglu_fn(g_ng, u_ng), warmup=50, rep=150)
                t_c_f = bench_us(lambda: swiglu_compiled(g_ng, u_ng), warmup=50, rep=150)
                t_t_f = bench_us(lambda: triton_swiglu(g_ng, u_ng), warmup=50, rep=150)

            def fb(fn) -> None:
                gu.grad = None
                g, u = gu.chunk(2, dim=-1)
                fn(g, u).backward(dy)

            t_e_fb = bench_us(lambda: fb(swiglu_fn), warmup=50, rep=150)
            t_c_fb = bench_us(lambda: fb(swiglu_compiled), warmup=50, rep=150)
            t_t_fb = bench_us(lambda: fb(triton_swiglu), warmup=50, rep=150)

            bytes_moved = 3 * m * d_ff * gu.element_size()
            bw_tri = bytes_moved / (t_t_f * 1e-6) / 1e9
            print(
                f"{f'({m}, {d_ff})':<16} {t_e_f:>9.1f}u {t_c_f:>9.1f}u {t_t_f:>10.1f}u "
                f"{bw_tri:>12.1f} | {t_e_fb:>9.1f}u {t_c_fb:>9.1f}u {t_t_fb:>10.1f}u {t_e_fb / t_t_fb:>8.2f}x"
            )


def benchmark_relaxed_linear(device: torch.device, dtype: torch.dtype) -> None:
    print("\n=== 3. RelaxedLinear: Base vs Split LoRA vs Merge-on-the-fly (bf16, r=64) ===")
    for in_f, out_f, label in ((1024, 1024, "Attn 1024->1024"), (1024, 3072, "FFN 1024->3072")):
        print(f"\n  [{label}, lora_rank=64]")
        print(
            f"  {'Tokens M':>8} {'Base (us)':>11} {'Split (us)':>12} {'Merge (us)':>12} "
            f"{'Split/Base':>11} {'Merge/Base':>11} {'Winner':>10}"
        )
        print("  " + "-" * 81)

        lin = RelaxedLinear(in_f, out_f, num_recursions=4, lora_rank=64).to(device=device, dtype=dtype)
        with torch.no_grad():
            lin.lora_B[0].normal_(0.0, 0.02)
        w = lin.weight
        a0 = lin.lora_A[0]
        b0 = lin.lora_B[0]
        scaling = lin.scaling

        def run_base(x: torch.Tensor) -> torch.Tensor:
            return F.linear(x, w)

        def run_split(x: torch.Tensor) -> torch.Tensor:
            return F.linear(x, w) + scaling * F.linear(F.linear(x, a0), b0)

        def run_merge(x: torch.Tensor) -> torch.Tensor:
            w_eff = w + scaling * (b0 @ a0)
            return F.linear(x, w_eff)

        for m in (1, 16, 64, 256, 512, 1024, 2048, 4096):
            x = torch.randn((m, in_f), device=device, dtype=dtype)
            t_base = bench_us(lambda: run_base(x))
            t_split = bench_us(lambda: run_split(x))
            t_merge = bench_us(lambda: run_merge(x))
            winner = "Split" if t_split < t_merge else "Merge"
            print(
                f"  {m:>8} {t_base:>11.1f} {t_split:>12.1f} {t_merge:>12.1f} "
                f"{t_split / t_base:>10.2f}x {t_merge / t_base:>10.2f}x {winner:>10}"
            )


def benchmark_depthwise_batching_naive(device: torch.device, dtype: torch.dtype) -> None:
    print("\n=== 4. Continuous Depth-Wise Batching: Naive Loop vs Triton SGMV (bf16, R=4, r=64) ===")
    print("  (Each token in the batch has a different recursion_idx k in {0, 1, 2, 3})")
    print(
        f"  {'Shape (M, D->Out)':<20} {'Base GEMM':>10} {'Naive Loop':>11} "
        f"{'Triton SGMV':>12} {'Triton (reused)':>16} {'vs Naive':>10} {'vs Base':>9}"
    )
    print("  " + "-" * 94)

    num_rec = 4
    rank = 64
    for in_f, out_f in ((1024, 1024), (1024, 3072)):
        w = torch.randn((out_f, in_f), device=device, dtype=dtype)
        lora_a = torch.randn((num_rec, rank, in_f), device=device, dtype=dtype) * 0.02
        lora_b = torch.randn((num_rec, out_f, rank), device=device, dtype=dtype) * 0.02

        for m in (64, 256, 1024):
            x = torch.randn((m, in_f), device=device, dtype=dtype)
            rec_ids = torch.arange(m, device=device) % num_rec
            reused_meta = prepare_recursion_groups(rec_ids, num_recursions=num_rec)

            def base_only() -> torch.Tensor:
                return F.linear(x, w)

            def naive_grouped_lora() -> torch.Tensor:
                out = F.linear(x, w)
                for k in range(num_rec):
                    mask = rec_ids == k
                    xk = x[mask]
                    out[mask] = out[mask] + F.linear(F.linear(xk, lora_a[k]), lora_b[k])
                return out

            def triton_sgmv_fresh() -> torch.Tensor:
                return triton_grouped_relaxed_linear(x, w, lora_a, lora_b, rec_ids=rec_ids, scaling=1.0)

            def triton_sgmv_reused() -> torch.Tensor:
                return triton_grouped_relaxed_linear(x, w, lora_a, lora_b, group_meta=reused_meta, scaling=1.0)

            # Warmup Triton JIT
            _ = triton_sgmv_fresh()
            torch.cuda.synchronize()

            t_base = bench_us(base_only, warmup=50, rep=150)
            t_naive = bench_us(naive_grouped_lora, warmup=50, rep=150)
            t_tri = bench_us(triton_sgmv_fresh, warmup=50, rep=150)
            t_tri_reused = bench_us(triton_sgmv_reused, warmup=50, rep=150)
            print(
                f"  {f'({m}, {in_f}->{out_f})':<20} {t_base:>9.1f}u {t_naive:>10.1f}u "
                f"{t_tri:>11.1f}u {t_tri_reused:>15.1f}u {t_naive / t_tri_reused:>9.2f}x {t_tri_reused / t_base:>8.2f}x"
            )


def benchmark_transformer_block(device: torch.device, dtype: torch.dtype) -> None:
    print("\n=== 5. Full RecursiveTransformerBlock (Qwen3-0.6B scale: d=1024, ff=3072, H=16, r=64) ===")
    print(f"{'Mode':<28} {'Fwd (us)':>12} {'Fwd+Bwd (us)':>14} {'Speedup Fwd':>13} {'Speedup F+B':>13}")
    print("-" * 84)

    bsz, seq_len = 8, 256  # 2048 tokens
    d_model, n_heads, d_ff = 1024, 16, 3072
    cos, sin = precompute_rope_freqs(d_model // n_heads, seq_len, device=device)

    strict_block = RecursiveTransformerBlock(
        d_model=d_model, n_heads=n_heads, d_ff=d_ff, num_recursions=4,
        lora_rank=0, lora_alpha=None, lora_dropout=0.0, dropout=0.0, share_norms=False,
    ).to(device=device, dtype=dtype)

    relaxed_split = RecursiveTransformerBlock(
        d_model=d_model, n_heads=n_heads, d_ff=d_ff, num_recursions=4,
        lora_rank=64, lora_alpha=64.0, lora_dropout=0.0, dropout=0.0, share_norms=False,
        lora_mode="split", fuse_projections=False,
    ).to(device=device, dtype=dtype)

    relaxed_fused_auto = RecursiveTransformerBlock(
        d_model=d_model, n_heads=n_heads, d_ff=d_ff, num_recursions=4,
        lora_rank=64, lora_alpha=64.0, lora_dropout=0.0, dropout=0.0, share_norms=False,
        lora_mode="auto", fuse_projections=True,
    ).to(device=device, dtype=dtype)

    relaxed_fused_triton = RecursiveTransformerBlock(
        d_model=d_model, n_heads=n_heads, d_ff=d_ff, num_recursions=4,
        lora_rank=64, lora_alpha=64.0, lora_dropout=0.0, dropout=0.0, share_norms=False,
        lora_mode="auto", fuse_projections=True, use_triton_kernels=True,
    ).to(device=device, dtype=dtype)

    relaxed_split_compiled = torch.compile(relaxed_split)
    relaxed_fused_compiled = torch.compile(relaxed_fused_auto)

    x_fwd = torch.randn((bsz, seq_len, d_model), device=device, dtype=dtype)
    grad_out = torch.randn_like(x_fwd)

    # Warmup compiled & custom Triton variants (fwd and fwd+bwd)
    for mod, raw in (
        (relaxed_fused_triton, relaxed_fused_triton),
        (relaxed_split_compiled, relaxed_split),
        (relaxed_fused_compiled, relaxed_fused_auto),
    ):
        y = mod(x_fwd, cos, sin, recursion_idx=0)
        y.backward(grad_out)
        raw.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    def measure_block(mod: torch.nn.Module) -> tuple[float, float]:
        with torch.no_grad():
            t_fwd = bench_us(lambda: mod(x_fwd, cos, sin, recursion_idx=0), warmup=100, rep=300)

        def fwd_bwd() -> None:
            mod.zero_grad(set_to_none=True)
            out = mod(x_fwd, cos, sin, recursion_idx=0)
            out.backward(grad_out)

        t_fb = bench_us(fwd_bwd, warmup=150, rep=400)
        return t_fwd, t_fb

    t_strict_f, t_strict_fb = measure_block(strict_block)
    t_rel_f, t_rel_fb = measure_block(relaxed_split)
    t_fused_f, t_fused_fb = measure_block(relaxed_fused_auto)
    t_trit_f, t_trit_fb = measure_block(relaxed_fused_triton)
    t_comp_f, t_comp_fb = measure_block(relaxed_split_compiled)
    t_fcomp_f, t_fcomp_fb = measure_block(relaxed_fused_compiled)

    rows = [
        ("strict_eager (r=0)", t_strict_f, t_strict_fb),
        ("relaxed_split_eager", t_rel_f, t_rel_fb),
        ("relaxed_fused_auto_eager", t_fused_f, t_fused_fb),
        ("relaxed_fused_triton", t_trit_f, t_trit_fb),
        ("relaxed_split_compile", t_comp_f, t_comp_fb),
        ("relaxed_fused_auto_compile", t_fcomp_f, t_fcomp_fb),
    ]
    for name, tf, tfb in rows:
        print(
            f"{name:<28} {tf:>12.1f} {tfb:>14.1f} "
            f"{t_rel_f / tf:>12.2f}x {t_rel_fb / tfb:>12.2f}x"
        )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark.py requires a CUDA GPU")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    print(f"GPU: {torch.cuda.get_device_name(0)} | PyTorch {torch.__version__} | Triton {triton.__version__}")

    benchmark_rmsnorm(device, dtype)
    benchmark_swiglu(device, dtype)
    benchmark_relaxed_linear(device, dtype)
    benchmark_depthwise_batching_naive(device, dtype)
    benchmark_transformer_block(device, dtype)


if __name__ == "__main__":
    main()
