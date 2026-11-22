# Relaxed Recursive Transformers (PyTorch + Triton)

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Language:** **English** | [Español](README.es.md)

A GPU-optimized PyTorch + Triton implementation of **Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA** ([Bae et al., 2024, arXiv:2410.20672](https://arxiv.org/abs/2410.20672)), extended with in-place Hugging Face compression (**Qwen3 / Qwen2 / Llama**), a single-16GB-GPU **2-stage Strong-to-Weak distillation pipeline**, custom **OpenAI Triton** kernels, **Continuous Depth-Wise Batching**, and a zero-dependency live web dashboard.

Recursive Transformers reduce model memory by looping a smaller stack of unique Transformer blocks $R$ times. To recover the capacity lost from strict weight tying, **Relaxed Recursive Transformers** attach lightweight, recursion-specific LoRA adapters $(A_k, B_k)$ (and optional per-channel DoRA magnitude scaling $m_k$) to every shared base weight $W$:

$$y_k = m_k \odot \left( x W^\top + \frac{\alpha}{r} (x A_k^\top) B_k^\top \right)$$

---

## Highlights

- **Core RRT Library (`src/rrt/`)**: `cycle`, `sequence`, `middle_cycle`, and `middle_sequence` weight-sharing schedules; per-depth `RMSNorm`, RoPE, SwiGLU; automatic FLOP-optimal split/merge LoRA dispatch; multi-exit training.
- **In-Place Hugging Face Conversion (`src/rrt/hf_converter.py`)**: Compresses `Qwen3` / `Qwen2` / `Llama` models while keeping native per-layer KV cache, FlashAttention/SDPA, activation checkpointing, and `generate()` intact. Includes **condition-bounded diagonal ASVD**, exact **QR/SVD LoRA factor balancing** ($\|A\|_F = \|B\|_F$), asymmetric super-weight protection (`--keep-first`, `--keep-last`), optional **DoRA** scaling, and budget-preserving **U-shaped adaptive rank**.
- **Custom Triton Kernels (`src/rrt/kernels/`)**:
  - **Fused `TritonRMSNorm`** (persistent backward without global atomics — **7.40×** vs eager PyTorch).
  - **Zero-copy 2D strided `triton_swiglu`** operating directly on fused Gate+Up `.chunk()` views (**1.44×** vs eager).
  - **Grouped / Segmented LoRA SGMV** for mixed-depth Continuous Depth-Wise Batching (**up to 11.54×** vs Python loop).
- **2-Stage Strong-to-Weak Distillation (`scripts/distill.py`)**: Compresses `Qwen/Qwen3-1.7B` $\rightarrow$ `1.24B` (`65.7%` layer params) on a single 16 GB GPU using **Stochastic BF16 Adafactor**, memory-exact **chunked Forward KL / Generalized JSD**, **int8 weight-only Teacher** (`Qwen/Qwen3-4B`), **Ridge-initialized DSKD cross-architecture hidden projectors**, **On-Policy Distillation (OPD)** rollouts, **WSD** schedule, and zero-VRAM **CPU Weight EMA**.
- **Live Web Dashboard (`scripts/dashboard.py`)**: Zero-dependency local UI with multi-run auto-discovery, baseline comparison, EMA smoothing, automatic health alerts, interactive samples timeline, WSD/EMA phase markers, GPU sparkline, and live log tail.

---

## Repository Layout

```text
.
├── src/rrt/
│   ├── layers.py               # RelaxedLinear (split/merge/auto), RMSNorm, RoPE
│   ├── model.py                # RRTConfig, RelaxedRecursiveTransformer, QKV/Gate+Up fusion
│   ├── conversion.py           # Truncated SVD conversion for native models
│   ├── hf_converter.py         # In-place HF converter (ASVD, QR/SVD balance, DoRA, adaptive rank)
│   ├── distill.py              # StochasticBF16Adafactor, chunked KL/JSD, int8 teacher, DSKD Ridge, CPU-EMA
│   ├── depthwise.py            # ContinuousDepthwiseEngine (early exit + KV copy + Triton SGMV)
│   └── kernels/
│       ├── rmsnorm.py          # Fused Triton RMSNorm (fwd + atomic-free persistent bwd)
│       ├── swiglu.py           # Zero-copy 2D strided Triton SwiGLU
│       └── grouped_lora.py     # Grouped/Segmented LoRA SGMV (fwd + bwd)
├── scripts/
│   ├── distill.py              # Single-GPU Strong-to-Weak knowledge distillation engine
│   ├── run_2stage_distill.sh   # Ready-to-run 2-stage pipeline (Qwen3-1.7B -> Qwen3-4B int8)
│   ├── dashboard.py            # Live local training monitor & diagnostics web UI
│   ├── evaluate.py             # Quantitative benchmark suite (PPL, ARC-Easy, BoolQ, HellaSwag, ablation)
│   ├── chat.py                 # Interactive CLI chat (single RRT or side-by-side vs original HF model)
│   ├── benchmark_kernels.py    # Triton kernel & fused Transformer block micro-benchmarks
│   ├── demo_depthwise.py       # Multi-exit training + Continuous Depth-Wise Batching demo
│   ├── train_scratch.py        # From-scratch comparison: Standard vs Strict vs Relaxed
│   ├── uptrain_local.py        # Local truncated SVD vs zero-init uptraining experiment
│   └── uptrain_hf.py           # Quick HF SVD conversion + WikiText-2 uptraining (Qwen3-0.6B-Base)
├── docs/
│   ├── architecture.md         # RRT sharing schedules, ASVD + QR/SVD balancing, DoRA, Depth-Wise Batching
│   ├── triton-kernels.md       # Triton kernel design and RTX 4080 micro-benchmarks
│   └── distillation.md         # 2-stage Strong-to-Weak distillation recipe and dashboard guide
├── tests/                      # 80 unit tests (model, SVD, HF Qwen3, distill, Triton kernels, depthwise)
├── pyproject.toml
├── LICENSE
├── README.md
└── README.es.md
```

---

## Installation & Tests

Requires Python `3.11–3.13` and [`uv`](https://docs.astral.sh/uv/) (NVIDIA CUDA GPU recommended for Triton kernels and distillation).

```bash
uv sync --python 3.11
TRITON_DEFAULT_FP_FUSION=1 uv run pytest -v
```

Verified on an **NVIDIA RTX 4080 (16 GB)**: **80/80 tests passing** (`torch 2.15.0+cu132`, `triton 3.9.0`, `transformers 5.24.0`).

---

## Quick Start

### 1. 2-Stage Strong-to-Weak Distillation (`Qwen/Qwen3-1.7B` $\rightarrow$ `1.24B`)

Runs Stage 1 self-healing (`Qwen3-1.7B` teacher, Forward KL + same-arch hidden alignment) followed automatically by Stage 2 Strong-to-Weak distillation (`Qwen3-4B int8` teacher, Generalized JSD + Ridge-initialized DSKD hidden projector + On-Policy rollouts + CPU-EMA):

```bash
./scripts/run_2stage_distill.sh
# On desktop Linux, prevent suspend during long runs:
# systemd-inhibit --what=idle:sleep ./scripts/run_2stage_distill.sh
```

Or run `scripts/distill.py` directly with custom flags (see [`docs/distillation.md`](docs/distillation.md) for all options).

### 2. Live Web Dashboard

```bash
uv run python scripts/dashboard.py --port 7860
# Open http://localhost:7860
```

### 3. Benchmark & Chat with Compressed Checkpoints

```bash
# Objective comparison: Original Qwen3-1.7B vs RRT vs Strict (LoRA OFF) ablation
uv run python scripts/evaluate.py --checkpoint checkpoints/qwen3_1.7b_rrt_s2_4b/best.pt --n-bench 200

# Interactive side-by-side chat
uv run python scripts/chat.py --checkpoint checkpoints/qwen3_1.7b_rrt_s2_4b/best.pt --compare
```

### 4. Research Demos & GPU Kernel Benchmarks

```bash
# Standard vs Strict Recursive vs Relaxed Recursive from scratch
uv run python scripts/train_scratch.py --dataset shakespeare --steps 300 --eval-interval 100

# Truncated SVD vs zero-init uptraining (local model)
uv run python scripts/uptrain_local.py --pretrain-steps 400 --uptrain-steps 200 --lora-rank 32

# Quick HF uptraining on Qwen/Qwen3-0.6B-Base (WikiText-2)
uv run python scripts/uptrain_hf.py --model Qwen/Qwen3-0.6B-Base --sharing-strategy middle_sequence --num-recursions 2 --lora-rank 64 --steps 120

# Custom Triton kernels & fused Transformer block micro-benchmarks
TRITON_DEFAULT_FP_FUSION=1 uv run python scripts/benchmark_kernels.py

# Multi-exit training + Continuous Depth-Wise Batching with Triton SGMV
TRITON_DEFAULT_FP_FUSION=1 uv run python scripts/demo_depthwise.py
```

---

## Verified Results (NVIDIA RTX 4080 16 GB, `bfloat16`)

### 1. Truncated SVD Conversion & Uptraining

**Local Transformer (`scripts/uptrain_local.py`, 12 virtual layers, 4× fewer unique blocks, `r=32`):**

| Variant | Parameters | Size vs Std | Val Loss @ Step 0 | Val Loss @ 200 Steps |
| :--- | ---: | ---: | ---: | ---: |
| `standard_source` | 9,509,376 | 100.0% | 1.5723 | 1.5723 |
| `strict_mean` | 2,394,624 | 25.2% | 6.1301 | 1.7836 |
| `relaxed_zero_init` | 4,268,544 | 44.9% | 6.1301 | 1.7491 |
| **`relaxed_svd_init`** | **4,268,544** | **44.9%** | **1.8976** | **1.5727** |

**Hugging Face Models:**
- **`Qwen/Qwen3-0.6B-Base`** (`scripts/uptrain_hf.py`, `middle_sequence`, $R=2$, `r=64`, WikiText-2): Layer params `440.5M` $\rightarrow$ **`273.5M` (62.1%)** (`7.42 GB` VRAM with `fp32` AdamW); WikiText-2 PPL drops from `4.46M` at step 0 to **`100.18`** in 120 steps (`76.2s`).
- **`Qwen/Qwen3-1.7B`** (`scripts/distill.py`, `middle_sequence`, $R=2$, `keep_first=3`, `keep_last=1`, `r=128` + DoRA): Layer params `1,409.4M` $\rightarrow$ **`925.5M` (65.7%)**, total `1.721B` $\rightarrow$ **`1.237B` (71.9%)**. Baseline 10h single-stage run (`v1`, 112.6M tokens, `8.04 GB` VRAM, ~`3,150 tok/s`) reached `Val KL = 0.6461` (`Val PPL = 9.96` vs Teacher `14.54`). With diagonal ASVD + QR/SVD balancing (`29.4s` conversion, `W-err 0.673 -> 0.527`, exact $\|A\|_F / \|B\|_F = 1.000$), Stage 1 reaches `chat_ppl = 16.46` (Teacher `11.48`) by step 200 (`1.64M` tokens).

### 2. Custom Triton Kernels & Structural Fusion (`scripts/benchmark_kernels.py`)

- **`TritonRMSNorm` (`4096 × 1024`, Fwd+Bwd):** `507.9 µs` (Eager) / `107.5 µs` (`torch.compile`) $\rightarrow$ **`68.6 µs`** (**7.40×** vs Eager, **1.57×** vs compile).
- **`triton_swiglu` (`4096 × 6144`, zero-copy `.chunk`, Fwd+Bwd):** `1539.3 µs` $\rightarrow$ **`1070.1 µs`** (**1.44×** vs Eager).
- **Grouped LoRA SGMV ($R=4, r=64$, reused metadata):**

| Shape `(M, D_in -> D_out)` | Base GEMM | PyTorch Loop | **Triton SGMV** | Speedup vs PyTorch |
| :--- | ---: | ---: | ---: | ---: |
| `(64, 1024 -> 1024)` | `14.3 µs` | `278.8 µs` | **`25.6 µs`** | **10.89×** |
| `(256, 1024 -> 1024)` | `11.3 µs` | `283.5 µs` | **`24.6 µs`** | **11.54×** |
| `(1024, 1024 -> 3072)` | `80.8 µs` | `405.7 µs` | **`106.5 µs`** | **3.81×** |

- **Full Transformer Block (`8 × 256 = 2048` tokens, `d=1024, d_ff=3072, r=64`):**
  - `relaxed_split_eager`: `1244.2 µs` Fwd / `3786.8 µs` Fwd+Bwd (`1.00×`)
  - `relaxed_fused_triton`: **`1002.7 µs` Fwd (1.24×)** / **`3094.5 µs` Fwd+Bwd (1.22×)**
  - `relaxed_fused_auto_compile`: **`822.3 µs` Fwd (1.51×)** / **`2725.9 µs` Fwd+Bwd (1.39×)**

### 3. Continuous Depth-Wise Batching with Early Exit (`scripts/demo_depthwise.py`, `B=64`, `R=4`)

| Mode | Avg Depth | Iterations | Time (ms) | Tok/s | Speedup | Exit Histogram (1..4) |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| Full Depth (no early exit) | `4.00/4` | `256` | `495.5` | `8,267` | `1.00×` | `[0, 0, 0, 4096]` |
| Early Exit ($\tau = 0.75$) | `3.08/4` | `220` | `481.3` | `8,510` | `1.03×` | `[1180, 99, 38, 2779]` |
| **Early Exit ($\tau = 0.60$)** | **`2.73/4`** | **`194`** | **`428.4`** | **`9,560`** | **`1.16×`** | **`[1679, 75, 24, 2318]`** |

*(In a static 64-sequence batch, wall-clock speedup is bounded by the slowest sequence; under continuous request arrival, throughput scales directly with $R / \text{avg depth}$, ~`1.47×` at $\tau=0.60$.)*

---

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — Layer sharing schedules, diagonal ASVD + QR/SVD factor balancing, DoRA scaling, in-place HF conversion, and Continuous Depth-Wise Batching.
- [`docs/triton-kernels.md`](docs/triton-kernels.md) — Fused RMSNorm, zero-copy 2D SwiGLU, Grouped LoRA SGMV internals, and full RTX 4080 benchmarks.
- [`docs/distillation.md`](docs/distillation.md) — 2-stage Strong-to-Weak distillation, Stochastic BF16 Adafactor, chunked KL/JSD, DSKD Ridge projector, On-Policy Distillation, CPU-EMA, and Dashboard usage.

## Citation & License

Based on:

```bibtex
@article{bae2024relaxed,
  title   = {Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA},
  author  = {Bae, Sangmin and Kim, Yujin and Bayat, Reza and Kim, Sungnyun and Ha, Jiyoun and Schuster, Tal and Fisch, Adam and Harutyunyan, Hrayr and Ji, Ziwei and Courville, Aaron and Yun, Se-Young},
  journal = {arXiv preprint arXiv:2410.20672},
  year    = {2024}
}
```

Released under the [MIT License](LICENSE).
