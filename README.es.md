# Relaxed Recursive Transformers (PyTorch + Triton)

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Idioma:** [English](README.md) | **Español**

Implementación optimizada para GPU en PyTorch + Triton de **Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA** ([Bae et al., 2024, arXiv:2410.20672](https://arxiv.org/abs/2410.20672)), ampliada con conversión in-place de modelos de Hugging Face (**Qwen3 / Qwen2 / Llama**), un **pipeline de destilación Strong-to-Weak en 2 etapas** diseñado para una sola GPU de 16 GB, kernels custom en **OpenAI Triton**, **Continuous Depth-Wise Batching** y un dashboard web local sin dependencias externas.

Un Recursive Transformer ahorra memoria repitiendo un conjunto reducido de bloques de capas $R$ veces. Para recuperar la capacidad perdida al compartir pesos de forma estricta, **Relaxed Recursive Transformers** añade adaptadores LoRA independientes por paso de recursión $(A_k, B_k)$ (y opcionalmente escalado de magnitud por canal tipo DoRA $m_k$) sobre cada matriz base compartida $W$:

$$y_k = m_k \odot \left( x W^\top + \frac{\alpha}{r} (x A_k^\top) B_k^\top \right)$$

---

## Características principales

- **Librería núcleo RRT (`src/rrt/`)**: estrategias de compartición `cycle`, `sequence`, `middle_cycle` y `middle_sequence`; `RMSNorm` independiente por profundidad, RoPE y SwiGLU; selección automática `split`/`merge` óptima en FLOPs; entrenamiento multi-salida (*early-exit*).
- **Conversión in-place de Hugging Face (`src/rrt/hf_converter.py`)**: comprime modelos `Qwen3` / `Qwen2` / `Llama` manteniendo intactos el KV cache nativo por capa, FlashAttention/SDPA, *activation checkpointing* y `generate()`. Incluye **ASVD diagonal con condición acotada**, **rebalanceo exacto QR/SVD de LoRA** ($\|A\|_F = \|B\|_F$), protección asimétrica de capas con *super-weights* (`--keep-first`, `--keep-last`), escalado **DoRA** opcional y rango adaptativo en U preservando el presupuesto de parámetros.
- **Kernels custom en Triton (`src/rrt/kernels/`)**:
  - **`TritonRMSNorm` fusionado** (*backward* persistente sin operaciones atómicas globales — **7.40×** frente a PyTorch eager).
  - **`triton_swiglu` 2D por strides zero-copy** directo sobre vistas `.chunk()` de proyecciones Gate+Up (**1.44×** vs eager).
  - **Grouped / Segmented LoRA SGMV** para *Continuous Depth-Wise Batching* con profundidades mixtas (**hasta 11.54×** vs bucle PyTorch).
- **Destilación Strong-to-Weak en 2 etapas (`scripts/distill.py`)**: comprime `Qwen/Qwen3-1.7B` $\rightarrow$ `1.24B` (`65.7%` de parámetros en capas) en una GPU de 16 GB usando **Stochastic BF16 Adafactor**, **Forward KL / Generalized JSD por chunks** exacto en memoria, **Teacher `int8` weight-only** (`Qwen/Qwen3-4B`), **proyectores cross-arch DSKD inicializados por Ridge**, rollouts **On-Policy Distillation (OPD)**, calendario **WSD** y **CPU Weight EMA** con 0 VRAM extra.
- **Dashboard web en vivo (`scripts/dashboard.py`)**: servidor local sin dependencias externas con auto-descubrimiento de corridas, comparación contra baselines, suavizado EMA, monitor automático de salud, línea temporal interactiva de respuestas, marcadores de fase WSD/EMA, *sparkline* de telemetría GPU y visor de logs en directo.

---

## Estructura del repositorio

```text
.
├── src/rrt/
│   ├── layers.py               # RelaxedLinear (split/merge/auto), RMSNorm, RoPE
│   ├── model.py                # RRTConfig, RelaxedRecursiveTransformer, fusión QKV/Gate+Up
│   ├── conversion.py           # Conversión SVD truncada para el modelo nativo
│   ├── hf_converter.py         # Conversor HF in-place (ASVD, balance QR/SVD, DoRA, rango adaptativo)
│   ├── distill.py              # StochasticBF16Adafactor, KL/JSD por chunks, teacher int8, DSKD Ridge, CPU-EMA
│   ├── depthwise.py            # ContinuousDepthwiseEngine (early-exit + copia KV + Triton SGMV)
│   └── kernels/
│       ├── rmsnorm.py          # Kernel Triton Fused RMSNorm (fwd + bwd persistente sin atomics)
│       ├── swiglu.py           # Kernel Triton SwiGLU 2D por strides (zero-copy)
│       └── grouped_lora.py     # Kernel Triton Grouped/Segmented LoRA SGMV (fwd + bwd)
├── scripts/
│   ├── distill.py              # Motor de destilación Strong-to-Weak para GPU de 16 GB
│   ├── run_2stage_distill.sh   # Pipeline completo en 2 etapas (Qwen3-1.7B -> Qwen3-4B int8)
│   ├── dashboard.py            # Dashboard web local de monitoreo y diagnóstico en vivo
│   ├── evaluate.py             # Suite de evaluación (PPL, ARC-Easy, BoolQ, HellaSwag, ablación LoRA)
│   ├── chat.py                 # Chat interactivo CLI (modelo RRT o cara a cara vs original)
│   ├── benchmark_kernels.py    # Micro-benchmarks GPU de kernels Triton y bloque fusionado
│   ├── demo_depthwise.py       # Entrenamiento multi-salida + demo Continuous Depth-Wise Batching
│   ├── train_scratch.py        # Comparativa desde cero: Standard vs Strict vs Relaxed
│   ├── uptrain_local.py        # Experimento local: SVD truncado vs zero-init + uptraining
│   └── uptrain_hf.py           # Conversión SVD + uptraining rápido en WikiText-2 (Qwen3-0.6B-Base)
├── docs/
│   ├── architecture.md         # Estrategias de compartición, ASVD + balance QR/SVD, DoRA, Depth-Wise Batching
│   ├── triton-kernels.md       # Diseño de kernels Triton y benchmarks en RTX 4080
│   └── distillation.md         # Receta Strong-to-Weak en 2 etapas y guía del dashboard
├── tests/                      # 80 tests unitarios (modelo, SVD, HF Qwen3, destilación, Triton, depthwise)
├── pyproject.toml
├── LICENSE
├── README.md
└── README.es.md
```

---

## Instalación y tests

Requiere Python `3.11–3.13` y [`uv`](https://docs.astral.sh/uv/) (se recomienda GPU NVIDIA CUDA para los kernels Triton y la destilación).

```bash
uv sync --python 3.11
TRITON_DEFAULT_FP_FUSION=1 uv run pytest -v
```

Verificado en **NVIDIA RTX 4080 (16 GB)**: **80/80 tests en verde** (`torch 2.15.0+cu132`, `triton 3.9.0`, `transformers 5.24.0`).

---

## Uso rápido

### 1. Destilación Strong-to-Weak en 2 etapas (`Qwen/Qwen3-1.7B` $\rightarrow$ `1.24B`)

Ejecuta Stage 1 (curación con `Qwen3-1.7B`, Forward KL + alineación de estados ocultos 1-a-1) y encadena automáticamente Stage 2 (Strong-to-Weak con `Qwen3-4B int8`, JSD generalizada + proyector DSKD inicializado por Ridge + rollouts On-Policy + CPU-EMA):

```bash
./scripts/run_2stage_distill.sh
# En Linux de escritorio, para evitar suspensión durante entrenamientos largos:
# systemd-inhibit --what=idle:sleep ./scripts/run_2stage_distill.sh
```

También puedes ejecutar `scripts/distill.py` directamente con flags personalizados (consulta [`docs/distillation.md`](docs/distillation.md)).

### 2. Dashboard web en vivo

```bash
uv run python scripts/dashboard.py --port 7860
# Abrir http://localhost:7860
```

### 3. Evaluación y chat con checkpoints comprimidos

```bash
# Comparativa objetiva: Qwen3-1.7B original vs RRT vs ablación Strict (LoRA OFF)
uv run python scripts/evaluate.py --checkpoint checkpoints/qwen3_1.7b_rrt_s2_4b/best.pt --n-bench 200

# Chat interactivo cara a cara
uv run python scripts/chat.py --checkpoint checkpoints/qwen3_1.7b_rrt_s2_4b/best.pt --compare
```

### 4. Demos de investigación y benchmarks de kernels GPU

```bash
# Standard vs Strict Recursive vs Relaxed Recursive desde cero
uv run python scripts/train_scratch.py --dataset shakespeare --steps 300 --eval-interval 100

# Conversión SVD truncado vs zero-init + uptraining (modelo local)
uv run python scripts/uptrain_local.py --pretrain-steps 400 --uptrain-steps 200 --lora-rank 32

# Uptraining rápido de Qwen/Qwen3-0.6B-Base en WikiText-2
uv run python scripts/uptrain_hf.py --model Qwen/Qwen3-0.6B-Base --sharing-strategy middle_sequence --num-recursions 2 --lora-rank 64 --steps 120

# Micro-benchmarks de kernels Triton y bloque Transformer fusionado
TRITON_DEFAULT_FP_FUSION=1 uv run python scripts/benchmark_kernels.py

# Entrenamiento multi-salida + Continuous Depth-Wise Batching con Triton SGMV
TRITON_DEFAULT_FP_FUSION=1 uv run python scripts/demo_depthwise.py
```

---

## Resultados verificados (NVIDIA RTX 4080 16 GB, `bfloat16`)

### 1. Conversión por SVD truncado y destilación

**Modelo local (`scripts/uptrain_local.py`, 12 capas virtuales, 4× menos bloques físicos, `r=32`):**

| Variante | Parámetros | Tamaño vs Std | Val Loss @ Paso 0 | Val Loss tras 200 pasos |
| :--- | ---: | ---: | ---: | ---: |
| `standard_source` | 9,509,376 | 100.0% | 1.5723 | 1.5723 |
| `strict_mean` | 2,394,624 | 25.2% | 6.1301 | 1.7836 |
| `relaxed_zero_init` | 4,268,544 | 44.9% | 6.1301 | 1.7491 |
| **`relaxed_svd_init`** | **4,268,544** | **44.9%** | **1.8976** | **1.5727** |

**Modelos Hugging Face:**
- **`Qwen/Qwen3-0.6B-Base`** (`scripts/uptrain_hf.py`, `middle_sequence`, $R=2$, `r=64`, WikiText-2): parámetros en capas `440.5M` $\rightarrow$ **`273.5M` (62.1%)** (`7.42 GB` VRAM con `fp32` AdamW); PPL en WikiText-2 baja de `4.46M` en el paso 0 a **`100.18`** tras 120 pasos (`76.2s`).
- **`Qwen/Qwen3-1.7B`** (`scripts/distill.py`, `middle_sequence`, $R=2$, `keep_first=3`, `keep_last=1`, `r=128` + DoRA): parámetros en capas `1,409.4M` $\rightarrow$ **`925.5M` (65.7%)**, totales `1.721B` $\rightarrow$ **`1.237B` (71.9%)**. La corrida baseline de 10h (`v1`, 112.6M tokens, `8.04 GB` VRAM, ~`3,150 tok/s`) alcanzó `Val KL = 0.6461` (`Val PPL = 9.96` frente al Teacher `14.54`). Con ASVD diagonal + balance QR/SVD (`29.4s` de conversión, `W-err 0.673 -> 0.527`, balance exacto $\|A\|_F / \|B\|_F = 1.000$), Stage 1 alcanza `chat_ppl = 16.46` (Teacher `11.48`) ya en el paso 200 (`1.64M` tokens).

### 2. Kernels custom en Triton y fusión estructural (`scripts/benchmark_kernels.py`)

- **`TritonRMSNorm` (`4096 × 1024`, Fwd+Bwd):** `507.9 µs` (Eager) / `107.5 µs` (`torch.compile`) $\rightarrow$ **`68.6 µs`** (**7.40×** vs Eager, **1.57×** vs compile).
- **`triton_swiglu` (`4096 × 6144`, vista `.chunk` zero-copy, Fwd+Bwd):** `1539.3 µs` $\rightarrow$ **`1070.1 µs`** (**1.44×** vs Eager).
- **Grouped LoRA SGMV ($R=4, r=64$, metadatos reutilizados):**

| Forma `(M, D_in -> D_out)` | GEMM Base | Bucle PyTorch | **Triton SGMV** | Speedup vs PyTorch |
| :--- | ---: | ---: | ---: | ---: |
| `(64, 1024 -> 1024)` | `14.3 µs` | `278.8 µs` | **`25.6 µs`** | **10.89×** |
| `(256, 1024 -> 1024)` | `11.3 µs` | `283.5 µs` | **`24.6 µs`** | **11.54×** |
| `(1024, 1024 -> 3072)` | `80.8 µs` | `405.7 µs` | **`106.5 µs`** | **3.81×** |

- **Bloque Transformer completo (`8 × 256 = 2048` tokens, `d=1024, d_ff=3072, r=64`):**
  - `relaxed_split_eager`: `1244.2 µs` Fwd / `3786.8 µs` Fwd+Bwd (`1.00×`)
  - `relaxed_fused_triton`: **`1002.7 µs` Fwd (1.24×)** / **`3094.5 µs` Fwd+Bwd (1.22×)**
  - `relaxed_fused_auto_compile`: **`822.3 µs` Fwd (1.51×)** / **`2725.9 µs` Fwd+Bwd (1.39×)**

### 3. Continuous Depth-Wise Batching con Early-Exit (`scripts/demo_depthwise.py`, `B=64`, `R=4`)

| Modo | Profundidad media | Iteraciones | Tiempo (ms) | Tok/s | Speedup | Histograma salidas (1..4) |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| Full Depth (sin early-exit) | `4.00/4` | `256` | `495.5` | `8,267` | `1.00×` | `[0, 0, 0, 4096]` |
| Early Exit ($\tau = 0.75$) | `3.08/4` | `220` | `481.3` | `8,510` | `1.03×` | `[1180, 99, 38, 2779]` |
| **Early Exit ($\tau = 0.60$)** | **`2.73/4`** | **`194`** | **`428.4`** | **`9,560`** | **`1.16×`** | **`[1679, 75, 24, 2318]`** |

*(En un batch estático de 64 secuencias el speedup total está acotado por la secuencia más lenta; en una cola continua donde cada slot libre recibe una petición nueva al instante, la ganancia tiende directamente a $R / \text{profundidad media}$, ~`1.47×` con $\tau=0.60$.)*

---

## Documentación técnica

- [`docs/architecture.md`](docs/architecture.md) — Estrategias de compartición, ASVD diagonal + balance QR/SVD, DoRA, conversión in-place de HF y Continuous Depth-Wise Batching.
- [`docs/triton-kernels.md`](docs/triton-kernels.md) — Diseño de kernels Triton (`RMSNorm`, `SwiGLU`, `SGMV`) y benchmarks completos en RTX 4080.
- [`docs/distillation.md`](docs/distillation.md) — Pipeline Strong-to-Weak en 2 etapas, Stochastic BF16 Adafactor, KL/JSD por chunks, proyector DSKD Ridge, On-Policy Distillation, CPU-EMA y guía del dashboard.

## Referencia y licencia

Basado en:

```bibtex
@article{bae2024relaxed,
  title   = {Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA},
  author  = {Bae, Sangmin and Kim, Yujin and Bayat, Reza and Kim, Sungnyun and Ha, Jiyoun and Schuster, Tal and Fisch, Adam and Harutyunyan, Hrayr and Ji, Ziwei and Courville, Aaron and Yun, Se-Young},
  journal = {arXiv preprint arXiv:2410.20672},
  year    = {2024}
}
```

Publicado bajo [Licencia MIT](LICENSE).
