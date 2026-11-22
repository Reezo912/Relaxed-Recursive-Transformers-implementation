# 2-Stage Strong-to-Weak Distillation Pipeline

`scripts/distill.py` + `src/rrt/distill.py` implement a memory-efficient knowledge distillation pipeline engineered to compress **`Qwen/Qwen3-1.7B`** (~`1.24B` params, `65.7%` layer params) on a single **16 GB GPU** while preserving—or improving—downstream quality.

## 1. Two-Stage Recipe (`scripts/run_2stage_distill.sh`)

```text
Pretrained Qwen3-1.7B
   │
   ▼  [ASVD Diagonal Init + QR/SVD LoRA Balancing + DoRA]
Compressed RRT Student (1.237B total / 65.7% layer params)
   │
   ▼  [Stage 1: Self-Heal — Frozen Qwen3-1.7B Teacher (bf16)]
      • Forward KL (0.9, T=1.5) + CE (0.1) + Same-Arch Hidden Alignment (2.0)
      • Uniform r=128 + DoRA, LoRA+ (2x LR), WSD schedule, CPU-EMA @ 70%
   │
   ▼  [Stage 2: Strong-to-Weak — Frozen Qwen3-4B Teacher (int8 weight-only)]
      • Generalized JSD (beta=0.25, canonically normalized) + CE (0.1)
      • DSKD Cross-Arch Hidden Projector (2048 -> 2560, Ridge-initialized before step 1)
      • On-Policy Distillation rollouts (every 12 steps, reverse-skewed JSD beta=0.1)
      • WSD schedule + Zero-VRAM CPU Weight EMA Model Soup @ 70%
```

### Why Two Stages?

1. **Stage 1 (Self-Heal)** rapidly repairs structural compression error using exact 1-to-1 same-architecture hidden supervision and mode-covering Forward KL against the original `Qwen3-1.7B` teacher.
2. **Stage 2 (Strong-to-Weak)** transfers higher-capacity reasoning, factuality, and sharpness from **`Qwen/Qwen3-4B`** (quantized on the fly to symmetric per-channel `int8` weight-only via `quantize_model_int8_weight_only`, cutting teacher decoder VRAM by ~50% with `< 2%` logit relative error).

## 2. Core Algorithmic Components (`src/rrt/distill.py`)

### Stochastic BF16 Adafactor (`StochasticBF16Adafactor`)
- Keeps weights, gradients, and optional momentum (`beta1=0.9`) in `bfloat16` while computing factored row/column RMS second moments and parameter updates in `fp32` one tensor at a time.
- Writes updates back to `bfloat16` using **exact unbiased stochastic rounding** (`fp32_to_bf16_stochastic`), preventing small gradient updates from stalling under round-to-nearest truncation while fitting Student + Teacher + activations inside ~10–12 GB VRAM.

### Memory-Exact Chunked KL / JSD (`chunked_kl_ce_backward`)
- Qwen3's `151,936`-token vocabulary would consume gigabytes if full `(B, T, V)` `fp32` logits were materialized at once.
- `chunked_kl_ce_backward` projects hidden states through `lm_head` in slices (`--chunk-tokens 128`), accumulates exact `dL/dh_s` into a leaf tensor, and runs a single combined autograd backward pass with optional `aux_loss` (hidden-state alignment).
- Supports **Forward KL** and **Generalized Jensen-Shannon Divergence** (`JSD_beta`, GKD / DistiLLM):
  $$\operatorname{JSD}_\beta^{\text{norm}}(P_T \| P_S) = \frac{1}{\beta(1-\beta)} \Big[\beta \operatorname{KL}(P_T \| M) + (1-\beta)\operatorname{KL}(P_S \| M)\Big], \quad M = \beta P_T + (1-\beta) P_S$$
  Dividing by $\beta(1-\beta)$ normalizes second-order curvature near $P_S \approx P_T$ so gradient scale matches Forward KL across any $\beta \in (0, 1)$.

### DSKD Cross-Architecture Hidden Alignment + Ridge Init (`CrossArchHiddenProjector`)
- Aligns 6 evenly spaced student/teacher depth pairs (`(5, 6), (9, 12), ..., (28, 36)` for `Qwen3-1.7B -> Qwen3-4B`) via training-only linear projectors (`2048 -> 2560`, discarded at inference: 0 extra params).
- Before step 1 of Stage 2, `fit_ridge_()` solves the closed-form ridge regression $W_k = (H_t^\top H_s)(H_s^\top H_s + \lambda I)^{-1}$ on 4 calibration batches (`< 1s` on GPU), eliminating early random-projector gradient shock to the already-healed Stage-1 student.

### On-Policy Distillation (`--on-policy-every`)
- Periodically samples real user prompts from SmolTalk, lets the student autoregressively generate its own responses (`use_cache=True`), and grades **only generated assistant tokens** (`token_mask=response_mask`, `ce_weight=0`) against the frozen teacher using reverse-skewed JSD (`--on-policy-jsd-beta 0.1`), directly fixing exposure bias.

### Zero-VRAM CPU Weight EMA (`CPUWeightEMA`)
- Maintains an `fp32` exponential moving average (Model Soup) of deduplicated trainable weights on host RAM starting at `--ema-start 0.70` (during WSD stable + cosine decay).
- At run completion, evaluates the EMA soup on validation blocks, saves `ema.pt`, and automatically promotes it to `best.pt` if it beats the best raw validation KL.

## 3. Data Mixes (`--dataset-mix`)

All datasets stream directly from locally cached Parquet shards via PyArrow (constant RAM footprint):

- **`elite` (default):**
  - `45%` FineWeb-Edu (`HuggingFaceFW/fineweb-edu`)
  - `25%` SmolTalk (`HuggingFaceTB/smoltalk`, formatted with Qwen3 chat template)
  - `15%` Cosmopedia-v2 (`HuggingFaceTB/smollm-corpus`)
  - `10%` FineMath-4plus (`HuggingFaceTB/finemath`)
  - `5%` English Wikipedia (`wikimedia/wikipedia`)
- **`basic`:** FineWeb-Edu + SmolTalk (`--chat-prob 0.30`).

## 4. Live Web Dashboard (`scripts/dashboard.py`)

Zero-dependency local HTTP monitoring server (`stdlib` + HTML5 Canvas):

```bash
uv run python scripts/dashboard.py --port 7860
```

Features:
- Auto-discovers all runs under `checkpoints/` with **Auto-follow latest**, baseline **Compare**, and **Crop compare** to match token budgets.
- **EMA smoothing presets** (`OFF / 0.5 / 0.8 / 0.9`) and toggleable **Log scale** on training curves.
- **Automatic Health Monitor** detecting NaN/Inf, loss spikes, plateaus, throughput drops, VRAM/temperature pressure, recent loss slope, OPD JSD, and CPU-EMA promotions.
- **Same-token interpolated comparison bar** against any selected baseline run.
- **WSD decay (`85%`) and CPU-EMA (`70%`) phase markers** across progress bar and charts.
- **Interactive Samples Timeline** scrubbing model replies from Step 0 to latest alongside Teacher reference outputs.
- **Live GPU telemetry sparkline** (VRAM, temperature, power, utilization) and live `.log` tail viewer.
