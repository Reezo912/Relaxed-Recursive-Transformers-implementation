# Architecture & Hugging Face Conversion

This document details the core Relaxed Recursive Transformer (RRT) architecture implemented in `src/rrt/` and how pretrained Hugging Face models (Qwen3, Qwen2, Llama) are compressed in-place without breaking native KV caching or `generate()`.

## 1. Relaxed Parameter Sharing

In a **Strict Recursive Transformer**, $N$ virtual layers are built by repeating $U = N / R$ unique Transformer blocks $R$ times. Strict tying cuts layer weights by $R\times$, but forces different depths to apply identical projections.

A **Relaxed Recursive Transformer** (Bae et al., 2024) restores per-depth specialization by pairing each shared linear weight $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ with a lightweight, recursion-specific low-rank adapter $(A_k, B_k)$ and optional DoRA per-output-channel magnitude vector $m_k \in \mathbb{R}^{d_{\text{out}}}$:

$$y_k = m_k \odot \left( x W^\top + \frac{\alpha}{r} (x A_k^\top) B_k^\top \right)$$

`RelaxedLinear` (`src/rrt/layers.py`) and `SharedWeightLoRALinear` (`src/rrt/hf_converter.py`) also support automatic **split vs. merge** dispatch (`lora_mode="auto"`):
- **Small token counts (decode / short sequences):** split path $x W^\top + \frac{\alpha}{r} (x A_k^\top) B_k^\top$ avoids materializing $W_{\text{eff}}$.
- **Large token counts (training / prefill):** merge path $W_{\text{eff}} = W + \frac{\alpha}{r} B_k A_k$ followed by a single large GEMM. The threshold is derived analytically from FLOP counts in `lora_merge_token_threshold(d_in, d_out)`.

Each virtual depth retains its own **unshared `RMSNorm` weights** (negligible parameter cost, essential for residual stream scale alignment across depths).

## 2. Layer-Sharing Strategies

Implemented in `build_layer_groups()` (`src/rrt/hf_converter.py`) and `RRTConfig` (`src/rrt/model.py`):

| Strategy | Tied Layer Pattern (example: 12 layers, $R=2$) | Characteristics |
| :--- | :--- | :--- |
| `cycle` | `(0, 6), (1, 7), ..., (5, 11)` | Cycles through the full block stack $R$ times |
| `sequence` | `(0, 1), (2, 3), ..., (10, 11)` | Repeats adjacent layers consecutively |
| `middle_cycle` | Unshared edges + cyclic tied core | Preserves entry/exit representations |
| `middle_sequence` | Unshared edges + adjacent tied core | Best default for pretrained LLMs (adjacent layers have highest weight cosine similarity) |

### Protecting Early Super-Weights (`keep_first`, `keep_last`)

Pretrained LLMs concentrate critical "super-weight" activations in the earliest decoder layers (typically layers `0..2` in Qwen3-1.7B) and task-specific output shaping in the final layer. `build_layer_groups()` supports asymmetric unshared boundaries such as `--keep-first 3 --keep-last 1`:

- On **`Qwen/Qwen3-1.7B`** (28 layers, `middle_sequence`, $R=2$, `keep_first=3`, `keep_last=1`, `lora_rank=128`, DoRA enabled):
  - Layers `0, 1, 2` and `27` remain dedicated unshared layers.
  - Layers `3..26` form 12 tied adjacent pairs: `(3, 4), (5, 6), ..., (25, 26)`.
  - **Transformer layer parameters:** `1,409.4M` $\rightarrow$ **`925.5M` (65.7%)**.
  - **Total parameters (including embeddings):** `1,720.6M` $\rightarrow$ **`1,236.6M` (71.9%)**.

## 3. Activation-Aware SVD Initialization (ASVD + QR/SVD Balancing)

When converting a pretrained model (`convert_hf_model_to_relaxed` in `src/rrt/hf_converter.py`), naive zero-initialized LoRA starts far from the source weights. Truncated SVD initializes each tied group $\{W_0, \dots, W_{R-1}\}$ directly from the pretrained checkpoints:

1. **Diagonal Activation Scaling (`_collect_input_activation_rms`, `_compute_activation_scales`)**:
   - Forward hooks record per-channel input RMS $\sigma_{k, j} = \sqrt{\mathbb{E}[x_{k, j}^2]}$ across shared input spaces (`attn_in`, `attn_o_in`, `mlp_in`, `mlp_down_in`) using calibration batches.
   - Scales are normalized to unit geometric mean and clamped to `[0.35, 3.5]` (`condition <= 10`). Unlike inverting a full $D \times D$ Cholesky covariance matrix, condition-bounded diagonal scaling avoids weight and LoRA blowup on ill-conditioned attention channels.
2. **Prior-Anchored Convex Base Weight**:
   - Channel $j$ of the shared base weight is the convex combination:
     $$W_{\text{base}, :, j} = \frac{\sum_k \sigma_{k, j}^2 W_{k, :, j} + \lambda \bar{\sigma}_j^2 \bar{W}_{:, j}}{\sum_k \sigma_{k, j}^2 + \lambda \bar{\sigma}_j^2}$$
   - Every entry of $W_{\text{base}}$ is strictly bounded within $[\min_k W_k, \max_k W_k]$.
3. **Weighted Truncated SVD + Exact QR/SVD Factor Rebalancing (`_rebalance_low_rank_factors`)**:
   - Each residual $\Delta_k = W_k - W_{\text{base}}$ is scaled by $\operatorname{diag}(s_k)$, decomposed via rank-$r$ truncated SVD, and unscaled back into weight space.
   - Unscaling only one factor causes $\|A_k\|_F \gg \|B_k\|_F$ (poorly conditioned for BF16 optimizers). `_rebalance_low_rank_factors` computes thin QR decompositions $B = Q_B R_B$ and $A^\top = Q_A R_A$, followed by a tiny $r \times r$ SVD of $R_B R_A^\top = U_r \Sigma_r V_r^\top$, setting:
     $$B_{\text{bal}} = Q_B U_r \sqrt{\Sigma_r}, \qquad A_{\text{bal}} = \sqrt{\Sigma_r} V_r^\top Q_A^\top$$
   - This preserves $B_{\text{bal}} A_{\text{bal}} = B A$ exactly while guaranteeing $\|A_{\text{bal}}\|_F = \|B_{\text{bal}}\|_F$ across every layer.

### Optional Architectural Extensions

- **DoRA Output Scaling (`--dora-scale`)**: Adds a learnable 1D per-output-channel scale vector (`out_scale`, initialized to `1.0`) to every `SharedWeightLoRALinear` projection, decoupling output magnitude from direction at `<0.1%` parameter cost.
- **U-Shaped Adaptive Rank (`--adaptive-rank` / `--no-adaptive-rank`)**: `compute_adaptive_layer_ranks()` redistributes the exact same parameter budget into Tensor-Core-aligned (multiples of 16) ranks giving higher rank to outer tied layers and lower rank to middle layers. For `Qwen3-1.7B` with `keep_first=3, keep_last=1`, uniform `r=128` (`--no-adaptive-rank`) starts with lower step-0 perplexity and is used by default in our 2-stage recipe.

## 4. In-Place Hugging Face Sharing & Native KV Cache

Instead of rewriting Hugging Face model classes, `convert_hf_model_to_relaxed()` replaces each tied `nn.Linear` projection in `model.model.layers` with a `SharedWeightLoRALinear` referencing the **exact same `nn.Parameter` object** (`weight.data_ptr()` identical across paired layers).

This brings three practical advantages:
- PyTorch's `model.parameters()` and optimizers deduplicate tied weights automatically.
- Hugging Face's native per-layer KV cache, RoPE, QK-Norm, FlashAttention/SDPA, activation checkpointing, and `model.generate()` work unmodified.
- Checkpoints store deduplicated trainable tensors in `bfloat16` alongside full RRT metadata so `scripts/chat.py` and `scripts/evaluate.py` reconstruct the exact architecture automatically.

## 5. Continuous Depth-Wise Batching & Early Exit

`ContinuousDepthwiseEngine` (`src/rrt/depthwise.py`) implements token-level **early exit** for the native `RelaxedRecursiveTransformer`:
- Each slot in a decode batch tracks its current recursion depth $k \in \{0, \dots, R-1\}$.
- When a token's confidence exceeds threshold $\tau$, it exits early, copies its KV states into remaining recursion caches (KV cache copy from Bae et al., 2024), and immediately begins the next token at $k=0$ on the following iteration.
- Because tokens within the same decode step sit at different recursion depths $k$, all active tokens share one base GEMM and execute their per-depth LoRA updates in a single fused **Triton Grouped/Segmented LoRA (SGMV)** kernel (`src/rrt/kernels/grouped_lora.py`).
