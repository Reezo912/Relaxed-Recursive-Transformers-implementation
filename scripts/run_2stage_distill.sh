#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p checkpoints/qwen3_1.7b_rrt_s1 checkpoints/qwen3_1.7b_rrt_s2_4b
rm -f checkpoints/qwen3_1.7b_rrt_s1/*.pt checkpoints/qwen3_1.7b_rrt_s1/metrics.jsonl
rm -f checkpoints/qwen3_1.7b_rrt_s2_4b/*.pt checkpoints/qwen3_1.7b_rrt_s2_4b/metrics.jsonl

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_DEFAULT_FP_FUSION=1

echo "=== Stage 1: Qwen3-1.7B self-heal (ASVD + DoRA + uniform r=128 + Forward KL + WSD + CPU-EMA) ==="
uv run python scripts/distill.py \
  --model Qwen/Qwen3-1.7B \
  --dataset-mix elite \
  --sharing-strategy middle_sequence \
  --num-recursions 2 \
  --keep-first 3 \
  --keep-last 1 \
  --lora-rank 128 \
  --no-adaptive-rank \
  --dora-scale \
  --kl-type forward \
  --lr-schedule wsd \
  --hidden-weight 2.0 \
  --lr 1.5e-4 \
  --lora-lr-mult 2.0 \
  --ema-start 0.70 \
  --ema-decay 0.95 \
  --max-hours 0.6 \
  --eval-interval 200 \
  --out-dir checkpoints/qwen3_1.7b_rrt_s1 \
  2>&1 | tee checkpoints/qwen3_1.7b_rrt_s1.log

echo "=== Stage 2: Strong-to-Weak Qwen3-4B int8 (DSKD Ridge + JSD beta=0.25 + On-Policy + WSD + CPU-EMA) ==="
uv run python scripts/distill.py \
  --model Qwen/Qwen3-1.7B \
  --init-from checkpoints/qwen3_1.7b_rrt_s1/best.pt \
  --teacher-model Qwen/Qwen3-4B \
  --teacher-int8 \
  --dataset-mix elite \
  --no-adaptive-rank \
  --dora-scale \
  --kl-type jsd \
  --jsd-beta 0.25 \
  --lr-schedule wsd \
  --cross-arch-hidden \
  --cross-arch-hidden-weight 1.0 \
  --on-policy-every 12 \
  --on-policy-batch 2 \
  --on-policy-tokens 64 \
  --on-policy-jsd-beta 0.1 \
  --lr 1.0e-4 \
  --lora-lr-mult 2.0 \
  --ema-start 0.70 \
  --ema-decay 0.95 \
  --max-hours 1.5 \
  --eval-interval 150 \
  --out-dir checkpoints/qwen3_1.7b_rrt_s2_4b \
  2>&1 | tee checkpoints/qwen3_1.7b_rrt_s2_4b.log
