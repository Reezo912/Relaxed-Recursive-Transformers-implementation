from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch

from rrt import ContinuousDepthwiseEngine, RRTConfig, RelaxedRecursiveTransformer
from train_scratch import load_dataset, sample_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a multi-exit Relaxed Recursive Transformer and benchmark Continuous Depth-Wise Batching (Triton SGMV)"
    )
    parser.add_argument("--dataset", choices=["shakespeare", "synthetic"], default="shakespeare")
    parser.add_argument("--train-steps", type=int, default=350)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-recursions", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--decode-batch", type=int, default=64)
    parser.add_argument("--prompt-len", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    dataset = load_dataset(args.dataset, Path("data"))
    cfg = RRTConfig(
        vocab_size=dataset.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        max_seq_len=max(128, args.prompt_len + args.max_new_tokens),
        num_layers=args.num_layers,
        num_recursions=args.num_recursions,
        sharing_strategy="cycle",
        lora_rank=args.lora_rank,
        lora_mode="auto",
        fuse_projections=True,
        use_triton_kernels=(device.type == "cuda"),
        dropout=0.05,
    )
    model = RelaxedRecursiveTransformer(cfg).to(device)
    params = model.parameter_summary()
    print(
        f"Model: {len(model.blocks)} unique blocks x {cfg.num_recursions} recursions = "
        f"{cfg.num_layers} virtual layers | total_params={params['total']:,} (lora={params['lora']:,})"
    )

    # 1. Quick multi-exit training so intermediate recursions emit calibrated logits
    print(f"\nTraining with multi-exit supervision ({args.train_steps} steps)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.05)
    warmup = max(10, args.train_steps // 10)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, args.train_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    t0 = time.perf_counter()
    model.train()
    for step in range(1, args.train_steps + 1):
        x, y = sample_batch(dataset.train_ids, args.batch_size, 128, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            _, loss = model(x, y, multi_exit_loss=True)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % 100 == 0 or step == args.train_steps:
            print(f"  step {step:4d}/{args.train_steps} | multi_exit_loss={loss.item():.4f}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"Multi-exit training finished in {time.perf_counter() - t0:.1f}s")

    # 2. Switch to bf16 inference and benchmark Full-Depth vs Continuous Depth-Wise Batching
    if use_bf16:
        model = model.to(dtype=torch.bfloat16)
    model.eval()
    engine = ContinuousDepthwiseEngine(model)

    # Build a batch of real prompts from validation split
    starts = torch.linspace(0, len(dataset.val_ids) - args.prompt_len - 1, args.decode_batch).long()
    offsets = torch.arange(args.prompt_len)
    prompts = dataset.val_ids[starts.unsqueeze(1) + offsets.unsqueeze(0)].to(device)

    # Warmup Triton SGMV kernels
    _ = engine.generate(prompts[:4], max_new_tokens=4, exit_threshold=0.80, temperature=0.0)

    thresholds = [
        ("Full Depth (no early exit)", 1.01),
        ("Early Exit (tau=0.90)", 0.90),
        ("Early Exit (tau=0.75)", 0.75),
        ("Early Exit (tau=0.60)", 0.60),
    ]

    print(
        f"\n=== Continuous Depth-Wise Batching Benchmark (Triton SGMV, B={args.decode_batch}, "
        f"new_tokens={args.max_new_tokens}, R={cfg.num_recursions}) ==="
    )
    print(f"{'Mode':<28} {'Avg Depth':>10} {'Loops':>7} {'Time (ms)':>10} {'Tok/s':>10} {'Speedup':>9}  Exit Histogram (1..R)")
    print("-" * 102)

    base_tps = None
    samples: list[tuple[str, str]] = []
    for label, tau in thresholds:
        seqs, stats = engine.generate(
            prompts,
            max_new_tokens=args.max_new_tokens,
            exit_threshold=tau,
            temperature=0.0,
        )
        if base_tps is None:
            base_tps = stats.tokens_per_sec
        speedup = stats.tokens_per_sec / base_tps
        print(
            f"{label:<28} {stats.mean_exit_depth:>7.2f}/{cfg.num_recursions:<2} "
            f"{stats.total_loop_iterations:>7d} {stats.elapsed_sec * 1000:>10.1f} "
            f"{stats.tokens_per_sec:>10,.0f} {speedup:>8.2f}x  {stats.exit_depth_histogram}"
        )
        samples.append((label, dataset.decode(seqs[0]).replace("\n", "\\n")))

    print("\nSample output (slot 0):")
    for label, text in samples:
        print(f"  [{label}] {text[:115]}")


if __name__ == "__main__":
    main()
