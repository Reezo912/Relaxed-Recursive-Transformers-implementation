from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch

from rrt import RRTConfig, RelaxedRecursiveTransformer, convert_standard_to_relaxed
from train_scratch import CharDataset, evaluate, load_dataset, sample_batch


def run_training_loop(
    model: RelaxedRecursiveTransformer,
    dataset: CharDataset,
    steps: int,
    batch_size: int,
    lr: float,
    eval_interval: int,
    eval_batches: int,
    device: torch.device,
    seed: int,
    label: str,
) -> tuple[float, float, float]:
    """Train or uptrain a model and return (val_loss_step0, best_val_loss, final_val_loss)."""
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    val_step0 = evaluate(model, dataset.val_ids, batch_size, model.config.max_seq_len, eval_batches, device, use_bf16)
    best_val = val_step0
    print(f"  [{label}] step    0/{steps} | val_loss={val_step0:.4f} (before training)")

    if steps <= 0:
        return val_step0, best_val, val_step0

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
    warmup_steps = max(5, steps // 10)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    last_val = val_step0
    for step in range(1, steps + 1):
        x, y = sample_batch(dataset.train_ids, batch_size, model.config.max_seq_len, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            _, loss = model(x, y)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % eval_interval == 0 or step == steps:
            last_val = evaluate(
                model, dataset.val_ids, batch_size, model.config.max_seq_len, eval_batches, device, use_bf16
            )
            best_val = min(best_val, last_val)
            print(
                f"  [{label}] step {step:4d}/{steps} | train_loss={loss.item():.4f} | "
                f"val_loss={last_val:.4f} | lr={scheduler.get_last_lr()[0]:.2e}"
            )

    return val_step0, best_val, last_val


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain Standard Transformer -> SVD Convert -> Uptrain Comparison")
    parser.add_argument("--dataset", choices=["shakespeare", "synthetic"], default="shakespeare")
    parser.add_argument("--pretrain-steps", type=int, default=400)
    parser.add_argument("--uptrain-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-recursions", type=int, default=4)
    parser.add_argument("--sharing-strategy", choices=["cycle", "middle_cycle"], default="cycle")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--base-init", choices=["mean", "first", "middle", "last"], default="mean")
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--uptrain-lr", type=float, default=5e-4)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    dataset = load_dataset(args.dataset, Path("data"))
    print(
        f"Dataset: {args.dataset} | vocab_size={dataset.vocab_size} | "
        f"train_tokens={len(dataset.train_ids):,} | val_tokens={len(dataset.val_ids):,}"
    )

    # 1. Pretrain standard (non-recursive) source Transformer
    std_cfg = RRTConfig(
        vocab_size=dataset.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        max_seq_len=args.seq_len,
        num_layers=args.num_layers,
        num_recursions=1,
        lora_rank=0,
        dropout=0.05,
    )
    torch.manual_seed(args.seed)
    source = RelaxedRecursiveTransformer(std_cfg).to(device)
    std_params = source.parameter_summary()["total"]

    print(f"\n=== Phase 1: Pretraining Standard Source ({args.num_layers} unique layers, {std_params:,} params) ===")
    t0 = time.perf_counter()
    _, std_best_val, std_final_val = run_training_loop(
        model=source,
        dataset=dataset,
        steps=args.pretrain_steps,
        batch_size=args.batch_size,
        lr=args.pretrain_lr,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        device=device,
        seed=args.seed,
        label="standard_source",
    )
    print(f"Source pretraining finished in {time.perf_counter() - t0:.1f}s (final val_loss={std_final_val:.4f})")

    # 2. Convert source into three compressed recursive variants and uptrain each
    experiments = [
        ("strict_mean", 0, False),
        ("relaxed_zero_init", args.lora_rank, False),
        ("relaxed_svd_init", args.lora_rank, True),
    ]

    summary_rows: list[dict[str, object]] = [
        {
            "name": "standard_source",
            "params": std_params,
            "err_after": 0.0,
            "energy": 100.0,
            "val_step0": std_final_val,
            "val_best": std_best_val,
            "val_final": std_final_val,
        }
    ]

    print(
        f"\n=== Phase 2: Converting ({args.sharing_strategy}, recursions={args.num_recursions}, "
        f"base_init={args.base_init}, lora_rank={args.lora_rank}) + Uptraining ({args.uptrain_steps} steps) ==="
    )

    for name, rank, use_svd in experiments:
        converted, report = convert_standard_to_relaxed(
            source=source,
            num_recursions=args.num_recursions,
            sharing_strategy=args.sharing_strategy,
            lora_rank=rank,
            base_init=args.base_init,
            use_svd=use_svd,
        )
        p_total = converted.parameter_summary()["total"]
        print(
            f"\n--- {name} | params={p_total:,} ({100.0 * p_total / std_params:.1f}%) | "
            f"weight_err: {report.mean_rel_error_strict:.3f} -> {report.mean_rel_error_relaxed:.3f} | "
            f"SVD energy: {100.0 * report.mean_energy_captured:.1f}% ---"
        )
        val0, val_best, val_final = run_training_loop(
            model=converted,
            dataset=dataset,
            steps=args.uptrain_steps,
            batch_size=args.batch_size,
            lr=args.uptrain_lr,
            eval_interval=args.eval_interval,
            eval_batches=args.eval_batches,
            device=device,
            seed=args.seed + 100,
            label=name,
        )
        summary_rows.append(
            {
                "name": name,
                "params": p_total,
                "err_after": report.mean_rel_error_relaxed,
                "energy": 100.0 * report.mean_energy_captured,
                "val_step0": val0,
                "val_best": val_best,
                "val_final": val_final,
            }
        )

    print("\n" + "=" * 98)
    print(
        f"{'Variant':<20} {'Params':>11} {'Size%':>7} {'W-Err':>7} {'SVD E%':>7} "
        f"{'Val@0 (post-cvt)':>17} {'Val@Best':>10} {'Val@End':>10}"
    )
    print("-" * 98)
    for row in summary_rows:
        pct = 100.0 * row["params"] / std_params  # type: ignore[operator]
        print(
            f"{row['name']:<20} {row['params']:>11,} {pct:>6.1f}% {row['err_after']:>7.3f} "
            f"{row['energy']:>6.1f}% {row['val_step0']:>17.4f} {row['val_best']:>10.4f} {row['val_final']:>10.4f}"
        )
    print("=" * 98)


if __name__ == "__main__":
    main()
