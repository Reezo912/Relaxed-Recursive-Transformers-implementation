from __future__ import annotations

import argparse
import math
import random
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch

from rrt import RRTConfig, RelaxedRecursiveTransformer

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


@dataclass
class CharDataset:
    train_ids: torch.Tensor
    val_ids: torch.Tensor
    itos: list[str]
    stoi: dict[str, int]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def build_synthetic_text(num_lines: int = 8000, seed: int = 42) -> str:
    """Structured synthetic corpus (modular arithmetic + reversals + templates) requiring multi-step computation."""
    rng = random.Random(seed)
    lines: list[str] = []
    words = ["alpha", "beta", "gamma", "delta", "omega", "sigma", "theta", "kappa"]
    for _ in range(num_lines):
        task = rng.randint(0, 2)
        if task == 0:
            a, b, c = rng.randint(10, 99), rng.randint(10, 99), rng.randint(1, 9)
            lines.append(f"calc: {a} + {b} - {c} = {a + b - c}")
        elif task == 1:
            seq = [rng.choice(words) for _ in range(4)]
            lines.append(f"rev: {' '.join(seq)} -> {' '.join(reversed(seq))}")
        else:
            nums = [rng.randint(0, 9) for _ in range(5)]
            lines.append(f"sort: {''.join(map(str, nums))} -> {''.join(map(str, sorted(nums)))}")
    return "\n".join(lines) + "\n"


def load_dataset(kind: str, data_dir: Path) -> CharDataset:
    if kind == "shakespeare":
        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / "tinyshakespeare.txt"
        if not path.exists():
            try:
                print(f"Downloading TinyShakespeare to {path}...")
                urllib.request.urlretrieve(SHAKESPEARE_URL, path)
            except Exception as exc:
                print(f"Download failed ({exc}); falling back to synthetic dataset.")
                kind = "synthetic"
        if kind == "shakespeare":
            text = path.read_text(encoding="utf-8")
        else:
            text = build_synthetic_text()
    elif kind == "synthetic":
        text = build_synthetic_text()
    else:
        raise ValueError(f"Unknown dataset: {kind}")

    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    split = int(0.9 * len(ids))
    return CharDataset(train_ids=ids[:split], val_ids=ids[split:], itos=chars, stoi=stoi)


def sample_batch(
    data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(data) - seq_len - 1
    starts = torch.randint(0, max_start, (batch_size,))
    offsets = torch.arange(seq_len)
    idx = starts.unsqueeze(1) + offsets.unsqueeze(0)
    x = data[idx].to(device, non_blocking=True)
    y = data[idx + 1].to(device, non_blocking=True)
    return x, y


@torch.no_grad()
def evaluate(
    model: RelaxedRecursiveTransformer,
    val_ids: torch.Tensor,
    batch_size: int,
    seq_len: int,
    eval_batches: int,
    device: torch.device,
    use_bf16: bool,
) -> float:
    model.eval()
    losses: list[float] = []
    for _ in range(eval_batches):
        x, y = sample_batch(val_ids, batch_size, seq_len, device)
        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            _, loss = model(x, y)
        assert loss is not None
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train_variant(
    name: str,
    config: RRTConfig,
    dataset: CharDataset,
    steps: int,
    batch_size: int,
    lr: float,
    eval_interval: int,
    eval_batches: int,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = RelaxedRecursiveTransformer(config).to(device)
    params = model.parameter_summary()
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
    warmup_steps = max(10, steps // 10)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(
        f"\n=== {name} === "
        f"(unique_blocks={len(model.blocks)}, virtual_layers={config.num_layers}, "
        f"recursions={config.num_recursions}, lora_rank={config.lora_rank}, "
        f"total_params={params['total']:,}, lora_params={params['lora']:,})"
    )

    best_val = float("inf")
    last_train = float("inf")
    t0 = time.perf_counter()

    model.train()
    for step in range(1, steps + 1):
        x, y = sample_batch(dataset.train_ids, batch_size, config.max_seq_len, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            _, loss = model(x, y)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        last_train = loss.item()

        if step == 1 or step % eval_interval == 0 or step == steps:
            val_loss = evaluate(model, dataset.val_ids, batch_size, config.max_seq_len, eval_batches, device, use_bf16)
            best_val = min(best_val, val_loss)
            print(f"  step {step:4d}/{steps} | train_loss={last_train:.4f} | val_loss={val_loss:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    toks_per_sec = (steps * batch_size * config.max_seq_len) / max(elapsed, 1e-6)

    lora_stats = model.lora_relaxation_stats() if config.lora_rank > 0 and config.num_recursions > 1 else []
    if lora_stats:
        summary_str = ", ".join(f"L{s['virtual_layer']}:{s['mean_rel_lora_norm']:.3f}" for s in lora_stats)
        print(f"  LoRA rel-norm (||dW||/||W||) per virtual layer: {summary_str}")

    prompt_text = dataset.decode(dataset.val_ids[:24].tolist())
    prompt_ids = torch.tensor([dataset.encode(prompt_text)], dtype=torch.long, device=device)
    sample_ids = model.generate(prompt_ids, max_new_tokens=80, temperature=0.8, top_k=20)[0].tolist()
    sample_preview = dataset.decode(sample_ids).replace("\n", "\\n")

    return {
        "name": name,
        "params": params,
        "unique_blocks": len(model.blocks),
        "train_loss": last_train,
        "val_loss": val_loss,
        "best_val": best_val,
        "toks_per_sec": toks_per_sec,
        "elapsed": elapsed,
        "sample": sample_preview,
    }


def build_variant_configs(args: argparse.Namespace, vocab_size: int) -> dict[str, RRTConfig]:
    base = dict(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        max_seq_len=args.seq_len,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    configs: dict[str, RRTConfig] = {
        "standard": RRTConfig(**base, num_recursions=1, sharing_strategy="cycle", lora_rank=0),
        "strict_recursive": RRTConfig(
            **base, num_recursions=args.num_recursions, sharing_strategy="cycle", lora_rank=0
        ),
        "relaxed_recursive": RRTConfig(
            **base,
            num_recursions=args.num_recursions,
            sharing_strategy="cycle",
            lora_rank=args.lora_rank,
        ),
    }

    # Include middle_cycle when (num_layers - 2) is divisible by num_recursions (or using a compatible recursion count)
    if args.num_layers >= 2 + args.num_recursions and (args.num_layers - 2) % args.num_recursions == 0:
        configs["relaxed_middle_cycle"] = RRTConfig(
            **base,
            num_recursions=args.num_recursions,
            sharing_strategy="middle_cycle",
            lora_rank=args.lora_rank,
        )
    return configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Standard, Strict Recursive, and Relaxed Recursive Transformers")
    parser.add_argument("--dataset", choices=["shakespeare", "synthetic"], default="shakespeare")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-recursions", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["standard", "strict_recursive", "relaxed_recursive"],
        help="Subset of variants to run (standard, strict_recursive, relaxed_recursive, relaxed_middle_cycle)",
    )
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

    all_configs = build_variant_configs(args, dataset.vocab_size)
    results: list[dict[str, object]] = []
    for variant in args.variants:
        if variant not in all_configs:
            raise ValueError(f"Variant '{variant}' unavailable for current layer/recursion config. Available: {list(all_configs)}")
        res = train_variant(
            name=variant,
            config=all_configs[variant],
            dataset=dataset,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            eval_interval=args.eval_interval,
            eval_batches=args.eval_batches,
            device=device,
            seed=args.seed,
        )
        results.append(res)

    standard_total = next((r["params"]["total"] for r in results if r["name"] == "standard"), None)  # type: ignore[index]

    print("\n" + "=" * 92)
    print(f"{'Variant':<22} {'Blocks':>6} {'Total Params':>14} {'LoRA Params':>12} {'Size%':>7} {'Best Val':>9} {'Tok/s':>10}")
    print("-" * 92)
    for r in results:
        p = r["params"]  # type: ignore[assignment]
        pct = f"{100.0 * p['total'] / standard_total:5.1f}%" if standard_total else "  n/a "
        print(
            f"{r['name']:<22} {r['unique_blocks']:>6} {p['total']:>14,} {p['lora']:>12,} "
            f"{pct:>7} {r['best_val']:>9.4f} {r['toks_per_sec']:>10,.0f}"
        )
    print("=" * 92)

    print("\nSample generations:")
    for r in results:
        print(f"  [{r['name']}] {r['sample'][:110]}")


if __name__ == "__main__":
    main()
