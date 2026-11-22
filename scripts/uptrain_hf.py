from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from datasets import load_dataset as hf_load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from rrt import SharedWeightLoRALinear, convert_hf_model_to_relaxed, hf_parameter_summary


def build_token_chunks(
    tokenizer: AutoTokenizer,
    split: str,
    seq_len: int,
    max_tokens: int | None = None,
) -> torch.Tensor:
    """Tokenize WikiText-2 into contiguous blocks of shape (num_blocks, seq_len)."""
    ds = hf_load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    texts = [row["text"] for row in ds if row["text"].strip()]
    joined = "\n\n".join(texts)
    token_ids = tokenizer(joined, add_special_tokens=False)["input_ids"]
    if max_tokens is not None:
        token_ids = token_ids[:max_tokens]
    n_blocks = len(token_ids) // seq_len
    if n_blocks == 0:
        raise ValueError(f"Not enough tokens ({len(token_ids)}) for seq_len={seq_len}")
    trimmed = token_ids[: n_blocks * seq_len]
    return torch.tensor(trimmed, dtype=torch.long).view(n_blocks, seq_len)


@torch.no_grad()
def evaluate_ppl(
    model: torch.nn.Module,
    val_blocks: torch.Tensor,
    batch_size: int,
    max_batches: int,
    device: torch.device,
    use_bf16: bool,
) -> tuple[float, float]:
    """Compute validation cross-entropy loss and perplexity."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for start in range(0, len(val_blocks), batch_size):
        if n_batches >= max_batches:
            break
        batch = val_blocks[start : start + batch_size].to(device, non_blocking=True)
        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            out = model(input_ids=batch, labels=batch, use_cache=False)
        total_loss += out.loss.item()
        n_batches += 1
    mean_loss = total_loss / max(n_batches, 1)
    ppl = math.exp(min(mean_loss, 20.0))
    return mean_loss, ppl


@torch.no_grad()
def sample_generation(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 48,
) -> str:
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text.replace("\n", " ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a Hugging Face LLM (e.g. Qwen3-0.6B-Base) via SVD + Uptrain")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--num-recursions", type=int, default=2)
    parser.add_argument(
        "--sharing-strategy",
        choices=["cycle", "middle_cycle", "sequence", "middle_sequence"],
        default="middle_sequence",
    )
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--base-init", choices=["mean", "first", "middle", "last"], default="mean")
    parser.add_argument("--no-svd", action="store_true", help="Disable SVD init (zero-init LoRA ablation)")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=24)
    parser.add_argument("--freeze-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prompt", type=str, default="The theory of relativity states that")
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    print(f"\nLoading tokenizer and model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device)
    orig_summary = hf_parameter_summary(model)
    orig_non_emb = orig_summary["total"] - orig_summary["embeddings"]
    print(
        f"Original params: total={orig_summary['total']:,} "
        f"(embeddings={orig_summary['embeddings']:,}, transformer_layers={orig_non_emb:,})"
    )

    print("\nPreparing WikiText-2 train/validation splits...")
    val_blocks = build_token_chunks(tokenizer, split="validation", seq_len=args.seq_len)
    train_blocks = build_token_chunks(tokenizer, split="train", seq_len=args.seq_len)
    print(f"WikiText-2 blocks (seq_len={args.seq_len}): train={len(train_blocks):,}, val={len(val_blocks):,}")

    # 1. Baseline evaluation before compression
    base_loss, base_ppl = evaluate_ppl(model, val_blocks, args.batch_size, args.eval_batches, device, use_bf16)
    base_gen = sample_generation(model, tokenizer, args.prompt, device)
    print(f"\n[Baseline {args.model}] val_loss={base_loss:.4f} | val_ppl={base_ppl:.2f}")
    print(f"  sample: {base_gen}")

    # 2. Convert in-place via truncated SVD
    use_svd = not args.no_svd
    print(
        f"\nConverting model in-place: strategy={args.sharing_strategy}, recursions={args.num_recursions}, "
        f"lora_rank={args.lora_rank}, base_init={args.base_init}, use_svd={use_svd}"
    )
    t_cvt = time.perf_counter()
    report = convert_hf_model_to_relaxed(
        model,
        num_recursions=args.num_recursions,
        sharing_strategy=args.sharing_strategy,
        lora_rank=args.lora_rank,
        base_init=args.base_init,
        use_svd=use_svd,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    cvt_summary = hf_parameter_summary(model)
    cvt_non_emb = cvt_summary["total"] - cvt_summary["embeddings"]
    print(
        f"Conversion done in {time.perf_counter() - t_cvt:.2f}s | "
        f"weight_err: {report.mean_rel_error_strict:.3f} -> {report.mean_rel_error_relaxed:.3f} | "
        f"SVD energy: {100.0 * report.mean_energy_captured:.1f}%"
    )
    print(
        f"Compressed params: total={cvt_summary['total']:,} ({100.0 * cvt_summary['total'] / orig_summary['total']:.1f}% total) | "
        f"transformer_layers={cvt_non_emb:,} ({100.0 * cvt_non_emb / orig_non_emb:.1f}% of original layers) | "
        f"shared_base={cvt_summary['shared_base']:,}, lora={cvt_summary['lora']:,}"
    )

    # 3. Zero-shot post-conversion evaluation (before any uptraining)
    step0_loss, step0_ppl = evaluate_ppl(model, val_blocks, args.batch_size, args.eval_batches, device, use_bf16)
    step0_gen = sample_generation(model, tokenizer, args.prompt, device)
    print(f"\n[Step 0 post-conversion] val_loss={step0_loss:.4f} | val_ppl={step0_ppl:.2f}")
    print(f"  sample: {step0_gen}")

    if args.steps <= 0:
        return

    # 4. Prepare precision & memory layout for stable uptraining on 16GB GPU:
    # Keep frozen embeddings in bf16; cast trainable parameters to fp32 master weights
    # and run forward/backward under bf16 autocast to avoid bf16 update underflow.
    if args.freeze_embeddings:
        for name, param in model.named_parameters():
            if "embed_tokens" in name or "lm_head" in name:
                param.requires_grad = False

    trainable_params: list[torch.nn.Parameter] = []
    seen_ids: set[int] = set()
    for param in model.parameters():
        if id(param) in seen_ids:
            continue
        seen_ids.add(id(param))
        if param.requires_grad:
            param.data = param.data.float()
            trainable_params.append(param)

    if args.grad_checkpointing:
        if hasattr(model, "activation_checkpointing_enable"):
            model.activation_checkpointing_enable()
        else:
            model.gradient_checkpointing_enable()

    trainable_count = sum(p.numel() for p in trainable_params)
    print(
        f"\nStarting uptraining: steps={args.steps}, eff_batch={args.batch_size * args.grad_accum}x{args.seq_len} "
        f"({args.batch_size * args.grad_accum * args.seq_len:,} tok/step), trainable_params={trainable_count:,}"
    )

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    warmup = max(5, args.steps // 10)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, args.steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    t0 = time.perf_counter()
    model.train()
    best_ppl = step0_ppl
    best_loss = step0_loss

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(args.grad_accum):
            idx = torch.randint(0, len(train_blocks), (args.batch_size,))
            batch = train_blocks[idx].to(device, non_blocking=True)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                out = model(input_ids=batch, labels=batch, use_cache=False)
                loss = out.loss / args.grad_accum
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        scheduler.step()

        if step % args.eval_interval == 0 or step == args.steps:
            val_loss, val_ppl = evaluate_ppl(model, val_blocks, args.batch_size, args.eval_batches, device, use_bf16)
            best_loss = min(best_loss, val_loss)
            best_ppl = min(best_ppl, val_ppl)
            peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
            print(
                f"  step {step:4d}/{args.steps} | train_loss={accum_loss:.4f} | "
                f"val_loss={val_loss:.4f} | val_ppl={val_ppl:.2f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e} | peak_vram={peak_gb:.2f}GB"
            )
            model.train()

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    total_tokens = args.steps * args.batch_size * args.grad_accum * args.seq_len
    tok_per_sec = total_tokens / max(elapsed, 1e-6)

    # Measure mean relative LoRA norm across converted layers
    lora_norms = [m.relative_lora_norm() for m in model.modules() if isinstance(m, SharedWeightLoRALinear)]
    mean_lora_norm = sum(lora_norms) / len(lora_norms) if lora_norms else 0.0

    final_gen = sample_generation(model, tokenizer, args.prompt, device)
    print("\n" + "=" * 90)
    print(f"Model: {args.model} -> {args.sharing_strategy} (R={args.num_recursions}, r={args.lora_rank}, svd={use_svd})")
    print(
        f"Layer params: {orig_non_emb:,} -> {cvt_non_emb:,} ({100.0 * cvt_non_emb / orig_non_emb:.1f}%) | "
        f"Total params: {orig_summary['total']:,} -> {cvt_summary['total']:,}"
    )
    print(
        f"Val PPL: baseline={base_ppl:.2f} -> step0={step0_ppl:.2f} -> best={best_ppl:.2f} "
        f"(elapsed={elapsed:.1f}s, {tok_per_sec:,.0f} tok/s, mean_lora_rel_norm={mean_lora_norm:.3f})"
    )
    print(f"Final sample: {final_gen}")
    print("=" * 90)

    if args.save_path:
        save_p = Path(args.save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_name": args.model,
                "args": vars(args),
                "state_dict": model.state_dict(),
            },
            save_p,
        )
        print(f"Saved checkpoint to {save_p}")


if __name__ == "__main__":
    main()
