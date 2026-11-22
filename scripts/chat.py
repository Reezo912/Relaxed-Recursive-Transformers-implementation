from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rrt import convert_hf_model_to_relaxed, hf_parameter_summary


def load_compressed_student(ckpt_path: Path, device: torch.device) -> tuple[torch.nn.Module, AutoTokenizer, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_name: str = ckpt["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    convert_hf_model_to_relaxed(
        model,
        num_recursions=int(ckpt["num_recursions"]),
        sharing_strategy=ckpt["sharing_strategy"],
        lora_rank=int(ckpt["lora_rank"]),
        base_init=ckpt.get("base_init", "mean"),
        use_svd=False,  # Weights will be overwritten immediately from checkpoint
        keep_first=ckpt.get("keep_first"),
        keep_last=ckpt.get("keep_last"),
        adaptive_rank=bool(ckpt.get("adaptive_rank", False)),
        use_dora_scale=bool(ckpt.get("use_dora_scale", False)),
    )

    param_map = dict(model.named_parameters())
    for name, tensor in ckpt["trainable_state"].items():
        if name in param_map:
            param_map[name].data.copy_(tensor.to(device=device, dtype=torch.bfloat16))

    model.eval()
    return model, tokenizer, ckpt


@torch.no_grad()
def generate_reply(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    device: torch.device,
    max_new_tokens: int = 256,
    temperature: float = 0.6,
    top_p: float = 0.9,
    thinking: bool = False,
) -> tuple[str, float, int]:
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    gen_kwargs: dict[str, object] = dict(
        max_new_tokens=max_new_tokens,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    if temperature > 0.0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kwargs.update(do_sample=False)

    with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        out = model.generate(**inputs, **gen_kwargs)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    new_toks = out.shape[1] - prompt_len
    reply = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
    return reply, new_toks / max(elapsed, 1e-6), new_toks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive chat demo for your Relaxed Recursive Qwen3 checkpoint")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/qwen3_1.7b_rrt/best.pt")
    parser.add_argument("--compare", action="store_true", help="Also load original Qwen3 for side-by-side comparison")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--thinking", action="store_true", help="Enable Qwen3 thinking mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading Relaxed Recursive checkpoint: {ckpt_path}...")
    student, tokenizer, ckpt = load_compressed_student(ckpt_path, device)
    s_summary = hf_parameter_summary(student)
    print(
        f"Loaded {ckpt['model_name']} -> {ckpt['sharing_strategy']} (R={ckpt['num_recursions']}, r={ckpt['lora_rank']}) | "
        f"trained {ckpt['tokens_seen']/1e6:.2f}M tokens (step {ckpt['step']}) | "
        f"params={s_summary['total']:,} (layers={s_summary['total']-s_summary['embeddings']:,})"
    )

    teacher = None
    if args.compare:
        print(f"Loading original {ckpt['model_name']} for side-by-side comparison...")
        teacher = AutoModelForCausalLM.from_pretrained(
            ckpt["model_name"], dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to(device)
        teacher.eval()

    print("\nType your message (or 'reset' to clear history, 'quit' to exit).\n")
    history: list[dict[str, str]] = []

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not user_text:
            continue
        if user_text.lower() in ("quit", "exit"):
            break
        if user_text.lower() == "reset":
            history.clear()
            print("[Conversation reset]\n")
            continue

        history.append({"role": "user", "content": user_text})
        reply_s, tps_s, n_s = generate_reply(
            student,
            tokenizer,
            history,
            device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            thinking=args.thinking,
        )
        print(f"\n[RRT ~1.2B | {tps_s:.1f} tok/s | {n_s} tok]\n{reply_s}\n")

        if teacher is not None:
            reply_t, tps_t, n_t = generate_reply(
                teacher,
                tokenizer,
                history,
                device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                thinking=args.thinking,
            )
            print(f"[Original {ckpt['model_name']} | {tps_t:.1f} tok/s | {n_t} tok]\n{reply_t}\n")

        history.append({"role": "assistant", "content": reply_s})


if __name__ == "__main__":
    main()
