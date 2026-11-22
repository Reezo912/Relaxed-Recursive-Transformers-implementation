from __future__ import annotations

import argparse
import gc
import math
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from chat import generate_reply, load_compressed_student
from distill import _format_smoltalk_messages
from rrt import SharedWeightLoRALinear, hf_parameter_summary


@torch.no_grad()
def eval_perplexity_blocks(
    model: torch.nn.Module,
    blocks: torch.Tensor,
    device: torch.device,
    batch_size: int = 4,
    chunk_tokens: int = 256,
) -> tuple[float, float]:
    model.eval()
    total_nll = 0.0
    total_toks = 0
    for start in range(0, len(blocks), batch_size):
        batch = blocks[start : start + batch_size].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            h = model.model(input_ids=batch, use_cache=False).last_hidden_state[:, :-1, :]
        h_2d = h.reshape(-1, h.size(-1))
        tgts = batch[:, 1:].reshape(-1)
        head_dtype = model.lm_head.weight.dtype
        for c0 in range(0, tgts.numel(), chunk_tokens):
            c1 = min(c0 + chunk_tokens, tgts.numel())
            logits = model.lm_head(h_2d[c0:c1].to(head_dtype)).float()
            total_nll += F.cross_entropy(logits, tgts[c0:c1], reduction="sum").item()
        total_toks += tgts.numel()
    ce = total_nll / max(total_toks, 1)
    return ce, math.exp(min(ce, 20.0))


@torch.no_grad()
def score_continuation_avg_logprob(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    continuation: str,
    device: torch.device,
) -> float:
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    c_ids = tokenizer(continuation, add_special_tokens=False)["input_ids"]
    if not c_ids:
        return -1e9
    full = p_ids + c_ids
    if len(full) > 1024:
        full = full[-1024:]
    c_len = min(len(c_ids), len(full) - 1)
    x = torch.tensor([full], dtype=torch.long, device=device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        h = model.model(input_ids=x, use_cache=False).last_hidden_state
        # Only project the continuation positions through lm_head
        h_cont = h[:, -(c_len + 1) : -1, :]
        logits = model.lm_head(h_cont.to(model.lm_head.weight.dtype)).float()
    log_probs = F.log_softmax(logits, dim=-1)
    tgt = x[:, -c_len:]
    tok_lp = log_probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return tok_lp.mean().item()


def eval_arc_easy(model: torch.nn.Module, tokenizer: AutoTokenizer, device: torch.device, n: int = 200) -> float:
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    correct = 0
    total = min(n, len(ds))
    for i in range(total):
        row = ds[i]
        prompt = f"Question: {row['question']}\nAnswer:"
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        scores = [score_continuation_avg_logprob(model, tokenizer, prompt, " " + t, device) for t in texts]
        pred_label = labels[int(max(range(len(scores)), key=lambda k: scores[k]))]
        if pred_label == row["answerKey"]:
            correct += 1
    return 100.0 * correct / total


def eval_boolq(model: torch.nn.Module, tokenizer: AutoTokenizer, device: torch.device, n: int = 200) -> float:
    ds = load_dataset("google/boolq", split="validation")
    correct = 0
    total = min(n, len(ds))
    for i in range(total):
        row = ds[i]
        passage = row["passage"][:1200]
        prompt = f"Passage: {passage}\nQuestion: {row['question']}?\nAnswer:"
        s_yes = score_continuation_avg_logprob(model, tokenizer, prompt, " yes", device)
        s_no = score_continuation_avg_logprob(model, tokenizer, prompt, " no", device)
        pred = s_yes > s_no
        if pred == bool(row["answer"]):
            correct += 1
    return 100.0 * correct / total


def eval_hellaswag(model: torch.nn.Module, tokenizer: AutoTokenizer, device: torch.device, n: int = 200) -> float:
    ds = load_dataset("Rowan/hellaswag", split="validation")
    correct = 0
    total = min(n, len(ds))
    for i in range(total):
        row = ds[i]
        prompt = f"{row['activity_label']}: {row['ctx']}"
        endings = row["endings"]
        scores = [score_continuation_avg_logprob(model, tokenizer, prompt, " " + e, device) for e in endings]
        pred = int(max(range(len(scores)), key=lambda k: scores[k]))
        if pred == int(row["label"]):
            correct += 1
    return 100.0 * correct / total


def build_test_blocks(tokenizer: AutoTokenizer, seq_len: int = 512, num_blocks: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    wt_ids: list[int] = []
    for row in wt:
        t = row["text"].strip()
        if t:
            wt_ids.extend(tokenizer(t, add_special_tokens=False)["input_ids"])
            wt_ids.append(eos_id)
        if len(wt_ids) >= num_blocks * seq_len:
            break
    wt_blocks = torch.tensor([wt_ids[i * seq_len : (i + 1) * seq_len] for i in range(num_blocks)], dtype=torch.long)

    test_parquet = hf_hub_download(
        repo_id="HuggingFaceTB/smoltalk",
        repo_type="dataset",
        filename="data/all/test-00000-of-00001.parquet",
    )
    pf = pq.ParquetFile(test_parquet)
    msgs_col = pf.read_row_group(0, columns=["messages"]).column(0).to_pylist()
    chat_ids: list[int] = []
    # Skip first 30 rows used in training validation split
    for msgs in msgs_col[30:]:
        text = _format_smoltalk_messages(msgs, tokenizer)
        if text:
            chat_ids.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
            chat_ids.append(eos_id)
        if len(chat_ids) >= num_blocks * seq_len:
            break
    chat_blocks = torch.tensor([chat_ids[i * seq_len : (i + 1) * seq_len] for i in range(num_blocks)], dtype=torch.long)
    return wt_blocks, chat_blocks


def lora_specialization_by_proj(model: torch.nn.Module) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, SharedWeightLoRALinear):
            proj = name.split(".")[-1]
            buckets.setdefault(proj, []).append(mod.relative_lora_norm())
    return {k: sum(v) / len(v) for k, v in sorted(buckets.items())}


def set_lora_enabled(model: torch.nn.Module, enabled: bool) -> dict[int, float]:
    saved: dict[int, float] = {}
    for mod in model.modules():
        if isinstance(mod, SharedWeightLoRALinear):
            saved[id(mod)] = mod.scaling
            if not enabled:
                mod.scaling = 0.0
    return saved


def restore_lora_scaling(model: torch.nn.Module, saved: dict[int, float]) -> None:
    for mod in model.modules():
        if isinstance(mod, SharedWeightLoRALinear) and id(mod) in saved:
            mod.scaling = saved[id(mod)]


def eval_one_model(
    label: str,
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    wt_blocks: torch.Tensor,
    chat_blocks: torch.Tensor,
    prompts: list[tuple[str, str]],
    device: torch.device,
    n_bench: int,
) -> dict:
    torch.cuda.synchronize()
    vram_gb = torch.cuda.memory_allocated() / 1e9
    summary = hf_parameter_summary(model)

    wt_ce, wt_ppl = eval_perplexity_blocks(model, wt_blocks, device)
    ch_ce, ch_ppl = eval_perplexity_blocks(model, chat_blocks, device)
    arc = eval_arc_easy(model, tokenizer, device, n=n_bench)
    boolq = eval_boolq(model, tokenizer, device, n=n_bench)
    hs = eval_hellaswag(model, tokenizer, device, n=n_bench)

    # Warmup generation once
    generate_reply(model, tokenizer, [{"role": "user", "content": "Say hello."}], device, max_new_tokens=16, temperature=0.0)

    replies: list[tuple[str, str, float]] = []
    speeds: list[float] = []
    for tag, prompt in prompts:
        rep, tps, _ = generate_reply(
            model, tokenizer, [{"role": "user", "content": prompt}], device, max_new_tokens=140, temperature=0.0
        )
        replies.append((tag, rep, tps))
        speeds.append(tps)

    return {
        "label": label,
        "params_total": summary["total"],
        "params_layers": summary["total"] - summary["embeddings"],
        "vram_gb": vram_gb,
        "wt_ce": wt_ce,
        "wt_ppl": wt_ppl,
        "chat_ce": ch_ce,
        "chat_ppl": ch_ppl,
        "arc_easy": arc,
        "boolq": boolq,
        "hellaswag": hs,
        "avg_tps": sum(speeds) / len(speeds),
        "replies": replies,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Objective comparison: Original Qwen3-1.7B vs Relaxed Recursive RRT")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/qwen3_1.7b_rrt/best.pt")
    parser.add_argument("--n-bench", type=int, default=200, help="Questions per zero-shot benchmark")
    args = parser.parse_args()

    device = torch.device("cuda")
    ckpt_path = Path(args.checkpoint)

    prompts = [
        ("Instructions", "List 3 pros and 3 cons of electric cars in bullet points. Keep each bullet under 12 words."),
        ("Tech Explain", "Explain how HTTPS encryption works to a 12-year-old in one short paragraph."),
        ("Python Code", "Write a concise Python function is_palindrome(s) ignoring spaces, case, and punctuation, plus 2 assert tests."),
        ("Math/Logic", "A train leaves at 14:15 at 80 km/h. Another leaves at 15:15 on the same track at 100 km/h. At what time does the second catch the first? Show brief steps."),
        ("Factual QA", "Who discovered penicillin, in what year, and from what organism?"),
        ("Spanish", "Explica en dos frases claras qué es la fotosíntesis."),
    ]

    print(f"Loading RRT checkpoint: {ckpt_path}...", flush=True)
    student, tokenizer, ckpt = load_compressed_student(ckpt_path, device)
    wt_blocks, chat_blocks = build_test_blocks(tokenizer, seq_len=512, num_blocks=32)

    print(f"\n[1/3] Evaluating RRT Compressed ({ckpt['tokens_seen']/1e6:.2f}M tokens, step {ckpt['step']})...", flush=True)
    t0 = time.perf_counter()
    res_rrt = eval_one_model("RRT Compressed (~1.2B)", student, tokenizer, wt_blocks, chat_blocks, prompts, device, args.n_bench)
    proj_norms = lora_specialization_by_proj(student)
    print(f"  done in {time.perf_counter() - t0:.1f}s", flush=True)

    print("\n[2/3] Evaluating Strict Recursive Ablation (same checkpoint with LoRA OFF)...", flush=True)
    saved_scales = set_lora_enabled(student, False)
    ab_wt_ce, ab_wt_ppl = eval_perplexity_blocks(student, wt_blocks, device)
    ab_ch_ce, ab_ch_ppl = eval_perplexity_blocks(student, chat_blocks, device)
    ab_arc = eval_arc_easy(student, tokenizer, device, n=args.n_bench)
    ab_rep, _, _ = generate_reply(
        student, tokenizer, [{"role": "user", "content": prompts[1][1]}], device, max_new_tokens=80, temperature=0.0
    )
    restore_lora_scaling(student, saved_scales)

    # Free student before loading teacher for clean VRAM reading
    del student
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n[3/3] Evaluating Original {ckpt['model_name']} (1.72B)...", flush=True)
    t0 = time.perf_counter()
    teacher = AutoModelForCausalLM.from_pretrained(
        ckpt["model_name"], dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    teacher.eval()
    res_orig = eval_one_model(f"Original {ckpt['model_name']}", teacher, tokenizer, wt_blocks, chat_blocks, prompts, device, args.n_bench)
    print(f"  done in {time.perf_counter() - t0:.1f}s", flush=True)

    rrt_b = f"RRT {res_rrt['params_total']/1e9:.2f}B"
    orig_b = f"Original {res_orig['params_total']/1e9:.2f}B"
    layer_pct = 100.0 * res_rrt["params_layers"] / max(res_orig["params_layers"], 1)

    # Print quantitative table
    print("\n" + "=" * 96)
    print(f"QUANTITATIVE COMPARISON: {orig_b} vs RELAXED RECURSIVE ({rrt_b})")
    print("=" * 96)
    header = f"{'Metric':26s} | {orig_b:>16s} | {f'{rrt_b} (LoRA)':>16s} | {'RRT Strict (No LoRA)':>20s}"
    print(header)
    print("-" * 96)
    print(f"{'Total Params':26s} | {res_orig['params_total']:>16,d} | {res_rrt['params_total']:>16,d} | {res_rrt['params_total']:>20,d}")
    print(f"{'Transformer Layer Params':26s} | {res_orig['params_layers']:>16,d} | {res_rrt['params_layers']:>16,d} | {f'({layer_pct:.1f}% of orig)':>20s}")
    print(f"{'Inference Weight VRAM':26s} | {res_orig['vram_gb']:>13.2f} GB | {res_rrt['vram_gb']:>13.2f} GB | {'-':>20s}")
    print(f"{'Decode Speed (greedy)':26s} | {res_orig['avg_tps']:>10.1f} tok/s | {res_rrt['avg_tps']:>10.1f} tok/s | {'-':>20s}")
    print(f"{'WikiText-2 Test PPL':26s} | {res_orig['wt_ppl']:>16.2f} | {res_rrt['wt_ppl']:>16.2f} | {ab_wt_ppl:>20.2f}")
    print(f"{'SmolTalk Test PPL':26s} | {res_orig['chat_ppl']:>16.2f} | {res_rrt['chat_ppl']:>16.2f} | {ab_ch_ppl:>20.2f}")
    print(f"{f'ARC-Easy ({args.n_bench}q) Acc':26s} | {res_orig['arc_easy']:>15.1f}% | {res_rrt['arc_easy']:>15.1f}% | {ab_arc:>19.1f}%")
    print(f"{f'BoolQ ({args.n_bench}q) Acc':26s} | {res_orig['boolq']:>15.1f}% | {res_rrt['boolq']:>15.1f}% | {'-':>20s}")
    print(f"{f'HellaSwag ({args.n_bench}q) Acc':26s} | {res_orig['hellaswag']:>15.1f}% | {res_rrt['hellaswag']:>15.1f}% | {'-':>20s}")
    avg_orig = (res_orig["arc_easy"] + res_orig["boolq"] + res_orig["hellaswag"]) / 3.0
    avg_rrt = (res_rrt["arc_easy"] + res_rrt["boolq"] + res_rrt["hellaswag"]) / 3.0
    print("-" * 96)
    print(
        f"{'Zero-Shot 3-Bench Avg':26s} | {avg_orig:>15.1f}% | {avg_rrt:>15.1f}% | "
        f"{f'({100.0 * avg_rrt / max(avg_orig, 1e-6):.1f}% retained)':>20s}"
    )

    print("\nLoRA Specialization (mean ||BA|| / ||W|| by projection):")
    for k, v in proj_norms.items():
        print(f"  {k:12s}: {v:.3f}")

    print("\n" + "=" * 96)
    print("SIDE-BY-SIDE QUALITATIVE RESPONSES (temperature=0, deterministic)")
    print("=" * 96)
    for (tag, p_txt), (_, rep_o, tps_o), (_, rep_r, tps_r) in zip(prompts, res_orig["replies"], res_rrt["replies"]):
        print(f"\n### [{tag}] Prompt: {p_txt}")
        print(f"--- Original 1.72B ({tps_o:.1f} tok/s) ---\n{rep_o}")
        print(f"--- RRT 1.20B ({tps_r:.1f} tok/s) ---\n{rep_r}")

    print(f"\n### [LoRA OFF Ablation on 'Tech Explain']\n{ab_rep}")


if __name__ == "__main__":
    main()
