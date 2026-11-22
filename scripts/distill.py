from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from rrt import (
    SharedWeightLoRALinear,
    build_layer_groups,
    compute_adaptive_layer_ranks,
    convert_hf_model_to_relaxed,
    hf_parameter_summary,
)
from rrt.distill import (
    CPUWeightEMA,
    CrossArchHiddenProjector,
    StochasticBF16Adafactor,
    chunked_kl_ce_backward,
    compute_layerwise_hidden_loss,
    compute_matched_layer_pairs,
    compute_projected_hidden_loss,
    quantize_model_int8_weight_only,
)


def _download_local_shards(dataset_mix: str = "elite") -> dict[str, tuple[list[str], str]]:
    """Download local Parquet shards once to HF cache for constant-memory PyArrow streaming.

    - ``basic``: FineWeb-Edu + SmolTalk
    - ``elite``: FineWeb-Edu (45%) + SmolTalk (25%) + Cosmopedia-v2 (15%)
                 + FineMath-4plus (10%) + Wikipedia EN (5%)
    """
    sources: dict[str, tuple[list[str], str]] = {
        "fw": (
            [
                hf_hub_download(
                    repo_id="HuggingFaceFW/fineweb-edu",
                    repo_type="dataset",
                    filename="sample/10BT/000_00000.parquet",
                )
            ],
            "text",
        ),
        "chat": (
            [
                hf_hub_download(
                    repo_id="HuggingFaceTB/smoltalk",
                    repo_type="dataset",
                    filename=f"data/all/train-0000{i}-of-00009.parquet",
                )
                for i in (0, 1)
            ],
            "messages",
        ),
    }
    if dataset_mix == "elite":
        sources["cosmo"] = (
            [
                hf_hub_download(
                    repo_id="HuggingFaceTB/smollm-corpus",
                    repo_type="dataset",
                    filename="cosmopedia-v2/train-00000-of-00104.parquet",
                )
            ],
            "text",
        )
        sources["math"] = (
            [
                hf_hub_download(
                    repo_id="HuggingFaceTB/finemath",
                    repo_type="dataset",
                    filename="finemath-4plus/train-00000-of-00064.parquet",
                )
            ],
            "text",
        )
        sources["wiki"] = (
            [
                hf_hub_download(
                    repo_id="wikimedia/wikipedia",
                    repo_type="dataset",
                    filename="20231101.en/train-00000-of-00041.parquet",
                )
            ],
            "text",
        )
    return sources


def _parquet_row_stream(paths: list[str], column: str, seed: int) -> Iterator[object]:
    """Constant-memory infinite iterator over local Parquet shards with row-group shuffling."""
    rng = random.Random(seed)
    while True:
        order = list(paths)
        rng.shuffle(order)
        for path in order:
            pf = pq.ParquetFile(path)
            rg_indices = list(range(pf.num_row_groups))
            rng.shuffle(rg_indices)
            for rg in rg_indices:
                table = pf.read_row_group(rg, columns=[column])
                values = table.column(0).to_pylist()
                del table
                rng.shuffle(values)
                for val in values:
                    yield val
                del values


def _format_smoltalk_messages(messages: list[dict], tokenizer: AutoTokenizer) -> str:
    if not messages:
        return ""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )


class PackedStreamLoader:
    """Packs weighted multi-source local Parquet shards into contiguous (B, seq_len) batches."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        seq_len: int,
        sources: dict[str, tuple[list[str], str]],
        weights: dict[str, float],
        seed: int = 42,
    ) -> None:
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        self.source_names = [k for k in weights if k in sources and weights[k] > 0.0]
        self.source_weights = [weights[k] for k in self.source_names]
        self.iters = {
            name: _parquet_row_stream(sources[name][0], column=sources[name][1], seed=seed + idx * 1000)
            for idx, name in enumerate(self.source_names)
        }
        self.cols = {name: sources[name][1] for name in self.source_names}
        self.bufs: dict[str, list[int]] = {name: [] for name in self.source_names}
        self.rng = random.Random(seed)

    def _next_chunk_from(self, source: str) -> list[int]:
        buf = self.bufs[source]
        stream = self.iters[source]
        is_chat = self.cols[source] == "messages"
        while len(buf) < self.seq_len:
            raw = next(stream)
            text = _format_smoltalk_messages(raw, self.tokenizer) if is_chat else raw
            if not text or not isinstance(text, str) or not text.strip():
                continue
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            if ids:
                buf.extend(ids)
                buf.append(self.eos_id)
        chunk = buf[: self.seq_len]
        del buf[: self.seq_len]
        return chunk

    def next_batch(self, batch_size: int, chat_prob_override: float | None = None) -> torch.Tensor:
        weights = list(self.source_weights)
        if chat_prob_override is not None and "chat" in self.source_names and len(self.source_names) > 1:
            chat_idx = self.source_names.index("chat")
            other_total = sum(w for i, w in enumerate(weights) if i != chat_idx)
            if other_total > 0:
                rem = max(0.0, 1.0 - chat_prob_override)
                weights = [
                    chat_prob_override if i == chat_idx else rem * (w / other_total)
                    for i, w in enumerate(weights)
                ]
        chosen = self.rng.choices(self.source_names, weights=weights, k=batch_size)
        rows = [self._next_chunk_from(src) for src in chosen]
        return torch.tensor(rows, dtype=torch.long)


def build_fixed_validation_blocks(tokenizer: AutoTokenizer, seq_len: int, num_blocks: int = 24) -> torch.Tensor:
    """Build a deterministic validation set mixing WikiText-2 and SmolTalk test split."""
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    half = num_blocks // 2

    # 1. General English from WikiText-2 validation
    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    wt_ids: list[int] = []
    for row in wt:
        txt = row["text"].strip()
        if txt:
            wt_ids.extend(tokenizer(txt, add_special_tokens=False)["input_ids"])
            wt_ids.append(eos_id)
        if len(wt_ids) >= half * seq_len:
            break
    wt_blocks = [wt_ids[i * seq_len : (i + 1) * seq_len] for i in range(half)]

    # 2. Chat conversations from SmolTalk test shard (local cached parquet)
    test_parquet = hf_hub_download(
        repo_id="HuggingFaceTB/smoltalk",
        repo_type="dataset",
        filename="data/all/test-00000-of-00001.parquet",
    )
    pf = pq.ParquetFile(test_parquet)
    msgs_col = pf.read_row_group(0, columns=["messages"]).column(0).to_pylist()
    chat_ids: list[int] = []
    for msgs in msgs_col:
        text = _format_smoltalk_messages(msgs, tokenizer)
        if text:
            chat_ids.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
            chat_ids.append(eos_id)
        if len(chat_ids) >= (num_blocks - half) * seq_len:
            break
    chat_blocks = [chat_ids[i * seq_len : (i + 1) * seq_len] for i in range(num_blocks - half)]

    return torch.tensor(wt_blocks + chat_blocks, dtype=torch.long)


@torch.no_grad()
def evaluate_student(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    val_blocks: torch.Tensor,
    batch_size: int,
    device: torch.device,
    chunk_tokens: int = 256,
) -> tuple[float, float, float]:
    """Evaluate validation CE, KL(teacher || student), and perplexity using chunked logits."""
    was_training = student.training
    student.eval()
    teacher.eval()
    total_ce = 0.0
    total_kl = 0.0
    total_toks = 0

    for start in range(0, len(val_blocks), batch_size):
        batch = val_blocks[start : start + batch_size].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            hs = student.model(input_ids=batch, use_cache=False).last_hidden_state[:, :-1, :]
            ht = teacher.model(input_ids=batch, use_cache=False).last_hidden_state[:, :-1, :]

        hs_2d = hs.reshape(-1, hs.size(-1))
        ht_2d = ht.reshape(-1, ht.size(-1))
        tgts = batch[:, 1:].reshape(-1)
        n_tok = tgts.numel()

        s_dtype = student.lm_head.weight.dtype
        t_dtype = teacher.lm_head.weight.dtype
        for c0 in range(0, n_tok, chunk_tokens):
            c1 = min(c0 + chunk_tokens, n_tok)
            s_log = student.lm_head(hs_2d[c0:c1].to(s_dtype)).float()
            t_log = teacher.lm_head(ht_2d[c0:c1].to(t_dtype)).float()
            total_ce += F.cross_entropy(s_log, tgts[c0:c1], reduction="sum").item()
            total_kl += F.kl_div(
                F.log_softmax(s_log, dim=-1),
                F.softmax(t_log, dim=-1),
                reduction="sum",
            ).item()
        total_toks += n_tok

    mean_ce = total_ce / max(total_toks, 1)
    mean_kl = total_kl / max(total_toks, 1)
    ppl = math.exp(min(mean_ce, 20.0))
    if was_training:
        student.train()
    return mean_ce, mean_kl, ppl


@torch.no_grad()
def chat_sample(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    user_prompt: str,
    device: torch.device,
    max_new_tokens: int = 80,
) -> str:
    was_training = model.training
    model.eval()
    messages = [{"role": "user", "content": user_prompt}]
    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.08,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    reply = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
    if was_training:
        model.train()
    return reply.replace("\n", " ")


class _PromptStream:
    """Streams real user prompts from cached SmolTalk Parquet shards for On-Policy Distillation."""

    def __init__(self, chat_paths: list[str], tokenizer: AutoTokenizer, seed: int = 1234) -> None:
        self.stream = _parquet_row_stream(chat_paths, column="messages", seed=seed)
        self.tokenizer = tokenizer

    def next_prompts(self, batch_size: int) -> list[str]:
        prompts: list[str] = []
        while len(prompts) < batch_size:
            msgs = next(self.stream)
            if not isinstance(msgs, list):
                continue
            user_text = ""
            for m in msgs:
                if isinstance(m, dict) and m.get("role") == "user":
                    content = str(m.get("content", "")).strip()
                    if 15 <= len(content) <= 320:
                        user_text = content
                    break
            if not user_text:
                continue
            try:
                formatted = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_text}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                formatted = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_text}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            prompts.append(formatted)
        return prompts


def run_on_policy_distill_step(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    device: torch.device,
    max_new_tokens: int = 96,
    temperature: float = 1.5,
    jsd_beta: float = 0.1,
    chunk_tokens: int = 128,
    kl_weight: float = 1.0,
) -> tuple[float, int]:
    """One On-Policy Distillation (GKD / Qwen3 OPD) rollout + Reverse-skewed JSD update.

    The student generates its own replies with KV cache, then the frozen teacher grades every
    generated assistant token via JSD_beta(P_T || P_S) with CE=0 (never reinforcing student bugs).
    Returns ``(opd_jsd_loss, num_active_generated_tokens)``.
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id
    orig_padding_side = getattr(tokenizer, "padding_side", "right")
    try:
        tokenizer.padding_side = "left"
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=160)
    finally:
        tokenizer.padding_side = orig_padding_side

    prompt_ids = enc["input_ids"].to(device)
    prompt_attn = enc["attention_mask"].to(device)
    prompt_len = prompt_ids.size(1)

    student.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        gen = student.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_attn,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            use_cache=True,
            pad_token_id=pad_id,
        )

    if gen.size(1) <= prompt_len:
        student.train()
        return 0.0, 0

    bsz, total_len = gen.shape
    full_attn = torch.zeros((bsz, total_len), dtype=torch.long, device=device)
    full_attn[:, :prompt_len] = prompt_attn
    response_mask = torch.zeros((bsz, total_len), dtype=torch.bool, device=device)
    for b in range(bsz):
        gen_row = gen[b, prompt_len:]
        if eos_id is not None:
            eos_pos = (gen_row == eos_id).nonzero(as_tuple=False)
            valid_gen = int(eos_pos[0].item()) + 1 if eos_pos.numel() > 0 else gen_row.numel()
        else:
            valid_gen = gen_row.numel()
        if valid_gen > 0:
            full_attn[b, prompt_len : prompt_len + valid_gen] = 1
            response_mask[b, prompt_len : prompt_len + valid_gen] = True

    active_tokens = int(response_mask[:, 1:].sum().item())
    if active_tokens == 0:
        student.train()
        return 0.0, 0

    position_ids = full_attn.cumsum(dim=-1) - 1
    position_ids.masked_fill_(full_attn == 0, 0)

    student.train()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        t_hidden = teacher.model(
            input_ids=gen, attention_mask=full_attn, position_ids=position_ids, use_cache=False
        ).last_hidden_state
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        s_hidden = student.model(
            input_ids=gen, attention_mask=full_attn, position_ids=position_ids, use_cache=False
        ).last_hidden_state

    _, opd_jsd = chunked_kl_ce_backward(
        student_hidden=s_hidden,
        teacher_hidden=t_hidden,
        student_lm_head=student.lm_head,
        teacher_lm_head=teacher.lm_head,
        input_ids=gen,
        kl_weight=kl_weight,
        ce_weight=0.0,
        temperature=temperature,
        chunk_tokens=chunk_tokens,
        grad_accum=1,
        kl_type="jsd",
        jsd_beta=jsd_beta,
        token_mask=response_mask,
    )
    return opd_jsd, active_tokens


def save_checkpoint(
    path: Path,
    student: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    tokens_seen: int,
    elapsed_sec: float,
    best_val_kl: float,
    args: argparse.Namespace,
    projector: CrossArchHiddenProjector | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Save deduplicated trainable weights + metadata in bf16 for compact, fast checkpoints
    seen_ids: set[int] = set()
    trainable_state: dict[str, torch.Tensor] = {}
    for name, param in student.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen_ids:
            continue
        seen_ids.add(id(param))
        trainable_state[name] = param.detach().to(dtype=torch.bfloat16, device="cpu")

    payload: dict[str, object] = {
        "model_name": args.model,
        "sharing_strategy": args.sharing_strategy,
        "num_recursions": args.num_recursions,
        "keep_first": args.keep_first,
        "keep_last": args.keep_last,
        "lora_rank": args.lora_rank,
        "adaptive_rank": args.adaptive_rank,
        "use_dora_scale": args.dora_scale,
        "base_init": args.base_init,
        "step": step,
        "tokens_seen": tokens_seen,
        "elapsed_sec": elapsed_sec,
        "best_val_kl": best_val_kl,
        "trainable_state": trainable_state,
        "optimizer_state": optimizer.state_dict(),
    }
    if projector is not None:
        payload["projector_state"] = {k: v.detach().cpu() for k, v in projector.state_dict().items()}

    tmp_path = path.with_suffix(".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def load_trainable_state(student: torch.nn.Module, ckpt_path: Path) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    trainable_state: dict[str, torch.Tensor] = ckpt["trainable_state"]
    param_map = dict(student.named_parameters())
    loaded = 0
    for name, tensor in trainable_state.items():
        if name in param_map:
            if param_map[name].shape != tensor.shape:
                raise ValueError(
                    f"Shape mismatch loading {name} from {ckpt_path}: "
                    f"model={tuple(param_map[name].shape)} vs ckpt={tuple(tensor.shape)}"
                )
            param_map[name].data.copy_(tensor.to(device=param_map[name].device, dtype=param_map[name].dtype))
            loaded += 1
    print(f"Loaded {loaded} deduplicated parameter tensors from {ckpt_path} (step={ckpt['step']}, tokens={ckpt['tokens_seen']:,})")
    return ckpt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SoTA Online Distillation of Whitened-SVD Relaxed Recursive Qwen3 (Multi-Domain + Strong-to-Weak + OPD)"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B", help="Student base model to compress")
    parser.add_argument(
        "--teacher-model",
        type=str,
        default="",
        help="Optional stronger teacher model with matching vocab (e.g. Qwen/Qwen3-4B); defaults to --model",
    )
    parser.add_argument(
        "--teacher-int8",
        action="store_true",
        help="Quantize frozen Teacher decoder layers to int8 weight-only (~4.4 GB VRAM for Qwen3-4B)",
    )
    parser.add_argument("--num-recursions", type=int, default=2)
    parser.add_argument(
        "--sharing-strategy",
        choices=["cycle", "middle_cycle", "sequence", "middle_sequence"],
        default="middle_sequence",
    )
    parser.add_argument("--keep-first", type=int, default=3, help="Unshared entry layers to protect super-weights")
    parser.add_argument("--keep-last", type=int, default=1, help="Unshared exit layers")
    parser.add_argument("--lora-rank", type=int, default=128)
    parser.add_argument(
        "--adaptive-rank",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Budget-preserving U-shaped LoRA rank allocation (higher rank at edge tied layers, lower in middle)",
    )
    parser.add_argument(
        "--dora-scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="DoRA-style per-output-channel learnable magnitude scaling on every shared projection",
    )
    parser.add_argument("--base-init", choices=["mean", "first", "middle", "last"], default="mean")
    parser.add_argument("--no-whitened-svd", action="store_true", help="Disable activation-whitened SVD calibration")
    parser.add_argument("--calib-blocks", type=int, default=16, help="Calibration blocks for Whitened SVD")
    parser.add_argument("--optimizer", choices=["bf16-adafactor", "adafactor", "adamw"], default="bf16-adafactor")
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--lora-lr-mult", type=float, default=2.0, help="LoRA+ learning rate multiplier")
    parser.add_argument("--beta1", type=float, default=0.9, help="Momentum beta1 for bf16-adafactor")
    parser.add_argument("--lr-schedule", choices=["cosine", "wsd"], default="wsd", help="LR schedule (cosine or Warmup-Stable-Decay)")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--kl-weight", type=float, default=0.85)
    parser.add_argument("--ce-weight", type=float, default=0.15)
    parser.add_argument("--kl-temp", type=float, default=1.5)
    parser.add_argument("--kl-type", choices=["forward", "jsd"], default="forward", help="forward KL (recommended for RRT recovery) or Generalized JSD")
    parser.add_argument("--jsd-beta", type=float, default=0.5, help="Beta for JSD_beta (0.5=symmetric GKD, 0.1=skew-reverse DistiLLM)")
    parser.add_argument("--hidden-weight", type=float, default=2.0, help="Layerwise hidden-state alignment weight (same-arch teacher)")
    parser.add_argument(
        "--cross-arch-hidden",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="DSKD training-only linear projectors (student_dim -> teacher_dim) for cross-architecture hidden alignment",
    )
    parser.add_argument("--cross-arch-taps", type=int, default=6, help="Number of matched interior depth taps for cross-arch projector")
    parser.add_argument("--cross-arch-hidden-weight", type=float, default=0.25, help="Loss weight for cross-arch hidden alignment")
    parser.add_argument(
        "--on-policy-every",
        type=int,
        default=0,
        help="Accumulate an On-Policy Distillation rollout gradient into the main step every N steps (0=off)",
    )
    parser.add_argument("--on-policy-batch", type=int, default=2, help="Number of prompts per On-Policy rollout")
    parser.add_argument("--on-policy-tokens", type=int, default=64, help="Max generated tokens per On-Policy rollout")
    parser.add_argument("--on-policy-jsd-beta", type=float, default=0.5, help="JSD beta for On-Policy rollouts (0.5=symmetric)")
    parser.add_argument("--on-policy-weight", type=float, default=0.15, help="Gradient weight for accumulated OPD rollout inside main step")
    parser.add_argument("--ema-start", type=float, default=0.70, help="Progress fraction to start zero-VRAM CPU Weight EMA (>=1.0 disables)")
    parser.add_argument("--ema-decay", type=float, default=0.95, help="CPU EMA decay (0.0=uniform SWA, 0.95=exponential moving average)")
    parser.add_argument("--ema-every", type=int, default=5, help="Update CPU EMA every N steps once active")
    parser.add_argument("--chunk-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-hours", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=0, help="If > 0, stop after this many optimizer steps")
    parser.add_argument(
        "--dataset-mix",
        choices=["elite", "basic"],
        default="elite",
        help="elite=45%% FineWeb-Edu + 25%% SmolTalk + 15%% Cosmopedia-v2 + 10%% FineMath-4plus + 5%% Wikipedia; basic=FineWeb-Edu+SmolTalk",
    )
    parser.add_argument("--chat-prob", type=float, default=0.30, help="Chat ratio for --dataset-mix basic")
    parser.add_argument("--phase2-frac", type=float, default=0.0, help="Optional final chat-heavy phase fraction")
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-every-mins", type=float, default=20.0)
    parser.add_argument("--out-dir", type=str, default="checkpoints/qwen3_1.7b_rrt_v2")
    parser.add_argument(
        "--init-from",
        type=str,
        default="",
        help="Initialize Student weights from a Stage-1 checkpoint (inherits architecture metadata, skips SVD)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if not torch.cuda.is_available():
        raise RuntimeError("scripts/distill.py requires a CUDA GPU")
    device = torch.device("cuda")
    print(
        f"GPU: {torch.cuda.get_device_name(0)} | PyTorch {torch.__version__} | "
        f"Flash-SDP={torch.backends.cuda.flash_sdp_enabled()} | cuDNN-SDP={torch.backends.cuda.cudnn_sdp_enabled()}",
        flush=True,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = out_dir / "latest.pt"
    best_ckpt = out_dir / "best.pt"
    ema_ckpt = out_dir / "ema.pt"
    resume_ckpt = latest_ckpt if latest_ckpt.exists() else (best_ckpt if best_ckpt.exists() else None)
    is_resuming = bool(args.resume and resume_ckpt is not None)
    init_from_ckpt = Path(args.init_from) if (args.init_from and not is_resuming) else None
    skip_svd = is_resuming or (init_from_ckpt is not None)
    teacher_name = args.teacher_model.strip() or args.model

    # Auto-inherit structural RRT metadata from checkpoint when resuming or initializing from Stage 1
    arch_source_ckpt = resume_ckpt if is_resuming else init_from_ckpt
    if arch_source_ckpt is not None and arch_source_ckpt.exists():
        meta_ckpt = torch.load(arch_source_ckpt, map_location="cpu", weights_only=False)
        args.sharing_strategy = meta_ckpt.get("sharing_strategy", args.sharing_strategy)
        args.num_recursions = int(meta_ckpt.get("num_recursions", args.num_recursions))
        if meta_ckpt.get("keep_first") is not None:
            args.keep_first = int(meta_ckpt["keep_first"])
        if meta_ckpt.get("keep_last") is not None:
            args.keep_last = int(meta_ckpt["keep_last"])
        args.lora_rank = int(meta_ckpt.get("lora_rank", args.lora_rank))
        args.adaptive_rank = bool(meta_ckpt.get("adaptive_rank", False))
        args.dora_scale = bool(meta_ckpt.get("use_dora_scale", False))
        del meta_ckpt
        print(
            f"Inherited RRT architecture from {arch_source_ckpt}: "
            f"strategy={args.sharing_strategy}, R={args.num_recursions}, keep_first={args.keep_first}, "
            f"keep_last={args.keep_last}, r={args.lora_rank}, adaptive_rank={args.adaptive_rank}, dora_scale={args.dora_scale}",
            flush=True,
        )

    print(f"\nLoading tokenizer ({args.model})...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    mix_weights = (
        {"fw": 0.45, "chat": 0.25, "cosmo": 0.15, "math": 0.10, "wiki": 0.05}
        if args.dataset_mix == "elite"
        else {"fw": max(0.0, 1.0 - args.chat_prob), "chat": args.chat_prob}
    )
    print(f"Preparing local Parquet shards (dataset_mix={args.dataset_mix}: {mix_weights})...", flush=True)
    data_sources = _download_local_shards(args.dataset_mix)

    calib_batches: list[torch.Tensor] | None = None
    if not skip_svd and not args.no_whitened_svd and args.calib_blocks > 0:
        calib_loader = PackedStreamLoader(
            tokenizer, seq_len=args.seq_len, sources=data_sources, weights=mix_weights, seed=args.seed + 9999
        )
        calib_batches = [
            calib_loader.next_batch(2) for _ in range(max(1, args.calib_blocks // 2))
        ]
        del calib_loader

    t_mode = "int8 weight-only" if args.teacher_int8 else "bf16"
    print(f"Loading frozen Teacher ({teacher_name}, {t_mode}, SDPA FlashAttention)...", flush=True)
    if args.teacher_int8:
        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_name, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        quantize_model_int8_weight_only(teacher, device=device)
        gc.collect()
    else:
        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_name, dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    print(f"  Teacher loaded -> GPU VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)

    svd_tag = (
        "Whitened SVD-LLM"
        if calib_batches is not None
        else ("skip-init (resume/init-from)" if skip_svd else "Unweighted SVD")
    )
    print(
        f"Loading Student ({args.model}) and converting ({args.sharing_strategy}, R={args.num_recursions}, "
        f"keep_first={args.keep_first}, keep_last={args.keep_last}, r={args.lora_rank}, {svd_tag})...",
        flush=True,
    )
    student = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    if teacher.config.vocab_size != student.config.vocab_size:
        raise ValueError(
            f"Teacher vocab ({teacher.config.vocab_size}) != Student vocab ({student.config.vocab_size})"
        )
    orig_summary = hf_parameter_summary(student)
    orig_layers = orig_summary["total"] - orig_summary["embeddings"]

    t_cvt = time.perf_counter()
    report = convert_hf_model_to_relaxed(
        student,
        num_recursions=args.num_recursions,
        sharing_strategy=args.sharing_strategy,
        lora_rank=args.lora_rank,
        base_init=args.base_init,
        use_svd=(not skip_svd),
        keep_first=args.keep_first,
        keep_last=args.keep_last,
        calibration_batches=calib_batches,
        adaptive_rank=args.adaptive_rank,
        use_dora_scale=args.dora_scale,
    )
    del calib_batches
    torch.cuda.synchronize()
    cvt_summary = hf_parameter_summary(student)
    cvt_layers = cvt_summary["total"] - cvt_summary["embeddings"]
    print(
        f"Conversion done in {time.perf_counter() - t_cvt:.1f}s | "
        f"layers: {orig_layers:,} -> {cvt_layers:,} ({100.0 * cvt_layers / orig_layers:.1f}%) | "
        f"total: {orig_summary['total']:,} -> {cvt_summary['total']:,} ({100.0 * cvt_summary['total'] / orig_summary['total']:.1f}%) | "
        f"W-err: {report.mean_rel_error_strict:.3f} -> {report.mean_rel_error_relaxed:.3f} (E={100.0 * report.mean_energy_captured:.1f}%)",
        flush=True,
    )

    # Compute intermediate hidden-state alignment tap points (same-arch 1-to-1 or cross-arch DSKD projector)
    same_embed = teacher_name == args.model
    same_arch = (
        teacher.config.hidden_size == student.config.hidden_size
        and teacher.config.num_hidden_layers == student.config.num_hidden_layers
    )
    num_layers = len(student.model.layers)
    groups = build_layer_groups(
        num_layers, args.num_recursions, args.sharing_strategy, keep_first=args.keep_first, keep_last=args.keep_last
    )
    if args.adaptive_rank:
        adp_ranks = list(compute_adaptive_layer_ranks(groups, args.lora_rank).values())
        print(f"  Adaptive U-shaped LoRA ranks ({len(adp_ranks)} tied groups): {adp_ranks} | DoRA out_scale={args.dora_scale}", flush=True)
    hidden_tap_indices = sorted({g[-1] + 1 for g in groups if len(g) > 1} | {num_layers})

    if same_embed:
        # Share identical frozen embed_tokens and lm_head between Teacher and Student (-622 MB VRAM)
        student.model.embed_tokens = teacher.model.embed_tokens
        student.lm_head = teacher.lm_head
    for name, param in student.named_parameters():
        if "embed_tokens" in name or "lm_head" in name:
            param.requires_grad = False

    if init_from_ckpt is not None:
        load_trainable_state(student, init_from_ckpt)

    use_same_arch_hidden = args.hidden_weight > 0.0 and same_arch
    use_cross_arch_hidden = (
        not same_arch and args.cross_arch_hidden and args.cross_arch_hidden_weight > 0.0 and args.cross_arch_taps > 0
    )
    use_bf16_weights = args.optimizer == "bf16-adafactor"
    hidden_projector: CrossArchHiddenProjector | None = None
    matched_layer_pairs: list[tuple[int, int]] = []
    if use_cross_arch_hidden:
        matched_layer_pairs = compute_matched_layer_pairs(
            student_layers=student.config.num_hidden_layers,
            teacher_layers=teacher.config.num_hidden_layers,
            num_taps=args.cross_arch_taps,
            include_final=False,
        )
        hidden_projector = CrossArchHiddenProjector(
            student_dim=student.config.hidden_size,
            teacher_dim=teacher.config.hidden_size,
            num_taps=len(matched_layer_pairs),
            device=device,
            dtype=torch.bfloat16 if use_bf16_weights else torch.float32,
        )
        proj_mb = sum(p.numel() * p.element_size() for p in hidden_projector.parameters()) / 1e6
        if not is_resuming:
            proj_loader = PackedStreamLoader(
                tokenizer, seq_len=args.seq_len, sources=data_sources, weights=mix_weights, seed=args.seed + 7777
            )
            proj_calib = [proj_loader.next_batch(2) for _ in range(4)]
            del proj_loader
            p_before, p_after = hidden_projector.fit_ridge_(
                student=student,
                teacher=teacher,
                layer_pairs=matched_layer_pairs,
                calib_batches=proj_calib,
                device=device,
                ridge=0.05,
            )
            del proj_calib
            torch.cuda.empty_cache()
            print(
                f"  [DSKD Projector] {student.config.hidden_size}->{teacher.config.hidden_size} across "
                f"{len(matched_layer_pairs)} matched layer pairs {matched_layer_pairs} ({proj_mb:.1f} MB training-only) | "
                f"ridge-init hid: {p_before:.4f} -> {p_after:.4f}",
                flush=True,
            )
        else:
            print(
                f"  [DSKD Projector] {student.config.hidden_size}->{teacher.config.hidden_size} across "
                f"{len(matched_layer_pairs)} matched layer pairs {matched_layer_pairs} ({proj_mb:.1f} MB training-only)",
                flush=True,
            )

    base_params: list[torch.nn.Parameter] = []
    lora_params: list[torch.nn.Parameter] = []
    seen_ids: set[int] = set()
    for name, param in student.named_parameters():
        if id(param) in seen_ids:
            continue
        seen_ids.add(id(param))
        if param.requires_grad:
            param.data = param.data.to(torch.bfloat16 if use_bf16_weights else torch.float32)
            if "lora_A" in name or "lora_B" in name or "out_scale" in name:
                lora_params.append(param)
            else:
                base_params.append(param)
    if hidden_projector is not None:
        lora_params.extend(hidden_projector.parameters())
    trainable_params = base_params + lora_params

    torch.cuda.empty_cache()

    if hasattr(student, "activation_checkpointing_enable"):
        try:
            student.activation_checkpointing_enable(activation_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            student.activation_checkpointing_enable()
    else:
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(student, "enable_input_require_grads"):
        student.enable_input_require_grads()

    param_groups = [
        {"params": base_params, "lr": args.lr, "lr_mult": 1.0},
        {"params": lora_params, "lr": args.lr * args.lora_lr_mult, "lr_mult": args.lora_lr_mult},
    ]
    if args.optimizer == "bf16-adafactor":
        optimizer: torch.optim.Optimizer = StochasticBF16Adafactor(
            param_groups, lr=args.lr, beta1=args.beta1, beta2_decay=-0.8, weight_decay=0.01
        )
    elif args.optimizer == "adafactor":
        optimizer = torch.optim.Adafactor(
            param_groups, lr=args.lr, beta2_decay=-0.8, eps=(None, 1e-3), weight_decay=0.01
        )
    else:
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    start_step = 0
    tokens_seen = 0
    prior_elapsed = 0.0
    best_val_kl = float("inf")

    if is_resuming and resume_ckpt is not None:
        ckpt = load_trainable_state(student, resume_ckpt)
        if hidden_projector is not None and "projector_state" in ckpt:
            hidden_projector.load_state_dict(ckpt["projector_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_step = int(ckpt["step"])
        tokens_seen = int(ckpt["tokens_seen"])
        prior_elapsed = float(ckpt["elapsed_sec"])
        best_val_kl = float(ckpt.get("best_val_kl", float("inf")))

    print("Building fixed validation split (12 WikiText-2 + 12 SmolTalk test blocks)...", flush=True)
    val_blocks = build_fixed_validation_blocks(tokenizer, seq_len=args.seq_len, num_blocks=24)
    wt_val_blocks = val_blocks[:12]
    chat_val_blocks = val_blocks[12:]

    # Evaluate Teacher baseline and Student step-0 separately on WikiText-2 & SmolTalk
    t_wt_ce, _, t_wt_ppl = evaluate_student(teacher, teacher, wt_val_blocks, args.batch_size, device, args.chunk_tokens)
    t_ch_ce, _, t_ch_ppl = evaluate_student(teacher, teacher, chat_val_blocks, args.batch_size, device, args.chunk_tokens)
    s_wt_ce, s_wt_kl, s_wt_ppl = evaluate_student(student, teacher, wt_val_blocks, args.batch_size, device, args.chunk_tokens)
    s_ch_ce, s_ch_kl, s_ch_ppl = evaluate_student(student, teacher, chat_val_blocks, args.batch_size, device, args.chunk_tokens)
    step0_mean_kl = 0.5 * (s_wt_kl + s_ch_kl)
    print(f"[Teacher {teacher_name} ({t_mode})] wt_ppl={t_wt_ppl:.2f} | chat_ppl={t_ch_ppl:.2f}", flush=True)
    print(
        f"[Student step {start_step}] wt_ppl={s_wt_ppl:.2f} (kl={s_wt_kl:.3f}) | "
        f"chat_ppl={s_ch_ppl:.2f} (kl={s_ch_kl:.3f}) | mean_kl={step0_mean_kl:.4f}",
        flush=True,
    )
    if init_from_ckpt is not None and not is_resuming:
        # Never overwrite best.pt with a checkpoint worse than the Stage-1 starting point
        best_val_kl = step0_mean_kl
        save_checkpoint(
            best_ckpt,
            student,
            optimizer,
            start_step,
            tokens_seen,
            prior_elapsed,
            best_val_kl,
            args,
            projector=hidden_projector,
        )

    sample_prompts = [
        "Explain how HTTPS encryption works to a 12-year-old in two short sentences.",
        "Who discovered penicillin, in what year, and from what organism?",
    ]
    teacher_samples: list[dict[str, str]] = []
    if start_step == 0:
        t_q1 = chat_sample(teacher, tokenizer, sample_prompts[0], device, max_new_tokens=55)
        t_q2 = chat_sample(teacher, tokenizer, sample_prompts[1], device, max_new_tokens=55)
        print(f"  [Teacher Q1] {t_q1}", flush=True)
        print(f"  [Teacher Q2] {t_q2}", flush=True)
        teacher_samples = [{"tag": "Q1 HTTPS", "text": t_q1}, {"tag": "Q2 Penicillin", "text": t_q2}]

    metrics_path = out_dir / "metrics.jsonl"
    if not is_resuming and metrics_path.exists():
        metrics_path.unlink()

    def log_metric(record: dict) -> None:
        with metrics_path.open("a", encoding="utf-8") as mf:
            mf.write(json.dumps(record, ensure_ascii=False) + "\n")

    if not is_resuming:
        log_metric(
            {
                "type": "meta",
                "model": f"{args.model} <- {teacher_name}",
                "teacher_model": teacher_name,
                "teacher_int8": args.teacher_int8,
                "dataset_mix": args.dataset_mix,
                "kl_type": args.kl_type,
                "jsd_beta": args.jsd_beta,
                "lr_schedule": args.lr_schedule,
                "sharing_strategy": args.sharing_strategy,
                "num_recursions": args.num_recursions,
                "keep_first": args.keep_first,
                "keep_last": args.keep_last,
                "lora_rank": args.lora_rank,
                "max_hours": args.max_hours,
                "max_steps": args.max_steps,
                "params_str": f"{cvt_summary['total']/1e9:.3f}B ({100.0 * cvt_layers / orig_layers:.1f}% layers)",
                "teacher_wt_ppl": round(t_wt_ppl, 4),
                "teacher_chat_ppl": round(t_ch_ppl, 4),
                "teacher_samples": teacher_samples,
            }
        )
        init_loras = [m.relative_lora_norm() for m in student.modules() if isinstance(m, SharedWeightLoRALinear)]
        log_metric(
            {
                "type": "eval",
                "step": start_step,
                "tokens_m": round(tokens_seen / 1e6, 4),
                "elapsed_min": round(prior_elapsed / 60.0, 2),
                "wt_ppl": round(s_wt_ppl, 4),
                "wt_kl": round(s_wt_kl, 4),
                "chat_ppl": round(s_ch_ppl, 4),
                "chat_kl": round(s_ch_kl, 4),
                "mean_kl": round(0.5 * (s_wt_kl + s_ch_kl), 4),
                "lora_rel": round(sum(init_loras) / len(init_loras), 4) if init_loras else 0.0,
            }
        )

    data_seed = args.seed + start_step + (10000 if init_from_ckpt is not None else 0)
    loader = PackedStreamLoader(
        tokenizer, seq_len=args.seq_len, sources=data_sources, weights=mix_weights, seed=data_seed
    )
    prompt_stream: _PromptStream | None = (
        _PromptStream(data_sources["chat"][0], tokenizer, seed=data_seed + 5555)
        if args.on_policy_every > 0 and "chat" in data_sources
        else None
    )
    ema: CPUWeightEMA | None = CPUWeightEMA(decay=args.ema_decay) if args.ema_start < 1.0 else None
    gc.collect()

    max_seconds = args.max_hours * 3600.0
    eff_toks_per_step = args.batch_size * args.grad_accum * (args.seq_len - 1)
    use_hidden_distill = use_same_arch_hidden or use_cross_arch_hidden
    hid_desc = (
        f"SameArch({args.hidden_weight}x{len(hidden_tap_indices)})"
        if use_same_arch_hidden
        else (
            f"DSKDProj({args.cross_arch_hidden_weight}x{len(matched_layer_pairs)})"
            if use_cross_arch_hidden
            else "OFF"
        )
    )
    kl_desc = f"JSD(b={args.jsd_beta})" if args.kl_type == "jsd" else "FwdKL"
    opd_desc = (
        f"every {args.on_policy_every} (b={args.on_policy_batch}, tok={args.on_policy_tokens}, jsd={args.on_policy_jsd_beta}, w={args.on_policy_weight})"
        if prompt_stream is not None
        else "OFF"
    )
    ema_desc = f"start@{int(args.ema_start*100)}% (decay={args.ema_decay}, every={args.ema_every})" if ema is not None else "OFF"
    print(
        f"\nStarting distillation: budget={args.max_hours:.2f}h (max_steps={args.max_steps or 'time-based'}), "
        f"eff_batch={args.batch_size}x{args.grad_accum}x{args.seq_len} ({eff_toks_per_step:,} tok/step), "
        f"opt={args.optimizer}(beta1={args.beta1}), lr={args.lr:.1e}(lora x{args.lora_lr_mult}, {args.lr_schedule}), "
        f"{kl_desc}={args.kl_weight}(T={args.kl_temp})/CE={args.ce_weight}/Hid={hid_desc}, "
        f"OPD={opd_desc}, CPU-EMA={ema_desc}, mix={args.dataset_mix} | dashboard -> {metrics_path}",
        flush=True,
    )

    session_t0 = time.perf_counter()
    last_save_t = session_t0
    last_log_t = session_t0
    last_log_tokens = tokens_seen
    win_ce = 0.0
    win_kl = 0.0
    win_hid = 0.0
    win_n = 0
    step = start_step

    student.train()
    try:
        while True:
            total_elapsed = prior_elapsed + (time.perf_counter() - session_t0)
            if args.max_steps > 0 and step >= args.max_steps:
                break
            if args.max_steps == 0 and total_elapsed >= max_seconds:
                break

            if args.max_steps > 0:
                progress = min(1.0, step / max(1, args.max_steps))
            else:
                progress = min(1.0, total_elapsed / max(1.0, max_seconds))

            # Warmup first 3% of run, then Cosine or WSD (82% stable -> 15% decay)
            if progress < 0.03:
                lr_scale = max(0.05, progress / 0.03)
            elif args.lr_schedule == "wsd":
                if progress < 0.85:
                    lr_scale = 1.0
                else:
                    dec_prog = (progress - 0.85) / 0.15
                    lr_scale = args.min_lr_ratio + (1.0 - args.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * dec_prog))
            else:
                cos_prog = (progress - 0.03) / 0.97
                lr_scale = args.min_lr_ratio + (1.0 - args.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * cos_prog))
            cur_lr = args.lr * lr_scale
            for group in optimizer.param_groups:
                group["lr"] = cur_lr * group.get("lr_mult", 1.0)

            in_phase2 = args.phase2_frac > 0.0 and progress >= (1.0 - args.phase2_frac)
            chat_override = 0.70 if in_phase2 else None

            optimizer.zero_grad(set_to_none=True)
            step_ce = 0.0
            step_kl = 0.0
            step_hid = 0.0

            for _ in range(args.grad_accum):
                batch = loader.next_batch(args.batch_size, chat_prob_override=chat_override).to(device, non_blocking=True)
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    t_out = teacher.model(
                        input_ids=batch, use_cache=False, output_hidden_states=use_hidden_distill
                    )
                    t_hidden = t_out.last_hidden_state
                    t_hiddens = t_out.hidden_states if use_hidden_distill else None

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    s_out = student.model(
                        input_ids=batch, use_cache=False, output_hidden_states=use_hidden_distill
                    )
                    s_hidden = s_out.last_hidden_state

                aux_loss = None
                if use_same_arch_hidden and t_hiddens is not None and s_out.hidden_states is not None:
                    hid_loss = compute_layerwise_hidden_loss(
                        s_out.hidden_states, t_hiddens, hidden_tap_indices, norm_weight=0.2
                    )
                    step_hid += hid_loss.item() / args.grad_accum
                    aux_loss = (args.hidden_weight * hid_loss) / args.grad_accum
                elif (
                    use_cross_arch_hidden
                    and hidden_projector is not None
                    and t_hiddens is not None
                    and s_out.hidden_states is not None
                ):
                    hid_loss = compute_projected_hidden_loss(
                        s_out.hidden_states, t_hiddens, matched_layer_pairs, hidden_projector, norm_weight=0.2
                    )
                    step_hid += hid_loss.item() / args.grad_accum
                    aux_loss = (args.cross_arch_hidden_weight * hid_loss) / args.grad_accum

                # Release all unused intermediate teacher/student hidden tuples before peak-VRAM logit chunks
                del t_out, t_hiddens, s_out

                ce_val, kl_val = chunked_kl_ce_backward(
                    student_hidden=s_hidden,
                    teacher_hidden=t_hidden,
                    student_lm_head=student.lm_head,
                    teacher_lm_head=teacher.lm_head,
                    input_ids=batch,
                    kl_weight=args.kl_weight,
                    ce_weight=args.ce_weight,
                    temperature=args.kl_temp,
                    chunk_tokens=args.chunk_tokens,
                    grad_accum=args.grad_accum,
                    aux_loss=aux_loss,
                    kl_type=args.kl_type,
                    jsd_beta=args.jsd_beta,
                )
                del s_hidden, t_hidden, aux_loss, batch
                step_ce += ce_val / args.grad_accum
                step_kl += kl_val / args.grad_accum

            # Optional On-Policy Distillation: accumulate weighted gradient into this same optimizer step
            # (never take an unaveraged full-LR optimizer.step() on a tiny 128-token rollout alone).
            opd_toks = 0
            opd_jsd = 0.0
            next_step = step + 1
            if prompt_stream is not None and args.on_policy_weight > 0.0 and next_step % args.on_policy_every == 0:
                opd_prompts = prompt_stream.next_prompts(args.on_policy_batch)
                opd_jsd, opd_toks = run_on_policy_distill_step(
                    student=student,
                    teacher=teacher,
                    tokenizer=tokenizer,
                    prompts=opd_prompts,
                    device=device,
                    max_new_tokens=args.on_policy_tokens,
                    temperature=args.kl_temp,
                    jsd_beta=args.on_policy_jsd_beta,
                    chunk_tokens=args.chunk_tokens,
                    kl_weight=args.on_policy_weight,
                )

            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            step = next_step
            tokens_seen += eff_toks_per_step + opd_toks
            win_ce += step_ce
            win_kl += step_kl
            win_hid += step_hid
            win_n += 1
            if opd_toks > 0 and (step % (args.on_policy_every * 2) == 0 or step == args.on_policy_every):
                print(f"  [OPD step {step:5d}] gen_toks={opd_toks:4d} | opd_jsd={opd_jsd:.4f}", flush=True)

            # Zero-VRAM CPU Weight EMA update during late training / WSD decay
            if ema is not None and progress >= args.ema_start and step % args.ema_every == 0:
                ema.update(student)

            now = time.perf_counter()
            if step % args.log_interval == 0 or step == start_step + 1:
                dt = max(now - last_log_t, 1e-6)
                tps = (tokens_seen - last_log_tokens) / dt
                peak_gb = torch.cuda.max_memory_allocated() / 1e9
                total_elapsed = prior_elapsed + (now - session_t0)
                eta_min = max(0.0, (max_seconds - total_elapsed) / 60.0) if args.max_steps == 0 else 0.0
                avg_ce = win_ce / max(1, win_n)
                avg_kl = win_kl / max(1, win_n)
                avg_hid = win_hid / max(1, win_n)
                print(
                    f"step {step:5d} | tok={tokens_seen/1e6:6.2f}M | "
                    f"ce={avg_ce:.3f} kl={avg_kl:.3f} hid={avg_hid:.4f} | lr={cur_lr:.2e} | "
                    f"{tps:6.0f} tok/s | vram={peak_gb:.2f}GB | elapsed={total_elapsed/60:.1f}m | eta={eta_min:.0f}m",
                    flush=True,
                )
                log_metric(
                    {
                        "type": "train",
                        "step": step,
                        "tokens_m": round(tokens_seen / 1e6, 4),
                        "elapsed_min": round(total_elapsed / 60.0, 2),
                        "ce": round(avg_ce, 4),
                        "kl": round(avg_kl, 4),
                        "hid": round(avg_hid, 5),
                        "lr": cur_lr,
                        "tps": round(tps, 1),
                        "vram_gb": round(peak_gb, 3),
                    }
                )
                win_ce = 0.0
                win_kl = 0.0
                win_hid = 0.0
                win_n = 0
                last_log_t = now
                last_log_tokens = tokens_seen

            if step % args.eval_interval == 0:
                s_wt_ce, s_wt_kl, s_wt_ppl = evaluate_student(
                    student, teacher, wt_val_blocks, args.batch_size, device, args.chunk_tokens
                )
                s_ch_ce, s_ch_kl, s_ch_ppl = evaluate_student(
                    student, teacher, chat_val_blocks, args.batch_size, device, args.chunk_tokens
                )
                mean_kl = 0.5 * (s_wt_kl + s_ch_kl)
                lora_norms = [m.relative_lora_norm() for m in student.modules() if isinstance(m, SharedWeightLoRALinear)]
                mean_lora = sum(lora_norms) / len(lora_norms) if lora_norms else 0.0
                q1 = chat_sample(student, tokenizer, sample_prompts[0], device, max_new_tokens=60)
                q2 = chat_sample(student, tokenizer, sample_prompts[1], device, max_new_tokens=60)
                print(
                    f"\n>>> [EVAL step {step}] wt_ppl={s_wt_ppl:.2f} (T={t_wt_ppl:.2f}, kl={s_wt_kl:.3f}) | "
                    f"chat_ppl={s_ch_ppl:.2f} (T={t_ch_ppl:.2f}, kl={s_ch_kl:.3f}) | "
                    f"mean_kl={mean_kl:.4f} | lora_rel={mean_lora:.3f}",
                    flush=True,
                )
                print(f"    [Q1] {q1}", flush=True)
                print(f"    [Q2] {q2}\n", flush=True)
                log_metric(
                    {
                        "type": "eval",
                        "step": step,
                        "tokens_m": round(tokens_seen / 1e6, 4),
                        "elapsed_min": round((prior_elapsed + (time.perf_counter() - session_t0)) / 60.0, 2),
                        "wt_ppl": round(s_wt_ppl, 4),
                        "wt_kl": round(s_wt_kl, 4),
                        "chat_ppl": round(s_ch_ppl, 4),
                        "chat_kl": round(s_ch_kl, 4),
                        "mean_kl": round(mean_kl, 4),
                        "lora_rel": round(mean_lora, 4),
                        "samples": [
                            {"tag": "Q1 HTTPS", "text": q1},
                            {"tag": "Q2 Penicillin", "text": q2},
                        ],
                    }
                )

                if mean_kl < best_val_kl:
                    best_val_kl = mean_kl
                    save_checkpoint(
                        best_ckpt,
                        student,
                        optimizer,
                        step,
                        tokens_seen,
                        prior_elapsed + (time.perf_counter() - session_t0),
                        best_val_kl,
                        args,
                        projector=hidden_projector,
                    )
                    print(f"    Saved new best checkpoint -> {best_ckpt} (mean_kl={best_val_kl:.4f})", flush=True)

            if (now - last_save_t) >= args.save_every_mins * 60.0:
                save_checkpoint(
                    latest_ckpt,
                    student,
                    optimizer,
                    step,
                    tokens_seen,
                    prior_elapsed + (now - session_t0),
                    best_val_kl,
                    args,
                    projector=hidden_projector,
                )
                print(f"  [checkpoint] saved {latest_ckpt} at step {step} ({tokens_seen/1e6:.2f}M tokens)", flush=True)
                last_save_t = now

    except KeyboardInterrupt:
        print("\nInterrupted by user; saving latest checkpoint before exit...", flush=True)

    # Final evaluation and checkpoint save (+ zero-VRAM CPU EMA Model Soup evaluation)
    total_elapsed = prior_elapsed + (time.perf_counter() - session_t0)
    s_wt_ce, s_wt_kl, s_wt_ppl = evaluate_student(
        student, teacher, wt_val_blocks, args.batch_size, device, args.chunk_tokens
    )
    s_ch_ce, s_ch_kl, s_ch_ppl = evaluate_student(
        student, teacher, chat_val_blocks, args.batch_size, device, args.chunk_tokens
    )
    mean_kl = 0.5 * (s_wt_kl + s_ch_kl)
    final_loras = [m.relative_lora_norm() for m in student.modules() if isinstance(m, SharedWeightLoRALinear)]
    log_metric(
        {
            "type": "eval",
            "step": step,
            "tokens_m": round(tokens_seen / 1e6, 4),
            "elapsed_min": round(total_elapsed / 60.0, 2),
            "wt_ppl": round(s_wt_ppl, 4),
            "wt_kl": round(s_wt_kl, 4),
            "chat_ppl": round(s_ch_ppl, 4),
            "chat_kl": round(s_ch_kl, 4),
            "mean_kl": round(mean_kl, 4),
            "lora_rel": round(sum(final_loras) / len(final_loras), 4) if final_loras else 0.0,
        }
    )
    save_checkpoint(
        latest_ckpt, student, optimizer, step, tokens_seen, total_elapsed, min(best_val_kl, mean_kl), args, projector=hidden_projector
    )
    if mean_kl < best_val_kl:
        best_val_kl = mean_kl
        save_checkpoint(best_ckpt, student, optimizer, step, tokens_seen, total_elapsed, best_val_kl, args, projector=hidden_projector)

    if ema is not None and ema.num_updates >= 2:
        backup = ema.swap_into(student)
        e_wt_ce, e_wt_kl, e_wt_ppl = evaluate_student(
            student, teacher, wt_val_blocks, args.batch_size, device, args.chunk_tokens
        )
        e_ch_ce, e_ch_kl, e_ch_ppl = evaluate_student(
            student, teacher, chat_val_blocks, args.batch_size, device, args.chunk_tokens
        )
        ema_mean_kl = 0.5 * (e_wt_kl + e_ch_kl)
        save_checkpoint(ema_ckpt, student, optimizer, step, tokens_seen, total_elapsed, ema_mean_kl, args, projector=hidden_projector)
        print(
            f"\n>>> [CPU-EMA Model Soup ({ema.num_updates} updates)] "
            f"wt_ppl={e_wt_ppl:.2f} (raw={s_wt_ppl:.2f}) | chat_ppl={e_ch_ppl:.2f} (raw={s_ch_ppl:.2f}) | "
            f"mean_kl={ema_mean_kl:.4f} (raw={mean_kl:.4f}) -> saved {ema_ckpt}",
            flush=True,
        )
        if ema_mean_kl < best_val_kl:
            best_val_kl = ema_mean_kl
            save_checkpoint(best_ckpt, student, optimizer, step, tokens_seen, total_elapsed, best_val_kl, args, projector=hidden_projector)
            print(f"    EMA beat best_val_kl! Promoted CPU-EMA to {best_ckpt} (mean_kl={best_val_kl:.4f})", flush=True)
        ema.restore(student, backup)

    print(
        f"\n=== Finished at step {step} ({tokens_seen/1e6:.2f}M tokens, {total_elapsed/3600:.2f}h) ===\n"
        f"WikiText-2 PPL={s_wt_ppl:.2f} (T={t_wt_ppl:.2f}, kl={s_wt_kl:.3f}) | "
        f"SmolTalk PPL={s_ch_ppl:.2f} (T={t_ch_ppl:.2f}, kl={s_ch_kl:.3f}) | Mean KL={mean_kl:.4f} | Best KL={best_val_kl:.4f}\n"
        f"Checkpoints: {latest_ckpt}, {best_ckpt}" + (f", {ema_ckpt}" if ema is not None and ema.num_updates >= 2 else ""),
        flush=True,
    )


if __name__ == "__main__":
    main()
