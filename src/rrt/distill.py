from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def fp32_to_bf16_stochastic(x: Tensor) -> Tensor:
    """Unbiased stochastic rounding from fp32 to bf16.

    Because IEEE-754 float32 is sign-magnitude and bfloat16 is its exact upper 16 bits,
    adding a uniform random 16-bit integer to the raw int32 bit pattern carries into
    bit 16 with probability equal to the dropped fractional mantissa, making
    E[fp32_to_bf16_stochastic(x)] == x exactly (preventing bf16 weight-update stalling).
    """
    x_fp32 = x.float().contiguous()
    bits = x_fp32.view(torch.int32)
    rand_bits = torch.randint(0, 1 << 16, bits.shape, device=bits.device, dtype=torch.int32)
    # Only round finite values so Inf/NaN bit patterns are never corrupted
    finite = torch.isfinite(x_fp32)
    rounded_bits = torch.where(finite, (bits + rand_bits) & ~0xFFFF, bits)
    return rounded_bits.view(torch.float32).to(torch.bfloat16)


class StochasticBF16Adafactor(torch.optim.Optimizer):
    """Memory-efficient Adafactor for bf16 parameters with optional bf16 momentum + stochastic rounding.

    Keeps weights, gradients, and optional first-moment momentum in bf16 while computing
    factored RMS second moments and parameter updates in fp32 one tensor at a time,
    writing back to bf16 via exact unbiased stochastic rounding.
    """

    def __init__(
        self,
        params,
        lr: float = 2e-4,
        beta1: float = 0.0,
        beta2_decay: float = -0.8,
        eps1: float = 1e-30,
        eps2: float = 1e-3,
        clip_threshold: float = 1.0,
        weight_decay: float = 0.01,
    ) -> None:
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"beta1 must be in [0, 1), got {beta1}")
        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2_decay=beta2_decay,
            eps1=eps1,
            eps2=eps2,
            clip_threshold=clip_threshold,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2_decay = group["beta2_decay"]
            eps1 = group["eps1"]
            clip_threshold = group["clip_threshold"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.float()
                param_fp32 = p.float()
                state = self.state[p]
                factored = grad.ndim >= 2

                if len(state) == 0:
                    state["step"] = 0
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(grad.shape[:-1], dtype=torch.float32, device=p.device)
                        state["exp_avg_sq_col"] = torch.zeros(
                            grad.shape[:-2] + grad.shape[-1:], dtype=torch.float32, device=p.device
                        )
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad, dtype=torch.float32)
                    if beta1 > 0.0:
                        state["exp_avg"] = torch.zeros_like(
                            p, dtype=torch.bfloat16 if p.dtype == torch.bfloat16 else torch.float32
                        )

                state["step"] += 1
                step = state["step"]
                beta2 = 1.0 - math.pow(step, beta2_decay)

                update = grad.mul(grad).add_(eps1)
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]
                    exp_avg_sq_row.mul_(beta2).add_(update.mean(dim=-1), alpha=1.0 - beta2)
                    exp_avg_sq_col.mul_(beta2).add_(update.mean(dim=-2), alpha=1.0 - beta2)

                    r_factor = (
                        (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True).clamp_(min=eps1))
                        .rsqrt_()
                        .unsqueeze(-1)
                    )
                    c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
                    update = grad * r_factor * c_factor
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2).add_(update, alpha=1.0 - beta2)
                    update = grad * exp_avg_sq.rsqrt()

                rms = update.pow(2).mean().sqrt().clamp_(min=1e-12)
                update.div_((rms / clip_threshold).clamp_(min=1.0))

                if beta1 > 0.0:
                    exp_avg = state["exp_avg"]
                    m_fp32 = exp_avg.float().mul_(beta1).add_(update, alpha=1.0 - beta1)
                    if exp_avg.dtype == torch.bfloat16:
                        exp_avg.copy_(fp32_to_bf16_stochastic(m_fp32))
                    else:
                        exp_avg.copy_(m_fp32)
                    update = m_fp32 / (1.0 - math.pow(beta1, step))

                if wd != 0.0 and p.ndim >= 2:
                    param_fp32.mul_(1.0 - lr * wd)
                param_fp32.add_(update, alpha=-lr)

                if p.dtype == torch.bfloat16:
                    p.copy_(fp32_to_bf16_stochastic(param_fp32))
                else:
                    p.copy_(param_fp32.to(p.dtype))

        return loss


def compute_layerwise_hidden_loss(
    student_hiddens: tuple[Tensor, ...],
    teacher_hiddens: tuple[Tensor, ...],
    layer_indices: list[int],
    norm_weight: float = 0.2,
) -> Tensor:
    """Scale-invariant direction (cosine) + relative magnitude hidden-state alignment loss.

    Supervises intermediate tied-block exits directly against the frozen teacher's hidden
    states so gradients do not have to travel solely from the final vocabulary head.
    """
    losses: list[Tensor] = []
    for idx in layer_indices:
        hs = student_hiddens[idx].float()
        ht = teacher_hiddens[idx].detach().float()
        cos_loss = (1.0 - F.cosine_similarity(hs, ht, dim=-1, eps=1e-6)).mean()
        if norm_weight > 0.0:
            ns = hs.norm(dim=-1)
            nt = ht.norm(dim=-1).clamp(min=1e-6)
            rel_norm_loss = ((ns - nt) / nt).pow(2).mean()
            losses.append(cos_loss + norm_weight * rel_norm_loss)
        else:
            losses.append(cos_loss)
    return torch.stack(losses).mean()


def compute_matched_layer_pairs(
    student_layers: int,
    teacher_layers: int,
    num_taps: int = 6,
    include_final: bool = True,
) -> list[tuple[int, int]]:
    """Map evenly spaced relative depths between student and teacher hidden_states tuples.

    Note: HF ``output_hidden_states=True`` has index 0 = embeddings and index L = final layer output.
    When ``include_final=False``, taps are spread strictly across interior layers ``1 .. L-1``
    so cross-architecture hidden alignment does not pull the student's final hidden state away
    from its own frozen ``lm_head``.
    """
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    s_max = student_layers if include_final else max(1, student_layers - 1)
    t_max = teacher_layers if include_final else max(1, teacher_layers - 1)
    denom = num_taps if include_final else (num_taps + 1)
    for k in range(1, num_taps + 1):
        s_idx = max(1, min(s_max, round(student_layers * k / denom)))
        t_idx = max(1, min(t_max, round(teacher_layers * k / denom)))
        pair = (s_idx, t_idx)
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    return pairs


class CrossArchHiddenProjector(nn.Module):
    """Training-only linear projectors (student_dim -> teacher_dim) for cross-architecture distillation.

    Discarded after training so inference adds 0 extra parameters (DSKD-style alignment,
    e.g. aligning Qwen3-1.7B/1.24B 2048-d states to Qwen3-4B 2560-d states).
    """

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        num_taps: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.projs = nn.ModuleList(
            [
                nn.Linear(student_dim, teacher_dim, bias=False, device=device, dtype=dtype)
                for _ in range(num_taps)
            ]
        )
        std = 1.0 / math.sqrt(student_dim)
        with torch.no_grad():
            for proj in self.projs:
                nn.init.normal_(proj.weight, mean=0.0, std=std)

    @torch.no_grad()
    def fit_ridge_(
        self,
        student: nn.Module,
        teacher: nn.Module,
        layer_pairs: list[tuple[int, int]],
        calib_batches: list[Tensor],
        device: torch.device | str,
        ridge: float = 0.05,
    ) -> tuple[float, float]:
        """Initialize each linear projector via closed-form ridge regression on calibration batches.

        Eliminates early random-projection gradient shock when starting Stage-2 Strong-to-Weak
        distillation from an already-healed Stage-1 student.
        Returns ``(loss_before, loss_after)`` measured on the last calibration batch.
        """
        if not calib_batches or len(layer_pairs) != len(self.projs):
            return 0.0, 0.0

        dev = torch.device(device)
        s_was_train = student.training
        t_was_train = teacher.training
        student.eval()
        teacher.eval()

        s_dim = self.projs[0].in_features
        t_dim = self.projs[0].out_features
        grams = [torch.zeros((s_dim, s_dim), dtype=torch.float32, device=dev) for _ in layer_pairs]
        cross = [torch.zeros((s_dim, t_dim), dtype=torch.float32, device=dev) for _ in layer_pairs]

        last_s_hiddens: tuple[Tensor, ...] | None = None
        last_t_hiddens: tuple[Tensor, ...] | None = None
        use_amp = dev.type == "cuda"

        for batch in calib_batches:
            b = batch.to(dev, non_blocking=True)
            with torch.amp.autocast(dev.type, dtype=torch.bfloat16, enabled=use_amp):
                s_h = student.model(input_ids=b, use_cache=False, output_hidden_states=True).hidden_states
                t_h = teacher.model(input_ids=b, use_cache=False, output_hidden_states=True).hidden_states
            for k, (s_idx, t_idx) in enumerate(layer_pairs):
                hs = s_h[s_idx].reshape(-1, s_dim).float()
                ht = t_h[t_idx].reshape(-1, t_dim).float()
                grams[k].addmm_(hs.T, hs)
                cross[k].addmm_(hs.T, ht)
            last_s_hiddens = s_h
            last_t_hiddens = t_h

        assert last_s_hiddens is not None and last_t_hiddens is not None
        loss_before = compute_projected_hidden_loss(last_s_hiddens, last_t_hiddens, layer_pairs, self, norm_weight=0.2).item()

        eye = torch.eye(s_dim, dtype=torch.float32, device=dev)
        for k, proj in enumerate(self.projs):
            diag_mean = grams[k].diagonal().mean().clamp(min=1e-6)
            reg_g = grams[k] + (ridge * diag_mean) * eye
            w_t = torch.linalg.solve(reg_g, cross[k])
            proj.weight.copy_(w_t.T.to(device=proj.weight.device, dtype=proj.weight.dtype))

        loss_after = compute_projected_hidden_loss(last_s_hiddens, last_t_hiddens, layer_pairs, self, norm_weight=0.2).item()
        del grams, cross, eye, last_s_hiddens, last_t_hiddens
        if s_was_train:
            student.train()
        if t_was_train:
            teacher.train()
        return loss_before, loss_after


def compute_projected_hidden_loss(
    student_hiddens: tuple[Tensor, ...],
    teacher_hiddens: tuple[Tensor, ...],
    layer_pairs: list[tuple[int, int]],
    projector: CrossArchHiddenProjector,
    norm_weight: float = 0.2,
) -> Tensor:
    """Cosine + relative norm hidden-state alignment across mismatched widths via linear projectors."""
    losses: list[Tensor] = []
    for k, (s_idx, t_idx) in enumerate(layer_pairs):
        proj = projector.projs[k]
        hs = proj(student_hiddens[s_idx].to(dtype=proj.weight.dtype)).float()
        ht = teacher_hiddens[t_idx].detach().float()
        cos_loss = (1.0 - F.cosine_similarity(hs, ht, dim=-1, eps=1e-6)).mean()
        if norm_weight > 0.0:
            ns = hs.norm(dim=-1)
            nt = ht.norm(dim=-1).clamp(min=1e-6)
            rel_norm_loss = ((ns - nt) / nt).pow(2).mean()
            losses.append(cos_loss + norm_weight * rel_norm_loss)
        else:
            losses.append(cos_loss)
    return torch.stack(losses).mean()


class CPUWeightEMA:
    """Zero-VRAM CPU Exponential Moving Average / Model Soup over deduplicated trainable parameters.

    Stores fp32 shadow copies on CPU so GPU memory stays untouched. Supports both exponential
    decay (``decay > 0``, e.g. 0.95) and uniform Stochastic Weight Averaging (``decay == 0``).
    """

    def __init__(self, decay: float = 0.95) -> None:
        if not (0.0 <= decay < 1.0):
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        self.decay = decay
        self.num_updates = 0
        self.shadow: dict[str, Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        seen: set[int] = set()
        self.num_updates += 1
        alpha = (1.0 / self.num_updates) if self.decay == 0.0 else (1.0 - self.decay)
        for name, param in model.named_parameters():
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            cur = param.detach().to(device="cpu", dtype=torch.float32)
            if name not in self.shadow:
                self.shadow[name] = cur.clone()
            else:
                self.shadow[name].lerp_(cur, alpha)

    @torch.no_grad()
    def swap_into(self, model: nn.Module) -> dict[str, Tensor]:
        """Copy EMA weights into ``model`` and return a CPU backup of the previous weights."""
        backup: dict[str, Tensor] = {}
        seen: set[int] = set()
        for name, param in model.named_parameters():
            if not param.requires_grad or id(param) in seen or name not in self.shadow:
                continue
            seen.add(id(param))
            backup[name] = param.detach().cpu().clone()
            param.data.copy_(self.shadow[name].to(device=param.device, dtype=param.dtype))
        return backup

    @torch.no_grad()
    def restore(self, model: nn.Module, backup: dict[str, Tensor]) -> None:
        seen: set[int] = set()
        for name, param in model.named_parameters():
            if not param.requires_grad or id(param) in seen or name not in backup:
                continue
            seen.add(id(param))
            param.data.copy_(backup[name].to(device=param.device, dtype=param.dtype))


def chunked_kl_ce_backward(
    student_hidden: Tensor,
    teacher_hidden: Tensor,
    student_lm_head: nn.Linear,
    teacher_lm_head: nn.Linear,
    input_ids: Tensor,
    kl_weight: float = 0.7,
    ce_weight: float = 0.3,
    temperature: float = 1.0,
    chunk_tokens: int = 256,
    grad_accum: int = 1,
    aux_loss: Tensor | None = None,
    kl_type: str = "forward",
    jsd_beta: float = 0.5,
    token_mask: Tensor | None = None,
) -> tuple[float, float]:
    """Compute exact (kl_weight * Div(teacher, student) + ce_weight * CE) and run backward
    chunk-by-chunk over tokens so 151k-vocab fp32 logits never blow up VRAM.

    Supports:
    - ``kl_type="forward"``: standard forward KL ``KL(P_T || P_S)`` (mode-covering).
    - ``kl_type="jsd"``: Generalized Jensen-Shannon Divergence ``JSD_beta(P_T || P_S)``
      (GKD / DistiLLM): ``beta * KL(P_T || M) + (1 - beta) * KL(P_S || M)`` where
      ``M = beta * P_T + (1 - beta) * P_S``. Bounded gradients and sharper mode-seeking.
    - ``token_mask``: optional ``(B, T)`` or ``(B, T-1)`` boolean mask selecting active target
      positions (e.g. generated assistant tokens only during On-Policy Distillation).

    If ``aux_loss`` (e.g. scaled layerwise hidden-state alignment loss, already divided by
    ``grad_accum``) is supplied, runs a single combined ``torch.autograd.backward`` pass
    so intermediate activations are freed immediately without ``retain_graph=True``.

    Returns:
        (mean_ce_loss, mean_kl_loss) as Python floats.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if kl_type not in ("forward", "jsd"):
        raise ValueError(f"kl_type must be 'forward' or 'jsd', got {kl_type!r}")
    if kl_type == "jsd" and not (0.0 < jsd_beta < 1.0):
        raise ValueError(f"jsd_beta must be in (0, 1), got {jsd_beta}")

    h_s = student_hidden[:, :-1, :].reshape(-1, student_hidden.size(-1))
    h_t = teacher_hidden[:, :-1, :].detach().reshape(-1, teacher_hidden.size(-1))
    targets = input_ids[:, 1:].reshape(-1)
    n_tokens = targets.numel()
    if n_tokens == 0:
        return 0.0, 0.0

    flat_mask: Tensor | None = None
    active_tokens = n_tokens
    if token_mask is not None:
        if token_mask.size(1) == input_ids.size(1):
            flat_mask = token_mask[:, 1:].reshape(-1).bool()
        elif token_mask.size(1) == input_ids.size(1) - 1:
            flat_mask = token_mask.reshape(-1).bool()
        else:
            raise ValueError(
                f"token_mask seq_len ({token_mask.size(1)}) must match input_ids ({input_ids.size(1)}) or T-1"
            )
        active_tokens = int(flat_mask.sum().item())
        if active_tokens == 0:
            if aux_loss is not None:
                aux_loss.backward()
            return 0.0, 0.0

    # Leaf copy accumulates exact dL/dh_s across chunks before a single backward through student.model
    h_s_leaf = h_s.detach().requires_grad_(True)

    s_head_dtype = student_lm_head.weight.dtype
    t_head_dtype = teacher_lm_head.weight.dtype
    scale_denom = float(active_tokens * grad_accum)
    t_sq = temperature * temperature
    log_beta = math.log(jsd_beta) if kl_type == "jsd" else 0.0
    log_1m_beta = math.log(1.0 - jsd_beta) if kl_type == "jsd" else 0.0
    # Canonical KL-curvature normalization: near P_S = P_T, JSD_beta ~= beta*(1-beta)*KL,
    # so dividing by beta*(1-beta) preserves consistent gradient scale across any beta (0.5, 0.25, 0.1).
    jsd_norm = (1.0 / (jsd_beta * (1.0 - jsd_beta))) if kl_type == "jsd" else 1.0

    total_ce = 0.0
    total_kl = 0.0

    for start in range(0, n_tokens, chunk_tokens):
        end = min(start + chunk_tokens, n_tokens)
        hs_c = h_s_leaf[start:end]
        ht_c = h_t[start:end]
        tgt_c = targets[start:end]
        if flat_mask is not None:
            m_c = flat_mask[start:end]
            if not bool(m_c.any().item()):
                continue
            hs_c = hs_c[m_c]
            ht_c = ht_c[m_c]
            tgt_c = tgt_c[m_c]

        s_logits = student_lm_head(hs_c.to(s_head_dtype)).float()
        with torch.no_grad():
            t_logits = teacher_lm_head(ht_c.to(t_head_dtype)).float()

        ce_sum = F.cross_entropy(s_logits, tgt_c, reduction="sum")
        log_p_s = F.log_softmax(s_logits / temperature, dim=-1)
        if kl_type == "forward":
            p_t = F.softmax(t_logits / temperature, dim=-1)
            kl_sum = F.kl_div(log_p_s, p_t, reduction="sum") * t_sq
        else:
            log_p_t = F.log_softmax(t_logits / temperature, dim=-1)
            log_m = torch.logaddexp(log_p_t + log_beta, log_p_s + log_1m_beta)
            p_t = log_p_t.exp()
            p_s = log_p_s.exp()
            kl_t_m = (p_t * (log_p_t - log_m)).sum()
            kl_s_m = (p_s * (log_p_s - log_m)).sum()
            kl_sum = (jsd_beta * kl_t_m + (1.0 - jsd_beta) * kl_s_m) * (t_sq * jsd_norm)

        if ce_weight > 0.0:
            chunk_loss = (kl_weight * kl_sum + ce_weight * ce_sum) / scale_denom
        else:
            chunk_loss = (kl_weight * kl_sum) / scale_denom
        chunk_loss.backward()

        total_ce += ce_sum.item()
        total_kl += kl_sum.item()

    assert h_s_leaf.grad is not None
    if aux_loss is not None:
        torch.autograd.backward([h_s, aux_loss], [h_s_leaf.grad, torch.ones_like(aux_loss)])
    else:
        h_s.backward(h_s_leaf.grad)

    return total_ce / active_tokens, total_kl / active_tokens


class Int8WeightOnlyLinear(nn.Module):
    """Frozen per-output-channel symmetric int8 weight-only Linear for low-VRAM Teacher inference.

    Stores weights as int8 (1 byte/param -> 50% VRAM vs bf16) plus a 1D bf16 output scale,
    dequantizing on the fly during forward without requiring bitsandbytes or external kernels.
    """

    def __init__(self, linear: nn.Linear, device: torch.device | str | None = None) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        target_dev = device if device is not None else linear.weight.device

        with torch.no_grad():
            w = linear.weight.detach().float()
            scale = w.abs().amax(dim=-1).clamp(min=1e-8) / 127.0
            w_q = torch.clamp(torch.round(w / scale.unsqueeze(-1)), -127, 127).to(torch.int8)

        self.register_buffer("weight_q", w_q.to(device=target_dev, non_blocking=True))
        self.register_buffer("scale", scale.to(device=target_dev, dtype=torch.bfloat16, non_blocking=True))
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().to(device=target_dev, dtype=torch.bfloat16))
        else:
            self.bias: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        out = F.linear(x, self.weight_q.to(dtype=x.dtype))
        out = out * self.scale.to(dtype=out.dtype)
        if self.bias is not None:
            out = out + self.bias.to(dtype=out.dtype)
        return out


@torch.no_grad()
def quantize_model_int8_weight_only(model: nn.Module, device: torch.device | str) -> nn.Module:
    """Quantize all decoder-layer nn.Linear weights to int8 and stream the model onto ``device``.

    Keeps embed_tokens, lm_head, and norms in bf16 so teacher logit accuracy is preserved
    while halving transformer-layer VRAM (~4.4 GB total for Qwen/Qwen3-4B) and avoiding a
    full bf16 peak on GPU during loading.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:  # type: ignore[union-attr]
            for parent_name, parent in list(layer.named_modules()):
                for child_name, child in list(parent.named_children()):
                    if isinstance(child, nn.Linear):
                        setattr(parent, child_name, Int8WeightOnlyLinear(child, device=device))
            layer.to(device=device)
    model.to(device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
