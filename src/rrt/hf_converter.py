from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from rrt.conversion import (
    BaseInitMethod,
    ConversionReport,
    MatrixDecompositionStats,
    decompose_weight_group_svd,
)
from rrt.layers import LoRAMode, lora_merge_token_threshold

HFSharingStrategy = Literal["cycle", "middle_cycle", "sequence", "middle_sequence"]

DEFAULT_PROJECTION_SPECS: tuple[tuple[str, str], ...] = (
    ("self_attn", "q_proj"),
    ("self_attn", "k_proj"),
    ("self_attn", "v_proj"),
    ("self_attn", "o_proj"),
    ("mlp", "gate_proj"),
    ("mlp", "up_proj"),
    ("mlp", "down_proj"),
)


class SharedWeightLoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with a shared base weight and layer-specific LoRA.

    Multiple virtual decoder layers share the exact same ``weight`` nn.Parameter object
    (identical storage / data_ptr, deduplicated automatically by PyTorch's parameters()
    and optimizers), while each virtual layer owns its own low-rank (lora_A, lora_B).
    This keeps Hugging Face's native per-layer KV cache, RoPE, QK-Norm, and generate()
    working without patching attention code.
    """

    def __init__(
        self,
        shared_weight: nn.Parameter,
        shared_bias: nn.Parameter | None = None,
        lora_rank: int = 0,
        lora_alpha: float | None = None,
        lora_dropout: float = 0.0,
        lora_mode: LoRAMode = "auto",
        use_dora_scale: bool = False,
    ) -> None:
        super().__init__()
        if lora_rank < 0:
            raise ValueError(f"lora_rank must be >= 0, got {lora_rank}")
        if lora_mode not in ("auto", "split", "merge"):
            raise ValueError(f"Unsupported lora_mode: {lora_mode}")

        self.out_features, self.in_features = shared_weight.shape
        self.weight = shared_weight
        self.bias = shared_bias

        self.lora_rank = lora_rank
        self.lora_alpha = float(lora_rank) if (lora_alpha is None and lora_rank > 0) else float(lora_alpha or 1.0)
        self.scaling = self.lora_alpha / lora_rank if lora_rank > 0 else 0.0
        self.lora_dropout_p = lora_dropout
        self.lora_mode: LoRAMode = lora_mode
        self._merge_threshold = lora_merge_token_threshold(self.in_features, self.out_features)
        self.dropout = nn.Dropout(lora_dropout) if (lora_rank > 0 and lora_dropout > 0.0) else nn.Identity()

        dev = shared_weight.device
        dtype = shared_weight.dtype
        if lora_rank > 0:
            self.lora_A: nn.Parameter | None = nn.Parameter(
                torch.empty((lora_rank, self.in_features), device=dev, dtype=dtype)
            )
            self.lora_B: nn.Parameter | None = nn.Parameter(
                torch.zeros((self.out_features, lora_rank), device=dev, dtype=dtype)
            )
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.lora_A = None
            self.lora_B = None

        if use_dora_scale:
            self.out_scale: nn.Parameter | None = nn.Parameter(
                torch.ones((self.out_features,), device=dev, dtype=dtype)
            )
        else:
            self.out_scale = None

    def _should_merge(self, num_tokens: int) -> bool:
        if self.lora_dropout_p > 0.0 and self.training:
            return False
        if self.lora_mode == "merge":
            return True
        if self.lora_mode == "split":
            return False
        return num_tokens >= self._merge_threshold

    def forward(self, x: Tensor) -> Tensor:
        if self.lora_A is None or self.lora_B is None:
            out = F.linear(x, self.weight, self.bias)
        else:
            num_tokens = x.numel() // self.in_features
            if self._should_merge(num_tokens):
                w_eff = self.weight + self.scaling * (self.lora_B @ self.lora_A)
                out = F.linear(x, w_eff, self.bias)
            else:
                out = F.linear(x, self.weight, self.bias)
                delta = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
                out = out + self.scaling * delta

        if self.out_scale is not None:
            out = out * self.out_scale.to(dtype=out.dtype)
        return out

    @torch.no_grad()
    def effective_weight(self) -> Tensor:
        if self.lora_A is None or self.lora_B is None:
            w = self.weight.detach().clone()
        else:
            w = self.weight.detach() + self.scaling * (self.lora_B @ self.lora_A)
        if self.out_scale is not None:
            w = w * self.out_scale.detach().to(dtype=w.dtype).unsqueeze(-1)
        return w

    @torch.no_grad()
    def relative_lora_norm(self) -> float:
        if self.lora_A is None or self.lora_B is None:
            return 0.0
        base_norm = self.weight.float().norm().item()
        if base_norm == 0.0:
            return 0.0
        delta_norm = (self.scaling * (self.lora_B.float() @ self.lora_A.float())).norm().item()
        return delta_norm / base_norm

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, lora_rank={self.lora_rank}, "
            f"scaling={self.scaling:.3g}, dora_scale={self.out_scale is not None}"
        )


def compute_adaptive_layer_ranks(groups: list[list[int]], base_rank: int) -> dict[int, int]:
    """Allocate a budget-preserving U-shaped LoRA rank per tied group index.

    Sensitive entry/exit tied groups receive up to 1.5x ``base_rank`` while redundant
    middle groups receive down to 0.5x ``base_rank``, keeping the exact same total rank
    sum (and parameter budget) as uniform ``base_rank`` and aligning ranks to Tensor Core
    multiples of 16 when ``base_rank >= 32``.
    """
    tied_indices = [idx for idx, g in enumerate(groups) if len(g) > 1]
    n = len(tied_indices)
    if n == 0 or base_rank <= 0:
        return {idx: base_rank for idx in tied_indices}
    if n <= 2:
        return {idx: base_rank for idx in tied_indices}

    step = 16 if base_rank >= 32 else max(1, base_rank // 4)
    min_r = max(step, (base_rank // 2 // step) * step)
    max_r = max(base_rank, ((base_rank * 3 // 2) // step) * step)

    ranks: list[int] = []
    for i in range(n):
        u = i / (n - 1)
        raw = base_rank * (1.0 + 0.5 * math.cos(2.0 * math.pi * u))
        r = int(round(raw / step)) * step
        ranks.append(max(min_r, min(max_r, r)))

    # Preserve exact budget: sum(ranks) == base_rank * n (symmetrically from edges inward)
    target_sum = base_rank * n
    diff = target_sum - sum(ranks)
    idx_order = list(range(n // 2)) + ([n // 2] if n % 2 == 1 else [])
    while abs(diff) >= step:
        changed = False
        for i in idx_order:
            j = n - 1 - i
            if i == j:
                delta = step if diff > 0 else -step
                if min_r <= ranks[i] + delta <= max_r:
                    ranks[i] += delta
                    diff -= delta
                    changed = True
            else:
                if abs(diff) >= 2 * step:
                    delta = step if diff > 0 else -step
                    if min_r <= ranks[i] + delta <= max_r and min_r <= ranks[j] + delta <= max_r:
                        ranks[i] += delta
                        ranks[j] += delta
                        diff -= 2 * delta
                        changed = True
            if abs(diff) < step:
                break
        if not changed:
            break

    return {g_idx: ranks[i] for i, g_idx in enumerate(tied_indices)}


def build_layer_groups(
    num_layers: int,
    num_recursions: int,
    sharing_strategy: HFSharingStrategy = "middle_cycle",
    keep_first: int | None = None,
    keep_last: int | None = None,
) -> list[list[int]]:
    """Partition layer indices 0..num_layers-1 into weight-sharing groups.

    Strategies (Bae et al., 2024):
      - cycle:           [0, U, 2U, ...], [1, U+1, ...], ...
      - middle_cycle:    [0..keep_first-1], tied middle cycle, [L-keep_last..L-1]
      - sequence:        [0, 1, ..., R-1], [R, ..., 2R-1], ...  (adjacent layers tied)
      - middle_sequence: [0..keep_first-1], adjacent tied middle pairs, [L-keep_last..L-1]
    """
    if num_recursions < 1:
        raise ValueError(f"num_recursions must be >= 1, got {num_recursions}")
    if num_recursions == 1:
        return [[i] for i in range(num_layers)]

    is_middle = sharing_strategy in ("middle_cycle", "middle_sequence")
    kf = (1 if is_middle else 0) if keep_first is None else keep_first
    kl = (1 if is_middle else 0) if keep_last is None else keep_last
    if kf < 0 or kl < 0:
        raise ValueError(f"keep_first ({kf}) and keep_last ({kl}) must be >= 0")

    if sharing_strategy in ("cycle", "middle_cycle"):
        middle_count = num_layers - kf - kl
        if middle_count < num_recursions or middle_count % num_recursions != 0:
            raise ValueError(
                f"For '{sharing_strategy}', (num_layers - keep_first - keep_last) = {middle_count} "
                f"must be a positive multiple of num_recursions ({num_recursions})"
            )
        u_mid = middle_count // num_recursions
        groups: list[list[int]] = [[i] for i in range(kf)]
        for idx in range(u_mid):
            groups.append([kf + idx + r * u_mid for r in range(num_recursions)])
        groups.extend([[i] for i in range(num_layers - kl, num_layers)])
        return groups

    if sharing_strategy in ("sequence", "middle_sequence"):
        middle_count = num_layers - kf - kl
        if middle_count < num_recursions or middle_count % num_recursions != 0:
            raise ValueError(
                f"For '{sharing_strategy}', (num_layers - keep_first - keep_last) = {middle_count} "
                f"must be a positive multiple of num_recursions ({num_recursions})"
            )
        u_mid = middle_count // num_recursions
        groups = [[i] for i in range(kf)]
        for g in range(u_mid):
            groups.append([kf + g * num_recursions + r for r in range(num_recursions)])
        groups.extend([[i] for i in range(num_layers - kl, num_layers)])
        return groups

    raise ValueError(f"Unsupported sharing_strategy: {sharing_strategy}")


@torch.no_grad()
def _compute_activation_scales(
    rms_sq: Tensor,
    min_rel: float = 0.35,
    max_rel: float = 3.5,
) -> Tensor:
    """Compute condition-bounded per-channel activation RMS scales s_j = clamp(sqrt(E[x_j^2]) / med, min_rel, max_rel).

    Bounding the dynamic range prevents ill-conditioned inverse whitening from exploding
    weight/LoRA entries on rare or dead channels while preserving AWQ/ASVD salience weighting.
    """
    s = rms_sq.float().clamp(min=1e-8).sqrt()
    med = s.median().clamp(min=1e-6)
    return (s / med).clamp(min=min_rel, max=max_rel)


@torch.no_grad()
def _collect_input_activation_rms(
    model: nn.Module,
    tied_layer_indices: list[int],
    projection_specs: tuple[tuple[str, str], ...],
    calibration_batches: list[Tensor],
) -> tuple[dict[tuple[int, str], Tensor], dict[tuple[str, str], str]]:
    """Collect per-channel mean squared input activations E[x_j^2] for every tied linear input."""
    layers: nn.ModuleList = model.model.layers  # type: ignore[assignment]
    # Deduplicate shared inputs (q/k/v share attn_in; gate/up share mlp_in)
    proj_to_space: dict[tuple[str, str], str] = {}
    space_representative: dict[str, tuple[str, str]] = {}
    for sub, proj in projection_specs:
        if (sub, proj) in (("self_attn", "q_proj"), ("self_attn", "k_proj"), ("self_attn", "v_proj")):
            space = "attn_in"
        elif (sub, proj) in (("mlp", "gate_proj"), ("mlp", "up_proj")):
            space = "mlp_in"
        else:
            space = f"{sub}.{proj}"
        proj_to_space[(sub, proj)] = space
        space_representative.setdefault(space, (sub, proj))

    rms_sq: dict[tuple[int, str], Tensor] = {}
    counts: dict[tuple[int, str], int] = {}
    hooks = []

    def _add_rms(key: tuple[int, str], x: Tensor) -> None:
        x2d = x.detach().float().reshape(-1, x.shape[-1])
        mean_sq = x2d.pow(2).mean(dim=0).cpu()
        if key not in rms_sq:
            rms_sq[key] = mean_sq
            counts[key] = 1
        else:
            rms_sq[key].add_(mean_sq)
            counts[key] += 1

    for li in tied_layer_indices:
        for space, (sub, proj) in space_representative.items():
            mod = getattr(getattr(layers[li], sub), proj)
            hooks.append(mod.register_forward_hook(lambda _m, inp, _o, k=(li, space): _add_rms(k, inp[0])))

    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    use_amp = dev.type == "cuda"
    try:
        for batch in calibration_batches:
            b = batch.to(dev, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                model.model(input_ids=b, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
        if was_training:
            model.train()

    for key in rms_sq:
        rms_sq[key].div_(max(counts[key], 1))
    return rms_sq, proj_to_space


@torch.no_grad()
def _rebalance_low_rank_factors(b_raw: Tensor, a_raw: Tensor, scaling: float) -> tuple[Tensor, Tensor]:
    """Re-factorize B_raw @ A_raw into symmetrically balanced (B, A) with exact B @ A == B_raw @ A_raw / scaling.

    Uses thin QR + tiny r x r SVD so ||B||_F == ||A||_F and singular values split equally,
    preventing asymmetric LoRA norms and outlier gradients after inverse activation un-scaling.
    """
    r = b_raw.shape[1]
    if r == 0:
        return b_raw, a_raw
    qb, rb = torch.linalg.qr(b_raw, mode="reduced")
    qa, ra = torch.linalg.qr(a_raw.T, mode="reduced")
    u_mid, s_mid, vh_mid = torch.linalg.svd(rb @ ra.T, full_matrices=False)
    inv_sqrt_scale = 1.0 / math.sqrt(scaling) if scaling > 0.0 else 1.0
    sr = s_mid.clamp(min=0.0).sqrt() * inv_sqrt_scale
    b_bal = (qb @ u_mid) * sr.unsqueeze(0)
    a_bal = sr.unsqueeze(1) * (vh_mid @ qa.T)
    return b_bal, a_bal


@torch.no_grad()
def _decompose_weight_group_whitened_svd(
    weights: list[Tensor],
    scales: list[Tensor],
    lora_rank: int,
    scaling: float,
    ridge: float = 0.05,
) -> tuple[Tensor, list[Tensor], list[Tensor], float, float, float]:
    """Condition-bounded ASVD decomposition minimizing sum_k ||(W_k - W_eff_k) diag(s_k)||_F^2.

    Uses a strictly convex activation-weighted shared base (prior-anchored to mean(W_k) so
    w_base can never exceed [min_k W_k, max_k W_k]) plus activation-weighted truncated SVD
    with exact QR-SVD symmetric (B, A) rebalancing.
    """
    ws = [w.float() for w in weights]
    dev = ws[0].device
    sc = [s.to(device=dev, dtype=torch.float32) for s in scales]

    # Convex activation-weighted average with ridge prior toward unweighted mean (guaranteed bounded!)
    w_mean = torch.stack(ws, dim=0).mean(dim=0)
    s2_list = [s * s for s in sc]
    denom = sum(s2_list) + ridge
    numer = sum(w * s2.unsqueeze(0) for w, s2 in zip(ws, s2_list)) + ridge * w_mean
    w_base = numer / denom.unsqueeze(0)

    lora_as: list[Tensor] = []
    lora_bs: list[Tensor] = []
    err_s_list: list[float] = []
    err_r_list: list[float] = []
    energy_list: list[float] = []

    for w, s in zip(ws, sc):
        w_norm = (w * s.unsqueeze(0)).norm().clamp(min=1e-12)
        delta_w = w - w_base
        delta_scaled = delta_w * s.unsqueeze(0)
        err_s = (delta_scaled.norm() / w_norm).item()

        if lora_rank > 0:
            u, sv, vh = torch.linalg.svd(delta_scaled, full_matrices=False)
            r = min(lora_rank, sv.numel())
            total_energy = (sv * sv).sum().item()
            kept_energy = (sv[:r] * sv[:r]).sum().item()
            energy = kept_energy / total_energy if total_energy > 0.0 else 1.0

            sr = sv[:r].sqrt()
            b_r = u[:, :r] * sr.unsqueeze(0)
            a_r = (sr.unsqueeze(1) * vh[:r, :]) / s.unsqueeze(0)
            b_r, a_r = _rebalance_low_rank_factors(b_r, a_r, scaling=scaling)

            b = torch.zeros((w.shape[0], lora_rank), device=dev, dtype=torch.float32)
            a = torch.zeros((lora_rank, w.shape[1]), device=dev, dtype=torch.float32)
            b[:, :r] = b_r
            a[:r, :] = a_r

            recon_scaled = (w_base + scaling * (b @ a) - w) * s.unsqueeze(0)
            err_r = (recon_scaled.norm() / w_norm).item()
        else:
            a = torch.empty((0, w.shape[1]), device=dev, dtype=torch.float32)
            b = torch.empty((w.shape[0], 0), device=dev, dtype=torch.float32)
            energy = 0.0
            err_r = err_s

        lora_as.append(a)
        lora_bs.append(b)
        err_s_list.append(err_s)
        err_r_list.append(err_r)
        energy_list.append(energy)

    n = len(ws)
    return (
        w_base,
        lora_as,
        lora_bs,
        sum(err_s_list) / n,
        sum(err_r_list) / n,
        sum(energy_list) / n,
    )


@torch.no_grad()
def convert_hf_model_to_relaxed(
    model: nn.Module,
    num_recursions: int,
    sharing_strategy: HFSharingStrategy = "middle_cycle",
    lora_rank: int = 64,
    lora_alpha: float | None = None,
    lora_dropout: float = 0.0,
    base_init: BaseInitMethod = "mean",
    use_svd: bool = True,
    projection_specs: tuple[tuple[str, str], ...] = DEFAULT_PROJECTION_SPECS,
    keep_first: int | None = None,
    keep_last: int | None = None,
    calibration_batches: list[Tensor] | None = None,
    whitening_ridge: float = 0.05,
    adaptive_rank: bool = False,
    use_dora_scale: bool = False,
) -> ConversionReport:
    """Convert a Hugging Face Llama/Qwen2/Qwen3 causal LM in-place into a Relaxed Recursive model.

    Replaces linear projections in each tied layer group with SharedWeightLoRALinear
    sharing one base weight tensor plus per-layer LoRA adapters initialized via truncated SVD.
    When ``calibration_batches`` is provided with ``use_svd=True``, performs exact
    covariance-whitened SVD (SVD-LLM style) minimizing activation output error.
    When ``adaptive_rank=True``, allocates a budget-preserving U-shaped LoRA rank schedule
    giving sensitive entry/exit tied layers up to 1.5x rank and middle layers down to 0.5x rank.
    When ``use_dora_scale=True``, adds a per-output-channel learnable magnitude vector to
    every shared projection (DoRA-style decoupled magnitude scaling).
    """
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Expected a Hugging Face CausalLM with 'model.model.layers' (e.g. Qwen3, Qwen2, Llama)")

    layers: nn.ModuleList = model.model.layers  # type: ignore[assignment]
    num_layers = len(layers)
    groups = build_layer_groups(
        num_layers,
        num_recursions,
        sharing_strategy,
        keep_first=keep_first,
        keep_last=keep_last,
    )
    group_ranks = (
        compute_adaptive_layer_ranks(groups, lora_rank)
        if adaptive_rank
        else {idx: lora_rank for idx, g in enumerate(groups) if len(g) > 1}
    )

    use_whitened = use_svd and calibration_batches is not None and len(calibration_batches) > 0
    rms_sq: dict[tuple[int, str], Tensor] = {}
    proj_to_space: dict[tuple[str, str], str] = {}
    if use_whitened:
        tied_indices = [li for g in groups if len(g) > 1 for li in g]
        rms_sq, proj_to_space = _collect_input_activation_rms(
            model=model,
            tied_layer_indices=tied_indices,
            projection_specs=projection_specs,
            calibration_batches=calibration_batches,  # type: ignore[arg-type]
        )

    report = ConversionReport(
        base_init=base_init,
        use_svd=use_svd,
        lora_rank=lora_rank,
        sharing_strategy=sharing_strategy,
        num_recursions=num_recursions,
    )

    unique_spaces = tuple(dict.fromkeys(proj_to_space.values())) if use_whitened else ()

    for group_idx, layer_indices in enumerate(groups):
        if len(layer_indices) == 1:
            # Dedicated unshared layer (e.g. entry/exit layers)
            continue

        group_rank = group_ranks[group_idx]
        group_scaling = (
            (float(group_rank) if lora_alpha is None else float(lora_alpha)) / group_rank
            if group_rank > 0
            else 0.0
        )

        group_scales: dict[tuple[int, str], Tensor] = {}
        if use_whitened:
            dev = getattr(getattr(layers[layer_indices[0]], projection_specs[0][0]), projection_specs[0][1]).weight.device
            for li in layer_indices:
                for sp in unique_spaces:
                    group_scales[(li, sp)] = _compute_activation_scales(rms_sq[(li, sp)].to(dev))

        for submodule_name, proj_name in projection_specs:
            src_linears: list[nn.Linear] = []
            for layer_i in layer_indices:
                submod = getattr(layers[layer_i], submodule_name)
                lin = getattr(submod, proj_name)
                if not isinstance(lin, nn.Linear):
                    raise TypeError(
                        f"Expected nn.Linear at layers[{layer_i}].{submodule_name}.{proj_name}, got {type(lin)}"
                    )
                src_linears.append(lin)

            dev = src_linears[0].weight.device
            dtype = src_linears[0].weight.dtype
            weights = [lin.weight for lin in src_linears]

            if use_whitened:
                sp = proj_to_space[(submodule_name, proj_name)]
                w_base, lora_as, lora_bs, err_s, err_r, energy = _decompose_weight_group_whitened_svd(
                    weights=weights,
                    scales=[group_scales[(li, sp)] for li in layer_indices],
                    lora_rank=group_rank,
                    scaling=group_scaling,
                    ridge=whitening_ridge,
                )
            else:
                w_base, lora_as, lora_bs, err_s, err_r, energy = decompose_weight_group_svd(
                    weights=weights,
                    lora_rank=group_rank,
                    scaling=group_scaling,
                    base_init=base_init,
                    use_svd=use_svd,
                )

            shared_weight = nn.Parameter(w_base.to(device=dev, dtype=dtype))

            has_bias = src_linears[0].bias is not None
            shared_bias: nn.Parameter | None = None
            if has_bias:
                mean_bias = torch.stack([lin.bias.float() for lin in src_linears if lin.bias is not None]).mean(dim=0)
                shared_bias = nn.Parameter(mean_bias.to(device=dev, dtype=dtype))

            for k, layer_i in enumerate(layer_indices):
                relaxed = SharedWeightLoRALinear(
                    shared_weight=shared_weight,
                    shared_bias=shared_bias,
                    lora_rank=group_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    use_dora_scale=use_dora_scale,
                )
                if use_svd and relaxed.lora_A is not None and relaxed.lora_B is not None:
                    relaxed.lora_A.copy_(lora_as[k].to(device=dev, dtype=dtype))
                    relaxed.lora_B.copy_(lora_bs[k].to(device=dev, dtype=dtype))

                submod = getattr(layers[layer_i], submodule_name)
                setattr(submod, proj_name, relaxed)

            report.matrix_stats.append(
                MatrixDecompositionStats(
                    name=f"group{group_idx}({layer_indices}).{submodule_name}.{proj_name}",
                    rel_error_strict=err_s,
                    rel_error_relaxed=err_r,
                    energy_captured=energy,
                )
            )

        del group_scales

    return report


def hf_parameter_summary(model: nn.Module) -> dict[str, int]:
    """Count unique parameters in a Hugging Face model split by role (deduplicating tied weights)."""
    seen: set[int] = set()
    counts = {
        "embeddings": 0,
        "shared_base": 0,
        "unshared_linear": 0,
        "lora": 0,
        "norms": 0,
        "total": 0,
    }

    # Build a lookup of parameter id -> whether it belongs to SharedWeightLoRALinear
    shared_weight_ids: set[int] = set()
    for mod in model.modules():
        if isinstance(mod, SharedWeightLoRALinear):
            shared_weight_ids.add(id(mod.weight))
            if mod.bias is not None:
                shared_weight_ids.add(id(mod.bias))

    for name, param in model.named_parameters():
        pid = id(param)
        if pid in seen:
            continue
        seen.add(pid)
        n = param.numel()
        counts["total"] += n

        if "embed_tokens" in name or "lm_head" in name:
            counts["embeddings"] += n
        elif "lora_A" in name or "lora_B" in name or "out_scale" in name:
            counts["lora"] += n
        elif "norm" in name:
            counts["norms"] += n
        elif pid in shared_weight_ids:
            counts["shared_base"] += n
        else:
            counts["unshared_linear"] += n

    return counts
