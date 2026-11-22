from __future__ import annotations

import argparse
from collections import deque
import json
import re
import subprocess
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


_V1_STEP_RE = re.compile(
    r"step\s+(\d+)\s+\[[^\]]+\]\s+\|\s+tok=\s*([\d.]+)M\s+\|\s+ce=([\d.]+)\s+kl=([\d.]+)\s+\|\s+lr=([\d.eE+-]+)\s+\|\s+(\d+)\s+tok/s\s+\|\s+vram=([\d.]+)GB\s+\|\s+elapsed=([\d.]+)m"
)
_V1_EVAL_RE = re.compile(
    r">>> \[EVAL step (\d+)\]\s+val_ce=([\d.]+)\s+\|\s+val_kl=([\d.]+)\s+\|\s+val_ppl=([\d.]+)\s+\(teacher=([\d.]+)\)\s+\|\s+mean_lora_rel=([\d.]+)"
)
_OPD_RE = re.compile(r"\[OPD step\s+(\d+)\]\s+gen_toks=\s*(\d+)\s+\|\s+opd_jsd=([\d.]+)")
_EMA_START_RE = re.compile(r"CPU-EMA=start@(\d+)%")
_OPD_CFG_RE = re.compile(r"OPD=([^)]+\)|[^,]+)")
_HID_CFG_RE = re.compile(r"Hid=([^)]+\)|[^,]+)")
_RIDGE_RE = re.compile(r"ridge-init hid:\s*([\d.]+)\s*->\s*([\d.]+)")
_INIT_FROM_RE = re.compile(r"Inherited RRT architecture from\s+(\S+)")


def query_gpu_telemetry() -> dict:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,temperature.gpu,power.draw,power.limit,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=2,
            text=True,
        ).strip()
        parts = [p.strip() for p in out.splitlines()[0].split(",")]
        return {
            "name": parts[0],
            "temp_c": float(parts[1]),
            "power_w": float(parts[2]),
            "power_limit_w": float(parts[3]),
            "mem_used_gb": round(float(parts[4]) / 1024.0, 2),
            "mem_total_gb": round(float(parts[5]) / 1024.0, 2),
            "util_pct": float(parts[6]),
        }
    except Exception:
        return {}


def parse_legacy_v1_log(log_path: Path | None) -> dict:
    if log_path is None or not log_path.exists():
        return {"meta": {}, "train": [], "eval": [], "samples": []}
    train_pts: list[dict] = []
    eval_pts: list[dict] = []
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = _V1_STEP_RE.search(line)
                if m:
                    train_pts.append(
                        {
                            "step": int(m.group(1)),
                            "tokens_m": float(m.group(2)),
                            "ce": float(m.group(3)),
                            "kl": float(m.group(4)),
                            "lr": float(m.group(5)),
                            "tps": float(m.group(6)),
                            "vram_gb": float(m.group(7)),
                            "elapsed_min": float(m.group(8)),
                        }
                    )
                    continue
                em = _V1_EVAL_RE.search(line)
                if em:
                    step = int(em.group(1))
                    eval_pts.append(
                        {
                            "step": step,
                            "tokens_m": round(step * 8176 / 1e6, 3),
                            "mean_kl": float(em.group(3)),
                            "val_ppl": float(em.group(4)),
                            "wt_ppl": float(em.group(4)),
                            "teacher_ppl": float(em.group(5)),
                            "lora_rel": float(em.group(6)),
                        }
                    )
    except OSError:
        pass
    if len(train_pts) > 600:
        stride = max(1, len(train_pts) // 600)
        train_pts = train_pts[::stride] + [train_pts[-1]]
    return {
        "meta": {"model": "Qwen/Qwen3-1.7B (v1 10h legacy)", "params_str": "1.196B (62.8% layers)"},
        "train": train_pts,
        "eval": eval_pts,
        "samples": [],
    }


def load_run_jsonl(metrics_path: Path) -> dict:
    meta: dict = {}
    train_pts: list[dict] = []
    eval_pts: list[dict] = []
    samples: list[dict] = []
    if not metrics_path.exists():
        return {"meta": meta, "train": train_pts, "eval": eval_pts, "samples": samples}

    try:
        with metrics_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = rec.get("type")
                if kind == "meta":
                    meta.update(rec)
                elif kind == "train":
                    train_pts.append(rec)
                elif kind == "eval":
                    eval_pts.append(rec)
                    if "samples" in rec:
                        samples.append(
                            {
                                "step": rec.get("step", 0),
                                "tokens_m": rec.get("tokens_m", 0.0),
                                "elapsed_min": rec.get("elapsed_min", 0.0),
                                "wt_ppl": rec.get("wt_ppl"),
                                "chat_ppl": rec.get("chat_ppl"),
                                "mean_kl": rec.get("mean_kl"),
                                "items": rec["samples"],
                            }
                        )
    except OSError:
        pass

    if len(train_pts) > 800:
        stride = max(1, len(train_pts) // 800)
        train_pts = train_pts[::stride] + [train_pts[-1]]

    return {"meta": meta, "train": train_pts, "eval": eval_pts, "samples": samples[-50:]}


def parse_log_events(log_path: Path | None) -> dict:
    """Parse Stage 2 OPD rollouts, CPU-EMA results, DSKD Ridge init, and parent ckpt from .log."""
    empty = {
        "opd": [],
        "ema_start": None,
        "ema_notes": [],
        "opd_cfg": "",
        "hid_cfg": "",
        "ridge_init": None,
        "init_from_dir": "",
    }
    if log_path is None or not log_path.exists():
        return empty
    opd_pts: list[dict] = []
    ema_start: float | None = None
    ema_notes: list[str] = []
    opd_cfg = ""
    hid_cfg = ""
    ridge_init: dict | None = None
    init_from_dir = ""
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.split("\r")[-1].strip()
                if not line:
                    continue
                if not init_from_dir:
                    im = _INIT_FROM_RE.search(line)
                    if im:
                        parent_ckpt = im.group(1).rstrip(":")
                        init_from_dir = Path(parent_ckpt).parent.name
                rm = _RIDGE_RE.search(line)
                if rm:
                    ridge_init = {"before": float(rm.group(1)), "after": float(rm.group(2))}
                if "Starting distillation:" in line:
                    em = _EMA_START_RE.search(line)
                    if em:
                        ema_start = int(em.group(1)) / 100.0
                    om = _OPD_CFG_RE.search(line)
                    if om:
                        opd_cfg = om.group(1).strip()
                    hm = _HID_CFG_RE.search(line)
                    if hm:
                        hid_cfg = hm.group(1).strip()
                    continue
                m = _OPD_RE.search(line)
                if m:
                    opd_pts.append(
                        {
                            "step": int(m.group(1)),
                            "gen_toks": int(m.group(2)),
                            "opd_jsd": float(m.group(3)),
                        }
                    )
                    continue
                if "[CPU-EMA" in line or "Promoted CPU-EMA" in line:
                    ema_notes.append(line.lstrip("> ").strip())
    except OSError:
        return empty
    return {
        "opd": opd_pts[-300:],
        "ema_start": ema_start,
        "ema_notes": ema_notes[-8:],
        "opd_cfg": opd_cfg,
        "hid_cfg": hid_cfg,
        "ridge_init": ridge_init,
        "init_from_dir": init_from_dir,
    }


def list_checkpoints(run_dirs: list[tuple[str, Path]]) -> list[dict]:
    """List saved .pt checkpoints with size and modification age for active run(s)."""
    now = time.time()
    out: list[dict] = []
    priority = {"best.pt": 0, "ema.pt": 1, "latest.pt": 2}
    for prefix, d in run_dirs:
        if not d.exists() or not d.is_dir():
            continue
        pts = sorted(d.glob("*.pt"), key=lambda p: (priority.get(p.name, 9), p.name))
        for p in pts:
            try:
                st = p.stat()
            except OSError:
                continue
            out.append(
                {
                    "tag": prefix,
                    "name": p.name,
                    "size_gb": round(st.st_size / (1024**3), 2),
                    "age_min": round(max(0.0, (now - st.st_mtime) / 60.0), 1),
                    "mtime": st.st_mtime,
                }
            )
    return out


def _interp_x_for_step(train_pts: list[dict], step: int, key: str) -> float | None:
    if not train_pts:
        return None
    if step <= train_pts[0]["step"]:
        return float(train_pts[0].get(key, 0.0))
    if step >= train_pts[-1]["step"]:
        return float(train_pts[-1].get(key, 0.0))
    for i in range(len(train_pts) - 1):
        s0, s1 = train_pts[i]["step"], train_pts[i + 1]["step"]
        if s0 <= step <= s1:
            v0, v1 = float(train_pts[i].get(key, 0.0)), float(train_pts[i + 1].get(key, 0.0))
            if s1 == s0:
                return v0
            return v0 + ((step - s0) / (s1 - s0)) * (v1 - v0)
    return float(train_pts[-1].get(key, 0.0))


def enrich_opd_coords(events: dict, train_pts: list[dict]) -> dict:
    """Attach tokens_m and elapsed_min coordinates to OPD points so they plot on any X-axis."""
    if not events or not events.get("opd") or not train_pts:
        return events
    enriched = []
    for p in events["opd"]:
        st = int(p["step"])
        ep = dict(p)
        ep["tokens_m"] = round(_interp_x_for_step(train_pts, st, "tokens_m") or 0.0, 4)
        ep["elapsed_min"] = round(_interp_x_for_step(train_pts, st, "elapsed_min") or 0.0, 2)
        enriched.append(ep)
    out = dict(events)
    out["opd"] = enriched
    return out


def stitch_pipeline_runs(
    s1_data: dict,
    s2_data: dict,
    s1_events: dict,
    s2_events: dict,
    s1_id: str,
    s2_id: str,
) -> tuple[dict, dict]:
    """Stitch Stage 1 and Stage 2 onto one continuous step / token / elapsed axis."""
    s1_t = s1_data.get("train", [])
    s1_e = s1_data.get("eval", [])
    s1_s = s1_data.get("samples", [])
    s1_m = dict(s1_data.get("meta", {}))

    s2_t = s2_data.get("train", [])
    s2_e = s2_data.get("eval", [])
    s2_s = s2_data.get("samples", [])
    s2_m = dict(s2_data.get("meta", {}))

    step_off = max(
        int(s1_t[-1]["step"]) if s1_t else 0,
        int(s1_e[-1]["step"]) if s1_e else 0,
    )
    tok_off = max(
        float(s1_t[-1]["tokens_m"]) if s1_t else 0.0,
        float(s1_e[-1]["tokens_m"]) if s1_e else 0.0,
    )
    time_off = max(
        float(s1_t[-1]["elapsed_min"]) if s1_t else 0.0,
        float(s1_e[-1]["elapsed_min"]) if s1_e else 0.0,
    )

    comb_train = [dict(p) for p in s1_t]
    for p in s2_t:
        q = dict(p)
        q["step"] = int(p.get("step", 0)) + step_off
        q["tokens_m"] = round(float(p.get("tokens_m", 0.0)) + tok_off, 4)
        q["elapsed_min"] = round(float(p.get("elapsed_min", 0.0)) + time_off, 2)
        comb_train.append(q)

    comb_eval = [dict(p) for p in s1_e]
    for p in s2_e:
        q = dict(p)
        q["step"] = int(p.get("step", 0)) + step_off
        q["tokens_m"] = round(float(p.get("tokens_m", 0.0)) + tok_off, 4)
        q["elapsed_min"] = round(float(p.get("elapsed_min", 0.0)) + time_off, 2)
        comb_eval.append(q)

    comb_samples = [dict(p) for p in s1_s]
    for p in s2_s:
        q = dict(p)
        q["step"] = int(p.get("step", 0)) + step_off
        q["tokens_m"] = round(float(p.get("tokens_m", 0.0)) + tok_off, 4)
        q["elapsed_min"] = round(float(p.get("elapsed_min", 0.0)) + time_off, 2)
        comb_samples.append(q)

    s2_teacher = s2_m.get("teacher_model") or "Stage 2"
    s2_max_h = float(s2_m.get("max_hours") or 0.0)
    s2_max_s = int(s2_m.get("max_steps") or 0)
    s1_actual_h = (time_off / 60.0) if time_off > 0 else float(s1_m.get("max_hours") or 0.0)
    comb_meta = dict(s2_m or s1_m)
    comb_meta["model"] = f"{s1_id} -> {s2_id} ({s2_teacher})"
    if s2_max_s > 0:
        comb_meta["max_steps"] = step_off + s2_max_s
        comb_meta["max_hours"] = round(s1_actual_h + s2_max_h, 3) if s2_max_h > 0 else 0.0
    else:
        comb_meta["max_steps"] = 0
        comb_meta["max_hours"] = round(s1_actual_h + s2_max_h, 3) if s2_max_h > 0 else round(s1_actual_h, 3)
    comb_meta["stage_boundary"] = {
        "step": step_off,
        "tokens_m": round(tok_off, 4),
        "elapsed_min": round(time_off, 2),
        "s2_max_hours": s2_max_h,
        "s2_max_steps": s2_max_s,
        "label": f"Stage 2 ({s2_teacher.split('/')[-1]})",
    }

    s2_ev_enriched = enrich_opd_coords(s2_events, s2_t)
    comb_opd = []
    for p in s2_ev_enriched.get("opd", []):
        q = dict(p)
        q["step"] = int(p["step"]) + step_off
        q["tokens_m"] = round(float(p.get("tokens_m", 0.0)) + tok_off, 4)
        q["elapsed_min"] = round(float(p.get("elapsed_min", 0.0)) + time_off, 2)
        comb_opd.append(q)

    comb_events = {
        "opd": comb_opd,
        "ema_start": s2_events.get("ema_start") if s2_events.get("ema_start") is not None else s1_events.get("ema_start"),
        "ema_notes": (
            [f"[S1] {n}" for n in s1_events.get("ema_notes", [])[-2:]]
            + [f"[S2] {n}" for n in s2_events.get("ema_notes", [])[-4:]]
        ),
        "opd_cfg": s2_events.get("opd_cfg", ""),
        "hid_cfg": s2_events.get("hid_cfg") or s1_events.get("hid_cfg", ""),
        "ridge_init": s2_events.get("ridge_init"),
        "init_from_dir": s1_id,
    }
    return {"meta": comb_meta, "train": comb_train, "eval": comb_eval, "samples": comb_samples[-60:]}, comb_events


def read_log_tail(log_path: Path | None, max_lines: int = 35) -> list[str]:
    if log_path is None or not log_path.exists():
        return []
    try:
        with log_path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 32768))
            raw = f.read().decode("utf-8", errors="ignore")
    except OSError:
        return []
    lines: list[str] = []
    for raw_line in raw.splitlines():
        # Strip carriage-return tqdm redraws, keeping final segment
        clean = raw_line.split("\r")[-1].strip()
        if not clean or clean.startswith("Loading weights:") or "it/s]" in clean:
            continue
        lines.append(clean)
    return lines[-max_lines:]


def discover_runs(checkpoints_root: Path, default_dir: Path, compare_log: Path | None) -> list[dict]:
    """Discover all JSONL runs, auto-stitched S1->S2 pipelines, and optional legacy v1 log."""
    runs: list[dict] = []
    seen_ids: set[str] = set()

    candidates: list[Path] = []
    if checkpoints_root.exists():
        for child in checkpoints_root.iterdir():
            if child.is_dir() and (child / "metrics.jsonl").exists():
                candidates.append(child)
    if (default_dir / "metrics.jsonl").exists() and default_dir not in candidates:
        candidates.append(default_dir)

    candidates.sort(key=lambda p: (p / "metrics.jsonl").stat().st_mtime, reverse=True)
    jsonl_by_id: dict[str, dict] = {}
    for d in candidates:
        mp = d / "metrics.jsonl"
        rid = d.name
        if rid in seen_ids:
            continue
        seen_ids.add(rid)
        log_guess = d.parent / f"{d.name}.log"
        entry = {
            "id": rid,
            "label": rid,
            "kind": "jsonl",
            "dir": str(d),
            "metrics_path": str(mp),
            "log_path": str(log_guess) if log_guess.exists() else "",
            "mtime": mp.stat().st_mtime,
        }
        runs.append(entry)
        jsonl_by_id[rid] = entry

    # Detect Stage-2 runs initialized from an earlier Stage-1 run to expose stitched continuous pipelines
    pipeline_entries: list[dict] = []
    for entry in list(runs):
        lp = Path(entry["log_path"]) if entry["log_path"] else None
        ev = parse_log_events(lp) if lp else {}
        parent_id = ev.get("init_from_dir", "")
        if not parent_id and "_s2" in entry["id"]:
            prefix = entry["id"].split("_s2")[0] + "_s1"
            if prefix in jsonl_by_id:
                parent_id = prefix
        if parent_id and parent_id in jsonl_by_id and parent_id != entry["id"]:
            pid = f"pipeline:{parent_id}+{entry['id']}"
            if pid not in seen_ids:
                seen_ids.add(pid)
                pipeline_entries.append(
                    {
                        "id": pid,
                        "label": f"Pipeline {parent_id} → {entry['id']} (S1+S2)",
                        "kind": "pipeline",
                        "s1_id": parent_id,
                        "s2_id": entry["id"],
                        "metrics_path": entry["metrics_path"],
                        "log_path": entry["log_path"],
                        "mtime": entry["mtime"],
                    }
                )
    runs = pipeline_entries + runs

    if compare_log is not None and compare_log.exists():
        vid = f"legacy:{compare_log.stem}"
        runs.append(
            {
                "id": vid,
                "label": f"{compare_log.stem} (v1 10h log)",
                "kind": "legacy",
                "dir": str(compare_log.parent / compare_log.stem),
                "metrics_path": str(compare_log),
                "log_path": str(compare_log),
                "mtime": compare_log.stat().st_mtime,
            }
        )
    return runs


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>RRT Live Training Dashboard</title>
<style>
  :root {
    --bg: #0b0f17;
    --panel: #131926;
    --border: #232d42;
    --text: #e6edf7;
    --muted: #8b9bb4;
    --accent: #38bdf8;
    --green: #34d399;
    --amber: #fbbf24;
    --rose: #fb7185;
    --purple: #a78bfa;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.4;
  }
  header {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 12px 20px;
    background: #0f1522;
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    z-index: 10;
  }
  .title-block h1 {
    margin: 0;
    font-size: 17px;
    font-weight: 700;
    letter-spacing: 0.2px;
  }
  .title-block .sub {
    font-size: 12px;
    color: var(--muted);
    margin-top: 2px;
  }
  .badges {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 6px;
  }
  .badge {
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 999px;
    background: #172337;
    border: 1px solid #263957;
    color: #b7d3f2;
  }
  .badge.cyan { color: var(--accent); border-color: #1f4b6e; }
  .badge.green { color: var(--green); border-color: #1d5144; }
  .badge.amber { color: var(--amber); border-color: #5b4416; }
  .badge.purple { color: var(--purple); border-color: #46326e; }
  .controls {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 10px 14px;
    font-size: 12px;
    color: var(--muted);
  }
  .controls label { display: flex; align-items: center; gap: 6px; }
  .controls select, .controls input[type=checkbox] {
    background: var(--panel);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 8px;
    font-size: 12px;
  }
  .preset-group { display: inline-flex; gap: 3px; align-items: center; }
  .preset-btn {
    background: #172133;
    color: var(--muted);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 2px 6px;
    font-size: 11px;
    cursor: pointer;
    font-weight: 600;
  }
  .preset-btn:hover { color: var(--text); border-color: #365077; }
  .preset-btn.active { background: #1c3653; color: var(--accent); border-color: #2f6599; }
  .bar-wrap { position: relative; }
  .bar-marker {
    position: absolute;
    top: -3px;
    height: 16px;
    width: 2px;
    background: rgba(251, 191, 36, 0.85);
    pointer-events: none;
  }
  .bar-marker.wsd { background: rgba(167, 139, 250, 0.9); }
  .bar-marker.stage2 { background: rgba(56, 189, 248, 0.95); }
  .action-btn {
    background: #172337;
    color: #cfe4fb;
    border: 1px solid #294162;
    border-radius: 6px;
    padding: 4px 9px;
    font-size: 11.5px;
    font-weight: 600;
    cursor: pointer;
    transition: border-color 0.12s, background 0.12s;
  }
  .action-btn:hover { background: #1e3250; border-color: var(--accent); color: #fff; }
  .action-btn.copied { background: rgba(52, 211, 153, 0.2); border-color: var(--green); color: var(--green); }
  .png-btn {
    background: transparent;
    color: var(--muted);
    border: 1px solid #24314a;
    border-radius: 5px;
    padding: 1px 6px;
    font-size: 10px;
    font-weight: 700;
    cursor: pointer;
    margin-left: 4px;
  }
  .png-btn:hover { color: var(--accent); border-color: var(--accent); background: rgba(56, 189, 248, 0.08); }
  .ckpt-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 0 0 10px;
    font-size: 11px;
  }
  .ckpt-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 8px;
    border-radius: 7px;
    background: #0f1625;
    border: 1px solid #24334f;
    color: #cbd5e1;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }
  .ckpt-pill.best { border-color: rgba(52, 211, 153, 0.45); color: var(--green); }
  .ckpt-pill.ema { border-color: rgba(56, 189, 248, 0.45); color: var(--accent); }
  .ckpt-pill .c-meta { color: var(--muted); font-size: 10.5px; }
  .bar-marker span {
    position: absolute;
    top: -15px;
    transform: translateX(-50%);
    font-size: 9.5px;
    font-weight: 700;
    color: var(--muted);
    white-space: nowrap;
  }
  .status-strip {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-bottom: 14px;
  }
  @media (max-width: 1050px) {
    .status-strip { grid-template-columns: 1fr; }
  }
  .health-box, .delta-box {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 14px;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px 12px;
    font-size: 12.5px;
  }
  .health-box.ok { border-color: #1f5c4a; background: linear-gradient(90deg, rgba(52, 211, 153, 0.08), var(--panel) 45%); }
  .health-box.warn { border-color: #6b4e16; background: linear-gradient(90deg, rgba(251, 191, 36, 0.11), var(--panel) 45%); }
  .health-box.err { border-color: #7c2334; background: linear-gradient(90deg, rgba(251, 113, 133, 0.14), var(--panel) 45%); }
  .h-badge {
    font-size: 11px;
    font-weight: 700;
    padding: 2px 9px;
    border-radius: 999px;
    text-transform: uppercase;
    letter-spacing: 0.4px;
  }
  .h-badge.ok { background: rgba(52, 211, 153, 0.18); color: var(--green); border: 1px solid rgba(52, 211, 153, 0.4); }
  .h-badge.warn { background: rgba(251, 191, 36, 0.18); color: var(--amber); border: 1px solid rgba(251, 191, 36, 0.4); }
  .h-badge.err { background: rgba(251, 113, 133, 0.2); color: var(--rose); border: 1px solid rgba(251, 113, 133, 0.45); }
  .delta-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 8px;
    border-radius: 7px;
    background: #0f1727;
    border: 1px solid #22304a;
    font-size: 12px;
  }
  .delta-good { color: var(--green); font-weight: 700; }
  .delta-bad { color: var(--rose); font-weight: 700; }
  .delta-flat { color: var(--muted); font-weight: 600; }
  .spark-canvas {
    width: 100%;
    height: 36px !important;
    margin-top: 6px;
    display: block;
    cursor: default !important;
    background: #0d131f;
    border: 1px solid #1d283d;
    border-radius: 6px;
  }
  .sample-nav {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px 10px;
    margin-bottom: 10px;
    padding: 8px 10px;
    background: #0e1422;
    border: 1px solid #202b40;
    border-radius: 8px;
    font-size: 12px;
    color: var(--muted);
  }
  .sample-nav input[type=range] { flex: 1; min-width: 120px; }
  .nav-btn {
    background: #172236;
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 2px 8px;
    cursor: pointer;
    font-weight: 700;
    font-size: 12px;
  }
  .nav-btn:hover { border-color: var(--accent); color: var(--accent); }
  .pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
    background: #172536;
    color: var(--accent);
    border: 1px solid #243b55;
  }
  .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 8px var(--green);
  }
  main { padding: 16px 20px 44px; max-width: 1680px; margin: 0 auto; }
  .progress-wrap {
    margin-bottom: 14px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 11px 16px;
  }
  .progress-head {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    margin-bottom: 18px;
  }
  .bar-bg {
    height: 10px;
    background: #1e293b;
    border-radius: 999px;
    overflow: hidden;
  }
  .bar-fill {
    height: 100%;
    width: 0%;
    background: linear-gradient(90deg, #38bdf8, #34d399);
    transition: width 0.5s ease;
  }
  .kpis {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
    gap: 12px;
    margin-bottom: 16px;
  }
  .kpi {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px;
    transition: border-color 0.15s ease;
  }
  .kpi[data-help]:hover { border-color: #365077; cursor: help; }
  .kpi .lbl { font-size: 11px; text-transform: uppercase; letter-spacing: 0.6px; color: var(--muted); display: flex; align-items: center; justify-content: space-between; gap: 6px; }
  .kpi .val { font-size: 21px; font-weight: 700; margin-top: 4px; }
  .kpi .sub { font-size: 11px; color: var(--muted); margin-top: 2px; }
  .help-q {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 15px;
    height: 15px;
    border-radius: 50%;
    font-size: 10px;
    font-weight: 700;
    color: var(--muted);
    background: #1b2436;
    border: 1px solid #2b3a55;
    flex-shrink: 0;
  }
  [data-help] { cursor: help; }
  .help-tip {
    position: fixed;
    pointer-events: none;
    max-width: 320px;
    background: rgba(11, 15, 23, 0.98);
    border: 1px solid #35507a;
    border-radius: 9px;
    padding: 9px 12px;
    font-size: 12.5px;
    line-height: 1.45;
    color: #e6edf7;
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.55);
    display: none;
    z-index: 120;
    white-space: normal;
  }
  .help-tip strong { color: var(--accent); display: block; margin-bottom: 3px; font-size: 12.5px; }
  .charts {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
    gap: 14px;
    margin-bottom: 18px;
  }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px 10px;
    position: relative;
  }
  .card-head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 8px;
    margin-bottom: 8px;
  }
  .card-head h2 { margin: 0; font-size: 14px; font-weight: 600; }
  .legend { display: flex; flex-wrap: wrap; gap: 10px; font-size: 11px; color: var(--muted); }
  .legend span { display: inline-flex; align-items: center; gap: 5px; }
  .swatch { width: 10px; height: 3px; border-radius: 2px; display: inline-block; }
  canvas { width: 100%; height: 240px; display: block; cursor: crosshair; }
  .tooltip {
    position: fixed;
    pointer-events: none;
    background: rgba(11, 15, 23, 0.96);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 10px;
    font-size: 12px;
    color: var(--text);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45);
    display: none;
    z-index: 99;
    white-space: nowrap;
  }
  .bottom-grid {
    display: grid;
    grid-template-columns: 1.15fr 1fr;
    gap: 14px;
  }
  @media (max-width: 1100px) {
    .bottom-grid { grid-template-columns: 1fr; }
  }
  .samples-card, .log-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
  }
  .samples-card h2, .log-card h2 { margin: 0 0 10px; font-size: 15px; }
  .teacher-ref {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 12px;
    font-size: 12.5px;
  }
  .sample-item {
    border-top: 1px solid var(--border);
    padding: 10px 0;
    font-size: 12.5px;
  }
  .sample-meta {
    font-size: 11.5px;
    color: var(--accent);
    font-weight: 600;
    margin-bottom: 4px;
  }
  .q-line { margin: 3px 0; color: #dbe7f5; }
  .q-tag { color: var(--amber); font-weight: 600; margin-right: 6px; }
  .empty { color: var(--muted); font-size: 13px; padding: 12px 0; }
  .log-box {
    background: #090d14;
    border: 1px solid #1d2739;
    border-radius: 8px;
    padding: 10px 12px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 11.5px;
    line-height: 1.48;
    max-height: 420px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
    color: #c7d4e6;
  }
  .log-line { margin: 1px 0; }
  .log-eval { color: var(--green); font-weight: 600; }
  .log-opd { color: var(--amber); }
  .log-ckpt { color: var(--purple); }
  .log-ema { color: var(--accent); font-weight: 600; }
</style>
</head>
<body>
<header>
  <div class="title-block">
    <h1>Relaxed Recursive Transformer — Live Distillation Dashboard</h1>
    <div class="sub" id="run-sub">Scanning checkpoints...</div>
    <div class="badges" id="run-badges"></div>
  </div>
  <div class="controls">
    <label data-help-title="Selector de Run" data-help="Elige qué experimento visualizar. 'Auto-follow latest' detecta automáticamente nuevos runs y salta solo a Etapa 2 en cuanto empiece a escribir métricas.">Run
      <select id="run-select">
        <option value="auto">Auto-follow latest</option>
      </select>
    </label>
    <label data-help-title="Run de Comparación" data-help="Superpone otro experimento (como v1 de 10h o Etapa 1) en línea gris punteada para ver en directo cuánto más rápido converge el actual.">Compare
      <select id="cmp-select">
        <option value="default">v1 baseline (10h)</option>
        <option value="none">None</option>
      </select>
    </label>
    <label data-help-title="Eje X" data-help="Cambia el eje horizontal entre pasos de optimizador (Step), millones de tokens vistos (Tokens M) o minutos transcurridos (Elapsed).">X-Axis
      <select id="x-mode">
        <option value="step">Step</option>
        <option value="tokens_m">Tokens (M)</option>
        <option value="elapsed_min">Elapsed (min)</option>
      </select>
    </label>
    <label data-help-title="Suavizado EMA" data-help="Aplica media móvil exponencial sobre las curvas ruidosas de entrenamiento (estilo TensorBoard/W&B). Usa los botones rápidos o el deslizador: OFF = valores exactos; 0.6–0.9 = tendencia limpia con los datos crudos tenues detrás.">Smooth
      <span class="preset-group">
        <button type="button" class="preset-btn" data-smooth="0">OFF</button>
        <button type="button" class="preset-btn" data-smooth="0.5">0.5</button>
        <button type="button" class="preset-btn" data-smooth="0.8">0.8</button>
        <button type="button" class="preset-btn" data-smooth="0.9">0.9</button>
      </span>
      <input id="smooth" type="range" min="0" max="0.95" step="0.05" value="0.6" /> <span id="smooth-val">0.60</span>
    </label>
    <label data-help-title="Zoom de Rango / Cola" data-help="Recorta el inicio de las gráficas para inspeccionar el pulido final sin que el pico del paso 0 aplaste la escala. También puedes arrastrar horizontalmente sobre cualquier gráfica para hacer zoom libre (doble clic para restaurar).">Zoom
      <span class="preset-group">
        <button type="button" class="preset-btn zoom-btn active" data-zoom="1">All</button>
        <button type="button" class="preset-btn zoom-btn" data-zoom="0.75">75%</button>
        <button type="button" class="preset-btn zoom-btn" data-zoom="0.5">50%</button>
        <button type="button" class="preset-btn zoom-btn" data-zoom="0.25">25%</button>
      </span>
      <button type="button" class="preset-btn" id="zoom-reset" style="display:none">Reset</button>
    </label>
    <label data-help-title="Escala Logarítmica (Train)" data-help="Activa escala logarítmica en las gráficas de pérdida de entrenamiento (KL/CE y Hidden Loss) para ver con claridad tanto la caída inicial como el pulido fino final."> <input id="log-train" type="checkbox" /> Log Train</label>
    <label data-help-title="Escala Logarítmica (PPL)" data-help="Usa escala logarítmica en el eje vertical de Perplejidad. Muy útil al principio para ver el descenso desde el paso 0 sin aplastar los valores bajos."> <input id="log-y" type="checkbox" /> Log PPL</label>
    <label data-help-title="Recortar Comparación" data-help="Recorta la curva gris de comparación (p. ej. v1 de 13,400 pasos) al progreso actual de este run para comparar paso a paso a pantalla completa. Desactívalo para ver todo el histórico de 10h."><input id="crop-cmp" type="checkbox" checked /> Crop compare</label>
    <button type="button" class="action-btn" id="btn-copy-md" data-help-title="Copiar Resumen Markdown" data-help="Copia al portapapeles una tabla Markdown lista para pegar en tu README de GitHub o borrador de LinkedIn con las métricas actuales, DSKD/OPD/EMA y el delta frente al baseline.">Copy MD</button>
    <span class="preset-group" data-help-title="Idioma / Language" data-help="Cambia los tooltips y diagnósticos del dashboard entre Español (ES) e Inglés (EN).">
      <button type="button" class="preset-btn lang-btn active" data-lang="es">ES</button>
      <button type="button" class="preset-btn lang-btn" data-lang="en">EN</button>
    </span>
    <span class="pill"><span class="dot" id="live-dot"></span><span id="status-txt">Polling 3s</span></span>
  </div>
</header>

<main>
  <div class="progress-wrap">
    <div class="progress-head">
      <span id="prog-left">Progress: —</span>
      <span id="prog-right">ETA: —</span>
    </div>
    <div class="bar-wrap">
      <div class="bar-bg"><div class="bar-fill" id="prog-bar"></div></div>
      <div class="bar-marker stage2" id="mk-stage2" style="left:30%;display:none" data-help-title="Transición a Etapa 2" data-help="Punto de unión en la vista combinada donde termina Etapa 1 (Qwen3-1.7B) y arranca Etapa 2 (Strong-to-Weak con Qwen3-4B int8)."><span id="mk-s2-lbl">Stage 2</span></div>
      <div class="bar-marker" id="mk-ema" style="left:70%;display:none" data-help-title="Inicio CPU-EMA" data-help="Punto del entrenamiento en el que comienza el promediado de pesos en RAM (CPU-EMA / Model Soup) sin consumir VRAM."><span id="mk-ema-lbl">EMA 70%</span></div>
      <div class="bar-marker wsd" id="mk-wsd" style="left:85%;display:none" data-help-title="Fase WSD Decay" data-help="Punto donde el calendario WSD pasa de LR estable al enfriamiento final tipo coseno (último 15% del entrenamiento)."><span>WSD 85%</span></div>
    </div>
  </div>

  <div class="status-strip">
    <div class="health-box ok" id="health-banner" data-help-title="Semáforo de Salud Automático" data-help="Vigila automáticamente tus curvas y GPU: detecta estancamientos (plateau), picos/NaN en la pérdida, caída brusca de tok/s, VRAM >95% o temperatura >80°C. Si todo va bien, muestra la velocidad de convergencia reciente y el estado de OPD/EMA.">
      <span class="h-badge ok" id="health-badge">Analizando</span>
      <span id="health-msg">Esperando suficientes pasos para diagnosticar tendencia...</span>
      <span class="help-q" style="margin-left:auto">?</span>
    </div>
    <div class="delta-box" id="delta-bar" data-help-title="Comparativa Exacta en los Mismos Tokens" data-help="Interpola linealmente el run de comparación (v1 o Etapa 1) exactamente en los mismos millones de tokens que llevas ahora, mostrando si vas más bajo en KL/CE/Val KL y más rápido en tok/s en igualdad de condiciones.">
      <span style="color:var(--muted)">Comparativa al mismo punto: esperando datos...</span>
      <span class="help-q" style="margin-left:auto">?</span>
    </div>
  </div>

  <div class="kpis">
    <div class="kpi" data-help-title="Step / Tokens" data-help="Pasos de optimizador completados y millones de tokens vistos. Cada paso estándar procesa 8,176 tokens (batch=4 × grad_accum=4 × seq=512). Más tokens = mayor recuperación de calidad.">
      <div class="lbl"><span>Step / Tokens</span><span class="help-q">?</span></div><div class="val" id="k-step">—</div><div class="sub" id="k-tok">—</div>
    </div>
    <div class="kpi" data-help-title="Train Div / CE &amp; Hid" data-help="Pérdida en entrenamiento: Div (JSD o KL) mide qué tan cerca están las probabilidades del Student de las del Teacher; CE es Cross-Entropy contra el texto real; hid es la pérdida de alineación de capas ocultas. Cuanto más bajas, mejor.">
      <div class="lbl"><span>Train Div / CE</span><span class="help-q">?</span></div><div class="val" id="k-kl">—</div><div class="sub" id="k-ce">—</div>
    </div>
    <div class="kpi" data-help-title="WikiText-2 Perplexity (PPL)" data-help="Perplejidad sobre inglés enciclopédico (WikiText-2). Mide entre cuántas palabras duda de media el modelo al predecir la siguiente. Cuanto más cerca del Teacher (~18.4), mejor retención de conocimiento general.">
      <div class="lbl"><span>WikiText-2 PPL</span><span class="help-q">?</span></div><div class="val" id="k-wt">—</div><div class="sub" id="k-wt-sub">—</div>
    </div>
    <div class="kpi" data-help-title="SmolTalk Perplexity (PPL)" data-help="Perplejidad sobre conversaciones e instrucciones (SmolTalk test). Cuanto más baja y equilibrada con WikiText-2 (sin disparar WikiText), mejor calidad conversacional sin sobreajuste.">
      <div class="lbl"><span>SmolTalk PPL</span><span class="help-q">?</span></div><div class="val" id="k-chat">—</div><div class="sub" id="k-chat-sub">—</div>
    </div>
    <div class="kpi" data-help-title="Val Mean KL &amp; LoRA Norm" data-help="Promedio de divergencia KL entre Teacher y Student en WikiText-2 y SmolTalk. Es la métrica principal para guardar best.pt (0.0 = idéntico al Teacher). Abajo: ||B·A||/||W|| muestra cuánto aportan los adaptadores LoRA.">
      <div class="lbl"><span>Val Mean KL</span><span class="help-q">?</span></div><div class="val" id="k-vkl">—</div><div class="sub" id="k-lora">—</div>
    </div>
    <div class="kpi" data-help-title="Throughput &amp; Learning Rate" data-help="Velocidad real de entrenamiento en tokens/segundo y tasa de aprendizaje (LR) base actual del calendario WSD/Cosine (LoRA, DoRA y proyectores DSKD entrenan a 2× este LR).">
      <div class="lbl"><span>Throughput</span><span class="help-q">?</span></div><div class="val" id="k-tps">—</div><div class="sub" id="k-lr">—</div>
    </div>
    <div class="kpi" data-help-title="Telemetría RTX 4080 + Sparkline" data-help="Lectura en directo vía nvidia-smi y mini-gráfica histórica en vivo: Ámbar = Temperatura (°C), Azul = Potencia (W), Rojo = VRAM (GB). Permite comprobar de un vistazo si hubo picos térmicos o throttling.">
      <div class="lbl"><span>RTX 4080 Telemetry</span><span class="help-q">?</span></div><div class="val" id="k-gpu">—</div><div class="sub" id="k-gpu-sub">—</div>
      <canvas id="c-gpu-spark" class="spark-canvas"></canvas>
    </div>
  </div>

  <div class="charts">
    <div class="card">
      <div class="card-head">
        <h2 data-help-title="Training Loss (KL/JSD &amp; CE)" data-help="Divergencia KL/JSD (azul) y Cross-Entropy (morado) medidas en cada lote de entrenamiento. Una bajada rápida y sin picos bruscos indica destilación estable; la línea gris punteada compara con v1 u otro run.">Training Loss (KL/JSD &amp; CE) <span class="help-q">?</span></h2>
        <div class="legend">
          <span><i class="swatch" style="background:#38bdf8"></i>Active KL/JSD</span>
          <span><i class="swatch" style="background:#a78bfa"></i>Active CE</span>
          <span><i class="swatch" style="background:#64748b"></i>Compare KL (dashed)</span>
          <button type="button" class="png-btn" data-export="c-train" title="Export high-DPI PNG">PNG</button>
        </div>
      </div>
      <canvas id="c-train"></canvas>
    </div>

    <div class="card">
      <div class="card-head">
        <h2 id="h-val-kl" data-help-title="Validation KL Divergence" data-help="Divergencia KL exacta evaluada cada 200 pasos en bloques fijos de WikiText-2 (azul) y SmolTalk (ámbar), junto con su media (verde). Sirve para comprobar que mejora en ambos dominios a la vez.">Validation KL Divergence (lower = closer to Teacher) <span class="help-q">?</span></h2>
        <div class="legend">
          <span><i class="swatch" style="background:#34d399"></i>Mean KL</span>
          <span><i class="swatch" style="background:#38bdf8"></i>WikiText KL</span>
          <span><i class="swatch" style="background:#fbbf24"></i>SmolTalk KL</span>
          <span><i class="swatch" style="background:#64748b"></i>Compare Val KL</span>
          <button type="button" class="png-btn" data-export="c-val-kl" title="Export high-DPI PNG">PNG</button>
        </div>
      </div>
      <canvas id="c-val-kl"></canvas>
    </div>

    <div class="card">
      <div class="card-head">
        <h2 data-help-title="Validation Perplexity (PPL)" data-help="Perplejidad exp(CE) en WikiText-2 y SmolTalk comparada con la línea roja del Teacher. Cuanto más se acerquen ambas curvas a las referencias del Teacher, mayor calidad real del modelo comprimido.">Validation Perplexity (WikiText-2 &amp; SmolTalk) <span class="help-q">?</span></h2>
        <div class="legend">
          <span><i class="swatch" style="background:#38bdf8"></i>WikiText PPL</span>
          <span><i class="swatch" style="background:#34d399"></i>SmolTalk PPL</span>
          <span><i class="swatch" style="background:#fb7185"></i>Teacher WikiText</span>
          <span><i class="swatch" style="background:#64748b"></i>Compare PPL</span>
          <button type="button" class="png-btn" data-export="c-val-ppl" title="Export high-DPI PNG">PNG</button>
        </div>
      </div>
      <canvas id="c-val-ppl"></canvas>
    </div>

    <div class="card">
      <div class="card-head">
        <h2 data-help-title="Hidden-State Alignment &amp; OPD JSD" data-help="Ámbar: error de alineación interna (coseno + norma relativa) entre capas del Student y del Teacher (13 capas en Etapa 1; 6 proyectores DSKD en Etapa 2). Morado/cian (solo Etapa 2): pérdida JSD de los rollouts On-Policy (OPD) parseados del .log.">Hidden-State Alignment &amp; Stage-2 OPD <span class="help-q">?</span></h2>
        <div class="legend">
          <span><i class="swatch" style="background:#fbbf24"></i>Hidden Cosine+Norm</span>
          <span id="leg-opd" style="display:none"><i class="swatch" style="background:#f472b6"></i>OPD Rollout JSD</span>
          <button type="button" class="png-btn" data-export="c-hid" title="Export high-DPI PNG">PNG</button>
        </div>
      </div>
      <canvas id="c-hid"></canvas>
    </div>

    <div class="card">
      <div class="card-head">
        <h2 data-help-title="Learning Rate &amp; LoRA Relative Norm" data-help="Morado: calendario del Learning Rate base (×1e4) en WSD (warmup → meseta estable → caída final). Verde: norma relativa media ||B·A|| / ||W|| de los adaptadores LoRA por capa compartida.">Learning Rate Schedule &amp; LoRA Relative Norm <span class="help-q">?</span></h2>
        <div class="legend">
          <span><i class="swatch" style="background:#a78bfa"></i>Base LR (×1e4)</span>
          <span><i class="swatch" style="background:#34d399"></i>Mean ||BA||/||W||</span>
          <button type="button" class="png-btn" data-export="c-lr" title="Export high-DPI PNG">PNG</button>
        </div>
      </div>
      <canvas id="c-lr"></canvas>
    </div>

    <div class="card">
      <div class="card-head">
        <h2 data-help-title="Throughput &amp; Peak VRAM" data-help="Azul: velocidad efectiva en miles de tokens por segundo. Rojo: pico máximo de VRAM reservada por PyTorch (GB) sobre los 16 GB de tu RTX 4080.">Throughput (tok/s) &amp; Peak VRAM (GB) <span class="help-q">?</span></h2>
        <div class="legend">
          <span><i class="swatch" style="background:#38bdf8"></i>tok/s (÷1000)</span>
          <span><i class="swatch" style="background:#fb7185"></i>Peak VRAM (GB)</span>
          <button type="button" class="png-btn" data-export="c-perf" title="Export high-DPI PNG">PNG</button>
        </div>
      </div>
      <canvas id="c-perf"></canvas>
    </div>
  </div>

  <div class="bottom-grid">
    <div class="samples-card">
      <h2 data-help-title="Línea Temporal de Muestras Generadas" data-help="Respuestas deterministas a Q1 (HTTPS) y Q2 (Penicilina) guardadas en cada evaluación. Usa el deslizador o las flechas ◀ ▶ para viajar del Paso 0 al último checkpoint y ver en directo cómo el Student pasa de balbucear a responder como el Teacher.">Live Generation Samples (Timeline) <span class="help-q">?</span></h2>
      <div id="teacher-box"></div>
      <div class="sample-nav" id="sample-nav" style="display:none" data-help-title="Deslizador de Evolución" data-help="Mueve la barra para comparar las respuestas en cualquier evaluación guardada. Con 'Auto-latest' activo salta solo a la evaluación más reciente; 'Ver todos' despliega el historial completo.">
        <button type="button" class="nav-btn" id="samp-prev">◀</button>
        <input type="range" id="samp-slider" min="0" max="0" step="1" value="0" />
        <button type="button" class="nav-btn" id="samp-next">▶</button>
        <span id="samp-label" style="color:var(--accent);font-weight:700">Step —</span>
        <label><input type="checkbox" id="samp-auto" checked /> Auto-latest</label>
        <label><input type="checkbox" id="samp-all" /> <span id="lbl-show-all">Ver todos</span></label>
      </div>
      <div id="samples-box" class="empty">No evaluation samples yet — first eval appears after Step 200.</div>
    </div>

    <div class="log-card">
      <h2 id="log-title" data-help-title="Log en Vivo" data-help="Últimas 35 líneas del archivo .log del entrenamiento actual. Destaca en colores las evaluaciones, pasos On-Policy (OPD), guardados de checkpoint y promediado CPU-EMA.">Live Training Log (OPD · DSKD · Checkpoints · CPU-EMA) <span class="help-q">?</span></h2>
      <div class="ckpt-strip" id="ckpt-strip" data-help-title="Checkpoints Guardados" data-help="Archivos .pt detectados en disco para el run activo (best.pt, ema.pt, latest.pt) con su tamaño en GB y cuánto hace que se actualizaron."></div>
      <div class="log-box" id="log-box">Waiting for log file...</div>
    </div>
  </div>
</main>

<div class="tooltip" id="tooltip"></div>
<div class="help-tip" id="help-tip"></div>

<script>
const state = {
  data: null,
  charts: {},
  knownRunsKey: '',
  zoomTail: 1.0,
  customXRange: null,
  drag: null,
  lang: (typeof localStorage !== 'undefined' && localStorage.getItem('rrt_dash_lang') === 'en') ? 'en' : 'es',
};

function t(es, en) {
  return state.lang === 'en' ? en : es;
}

const HELP_EN = {
  'Selector de Run': ['Run Selector', 'Choose which experiment to view. "Auto-follow latest" tracks the newest run automatically, and "Pipeline S1 → S2" stitches Stage 1 and Stage 2 onto one continuous axis.'],
  'Run de Comparación': ['Compare Baseline', 'Overlays another run (such as the 10h v1 baseline or Stage 1) as a dashed gray curve to compare convergence speed directly.'],
  'Eje X': ['X-Axis', 'Switch the horizontal axis between optimizer Steps, Millions of tokens seen, or Elapsed minutes.'],
  'Suavizado EMA': ['EMA Smoothing', 'Applies exponential moving average smoothing to noisy training curves (TensorBoard/W&B style). Use quick presets or slider: OFF = exact values; 0.6–0.9 = clean trend with faint raw curve behind.'],
  'Zoom de Rango / Cola': ['Range / Tail Zoom', 'Skip the early high-loss spike to inspect fine late-stage convergence (All / last 75% / 50% / 25%). You can also click-and-drag horizontally on any chart to zoom freely (double-click to reset).'],
  'Escala Logarítmica (Train)': ['Log Scale (Train)', 'Enables log-10 scale on training loss charts (KL/JSD, CE, Hidden Loss) to view both the steep initial drop and fine late-stage polishing clearly.'],
  'Escala Logarítmica (PPL)': ['Log Scale (PPL)', 'Uses log-10 Y-axis on Perplexity charts so early step-0 perplexity does not squash low converged values.'],
  'Recortar Comparación': ['Crop Comparison', 'Trims the dashed gray baseline curve to the active run X-range so early steps stay full-width readable. Disable to inspect the full baseline history.'],
  'Copiar Resumen Markdown': ['Copy Markdown Summary', 'Copies a ready-to-paste Markdown summary table (metrics, DSKD/OPD/EMA status, and same-token delta vs baseline) for GitHub READMEs or LinkedIn posts.'],
  'Idioma / Language': ['Language / Idioma', 'Switch dashboard tooltips, health monitor messages, and labels between Spanish (ES) and English (EN).'],
  'Transición a Etapa 2': ['Stage-2 Transition', 'Boundary in the stitched Pipeline view where Stage 1 (Qwen3-1.7B self-heal) finishes and Stage 2 (Strong-to-Weak Qwen3-4B int8) begins.'],
  'Inicio CPU-EMA': ['CPU-EMA Start', 'Point where zero-VRAM host RAM weight averaging (CPU-EMA / Model Soup) begins accumulating smoothed weights.'],
  'Fase WSD Decay': ['WSD Decay Phase', 'Point where Warmup-Stable-Decay transitions from constant LR into the final cosine cooldown (last 15% of training).'],
  'Semáforo de Salud Automático': ['Automatic Health Monitor', 'Continuously watches loss curves and GPU telemetry for NaN/Inf, loss spikes, plateaus, throughput drops, VRAM >95%, or temp >80°C, while summarizing recent convergence speed and OPD/DSKD/EMA status.'],
  'Comparativa Exacta en los Mismos Tokens': ['Same-Token Baseline Delta', 'Linearly interpolates the comparison baseline at your exact current token count to show KL, CE, Val KL, and throughput deltas apples-to-apples.'],
  'Step / Tokens': ['Step / Tokens', 'Completed optimizer steps and millions of tokens processed (each standard step processes 8,176 tokens: batch=4 × grad_accum=4 × seq=512).'],
  'Train Div / CE & Hid': ['Train Div / CE & Hid', 'Training losses averaged over the logging window: Div (KL or JSD) measures closeness to Teacher probabilities; CE is cross-entropy on text; hid is hidden-state alignment loss. Lower is better.'],
  'WikiText-2 Perplexity (PPL)': ['WikiText-2 Perplexity (PPL)', 'Encyclopedic English perplexity on fixed WikiText-2 blocks. Closer to Teacher reference means stronger general knowledge retention.'],
  'SmolTalk Perplexity (PPL)': ['SmolTalk Perplexity (PPL)', 'Conversational & instruction perplexity on fixed SmolTalk test blocks. Lower values balanced with WikiText-2 indicate strong chat fluency without overfitting.'],
  'Val Mean KL & LoRA Norm': ['Val Mean KL & LoRA Norm', 'Mean forward KL divergence between Teacher and Student across WikiText-2 and SmolTalk (metric used to save best.pt; 0.0 = identical to Teacher). Below: mean ||B·A||/||W|| LoRA contribution.'],
  'Throughput & Learning Rate': ['Throughput & Learning Rate', 'Effective training speed (tokens/second) and current base learning rate (LoRA, DoRA, and DSKD projectors train at 2× base LR).'],
  'Telemetría RTX 4080 + Sparkline': ['GPU Telemetry + Sparkline', 'Live nvidia-smi reading and rolling history sparkline: Amber = Temp (°C), Cyan = Power (W), Rose = VRAM (GB).'],
  'Training Loss (KL/JSD & CE)': ['Training Loss (KL/JSD & CE)', 'Window-averaged training divergence KL/JSD (cyan) and cross-entropy CE (purple). Dashed gray overlays the comparison run. Drag horizontally to zoom; double-click to reset.'],
  'Validation KL Divergence': ['Validation KL Divergence', 'Exact forward KL evaluated on fixed WikiText-2 (cyan), SmolTalk (amber), and their mean (green). Confirms balanced recovery across both domains.'],
  'Validation Perplexity (PPL)': ['Validation Perplexity (PPL)', 'Perplexity exp(CE) on WikiText-2 and SmolTalk versus the dashed rose Teacher line.'],
  'Hidden-State Alignment & OPD JSD': ['Hidden-State Alignment & Stage-2 OPD', 'Amber: cosine + relative-norm hidden alignment loss (13 same-arch taps in Stage 1; 6 Ridge-initialized DSKD projectors in Stage 2). Pink (Stage 2): On-Policy Distillation rollout JSD parsed from .log.'],
  'Learning Rate & LoRA Relative Norm': ['Learning Rate & LoRA Relative Norm', 'Purple: base LR schedule (×1e4). Green: average ||B·A|| / ||W|| specialization norm across shared projections.'],
  'Throughput & Peak VRAM': ['Throughput & Peak VRAM', 'Cyan: effective throughput in thousands of tokens/sec. Rose: peak PyTorch CUDA VRAM allocated (GB).'],
  'Línea Temporal de Muestras Generadas': ['Live Generation Samples (Timeline)', 'Deterministic replies to Q1 (HTTPS) and Q2 (Penicillin) saved at each evaluation. Scrub the slider or use ◀ ▶ to watch quality improve from Step 0 to latest.'],
  'Deslizador de Evolución': ['Timeline Scrubber', 'Move the slider to inspect replies from any checkpoint. Auto-latest jumps to new evals; Show all expands the full history.'],
  'Log en Vivo': ['Live Training Log', 'Tail of the active run .log file with color-coded highlights for evaluations, OPD rollouts, DSKD Ridge init, checkpoints, and CPU-EMA promotions.'],
  'Checkpoints Guardados': ['Saved Checkpoints', 'Detected .pt checkpoint files on disk for the active run (best.pt, ema.pt, latest.pt) with file size (GB) and time since last write.'],
};

function ema(values, alpha) {
  if (!values.length || alpha <= 0) return values.slice();
  const out = [];
  let acc = values[0];
  for (let i = 0; i < values.length; i++) {
    acc = alpha * acc + (1 - alpha) * values[i];
    out.push(acc);
  }
  return out;
}

function fmt(n, digits = 2) {
  if (n === null || n === undefined || Number.isNaN(n)) return '—';
  return Number(n).toFixed(digits);
}

function escapeHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function renderChartToCtx(ctx, W, H, seriesList, opts = {}, dragOverlay = null) {
  const pad = { l: 54, r: 16, t: 12, b: 28 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;

  const useLog = !!opts.logY;
  const transformY = v => (useLog ? Math.log10(Math.max(v, 1e-3)) : v);

  let xs = [], ys = [];
  for (const s of seriesList) {
    if (s.hline !== undefined) {
      ys.push(transformY(s.hline));
      continue;
    }
    for (const pt of s.pts) {
      if (pt.x !== null && pt.y !== null && Number.isFinite(pt.x) && Number.isFinite(pt.y)) {
        xs.push(pt.x);
        ys.push(transformY(pt.y));
      }
    }
  }
  if (!xs.length) {
    ctx.fillStyle = '#64748b';
    ctx.font = '12px ui-sans-serif, system-ui';
    const isZoomed = state.customXRange !== null || state.zoomTail < 0.99;
    const msg = isZoomed
      ? t('Sin puntos en el rango con zoom (doble clic para restaurar)', 'No points in zoomed range (double-click to reset)')
      : t('Esperando puntos de datos...', 'Waiting for data points...');
    ctx.fillText(msg, pad.l + 10, H / 2);
    return null;
  }

  let xMin = Math.min(...xs), xMax = Math.max(...xs);
  if (xMax === xMin) { xMin = Math.max(0, xMin - 1); xMax += 1; }
  let yMin = Math.min(...ys), yMax = Math.max(...ys);
  if (yMax === yMin) { yMin -= 0.5; yMax += 0.5; }
  const yPad = (yMax - yMin) * 0.08;
  yMin -= yPad; yMax += yPad;

  const toX = x => pad.l + ((x - xMin) / (xMax - xMin)) * pw;
  const toY = y => pad.t + ph - ((transformY(y) - yMin) / (yMax - yMin)) * ph;

  ctx.strokeStyle = '#1f2a3e';
  ctx.lineWidth = 1;
  ctx.font = '11px ui-monospace, monospace';
  ctx.fillStyle = '#8b9bb4';
  const yTicks = 5;
  for (let i = 0; i <= yTicks; i++) {
    const frac = i / yTicks;
    const py = pad.t + ph * (1 - frac);
    const val = yMin + (yMax - yMin) * frac;
    const disp = useLog ? Math.pow(10, val) : val;
    ctx.beginPath(); ctx.moveTo(pad.l, py); ctx.lineTo(pad.l + pw, py); ctx.stroke();
    const label = Math.abs(disp) >= 100 ? disp.toFixed(0) : (Math.abs(disp) >= 10 ? disp.toFixed(1) : disp.toFixed(2));
    ctx.fillText(label, 6, py + 4);
  }
  const xTicks = 5;
  for (let i = 0; i <= xTicks; i++) {
    const frac = i / xTicks;
    const px = pad.l + pw * frac;
    const xv = xMin + (xMax - xMin) * frac;
    ctx.beginPath(); ctx.moveTo(px, pad.t); ctx.lineTo(px, pad.t + ph); ctx.stroke();
    const xl = xv >= 100 ? xv.toFixed(0) : xv.toFixed(1);
    ctx.fillText(xl, px - 12, H - 8);
  }

  if (opts.vlines && opts.vlines.length) {
    for (const vl of opts.vlines) {
      if (vl.x === null || vl.x === undefined || !Number.isFinite(vl.x)) continue;
      if (vl.x < xMin || vl.x > xMax) continue;
      const px = toX(vl.x);
      ctx.save();
      ctx.strokeStyle = vl.color || '#fbbf24';
      ctx.globalAlpha = 0.72;
      ctx.lineWidth = 1.25;
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(px, pad.t); ctx.lineTo(px, pad.t + ph); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = vl.color || '#fbbf24';
      ctx.font = '600 10px ui-sans-serif, system-ui';
      const tw = ctx.measureText(vl.label).width;
      const lx = Math.min(pad.l + pw - tw - 2, Math.max(pad.l + 2, px + 4));
      ctx.fillText(vl.label, lx, pad.t + 11);
      ctx.restore();
    }
  }

  for (const s of seriesList) {
    ctx.save();
    ctx.strokeStyle = s.color;
    ctx.lineWidth = s.width || 2;
    if (s.dashed) ctx.setLineDash([5, 4]);
    if (s.hline !== undefined) {
      const py = toY(s.hline);
      ctx.beginPath(); ctx.moveTo(pad.l, py); ctx.lineTo(pad.l + pw, py); ctx.stroke();
      ctx.restore();
      continue;
    }
    if (s.rawPts && s.rawPts.length > 1) {
      ctx.save();
      ctx.strokeStyle = s.color;
      ctx.globalAlpha = 0.22;
      ctx.lineWidth = 1;
      ctx.beginPath();
      s.rawPts.forEach((p, idx) => {
        const px = toX(p.x), py = toY(p.y);
        if (idx === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      ctx.stroke();
      ctx.restore();
    }
    ctx.beginPath();
    s.pts.forEach((p, idx) => {
      const px = toX(p.x), py = toY(p.y);
      if (idx === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.stroke();
    if (s.dots && s.pts.length <= 80) {
      ctx.fillStyle = s.color;
      for (const p of s.pts) {
        ctx.beginPath(); ctx.arc(toX(p.x), toY(p.y), 2.8, 0, Math.PI * 2); ctx.fill();
      }
    }
    ctx.restore();
  }

  if (dragOverlay) {
    const x0 = Math.max(pad.l, Math.min(pad.l + pw, Math.min(dragOverlay.startX, dragOverlay.curX)));
    const x1 = Math.max(pad.l, Math.min(pad.l + pw, Math.max(dragOverlay.startX, dragOverlay.curX)));
    if (x1 - x0 > 3) {
      ctx.save();
      ctx.fillStyle = 'rgba(56, 189, 248, 0.16)';
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.75)';
      ctx.lineWidth = 1;
      ctx.fillRect(x0, pad.t, x1 - x0, ph);
      ctx.strokeRect(x0, pad.t, x1 - x0, ph);
      ctx.restore();
    }
  }

  return { seriesList, opts, xMin, xMax, pad, pw, ph };
}

function drawChart(canvasId, seriesList, opts = {}) {
  const canvas = document.getElementById(canvasId);
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(100, Math.floor(rect.width * dpr));
  canvas.height = Math.max(100, Math.floor(rect.height * dpr));
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, rect.width, rect.height);
  const dragOverlay = (state.drag && state.drag.canvasId === canvasId) ? state.drag : null;
  state.charts[canvasId] = renderChartToCtx(ctx, rect.width, rect.height, seriesList, opts, dragOverlay);
}

function attachHover(canvasId) {
  const canvas = document.getElementById(canvasId);
  const tip = document.getElementById('tooltip');

  canvas.addEventListener('mousedown', e => {
    const info = state.charts[canvasId];
    if (!info || e.button !== 0) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    if (mx < info.pad.l || mx > info.pad.l + info.pw) return;
    state.drag = { canvasId, startX: mx, curX: mx };
  });

  canvas.addEventListener('dblclick', () => {
    if (state.customXRange !== null || state.zoomTail !== 1.0) {
      state.customXRange = null;
      state.zoomTail = 1.0;
      render();
    }
  });

  canvas.addEventListener('mousemove', e => {
    const info = state.charts[canvasId];
    if (!info) { tip.style.display = 'none'; return; }
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    if (state.drag && state.drag.canvasId === canvasId) {
      state.drag.curX = mx;
      drawChart(canvasId, info.seriesList, info.opts);
      tip.style.display = 'none';
      return;
    }
    const { pad, pw, xMin, xMax, seriesList } = info;
    if (mx < pad.l || mx > pad.l + pw) { tip.style.display = 'none'; return; }
    const xVal = xMin + ((mx - pad.l) / pw) * (xMax - xMin);
    const lines = [`<strong>x = ${xVal >= 100 ? xVal.toFixed(0) : xVal.toFixed(2)}</strong>`];
    for (const s of seriesList) {
      if (s.hline !== undefined) {
        lines.push(`<span style="color:${s.color}">●</span> ${escapeHtml(s.name)}: ${fmt(s.hline, 3)}`);
        continue;
      }
      if (!s.pts.length) continue;
      let best = s.pts[0], bestD = Math.abs(s.pts[0].x - xVal);
      for (const p of s.pts) {
        const d = Math.abs(p.x - xVal);
        if (d < bestD) { bestD = d; best = p; }
      }
      lines.push(`<span style="color:${s.color}">●</span> ${escapeHtml(s.name)}: ${fmt(best.y, 3)}`);
    }
    tip.innerHTML = lines.join('<br>');
    tip.style.left = (e.clientX + 14) + 'px';
    tip.style.top = (e.clientY + 14) + 'px';
    tip.style.display = 'block';
  });
  canvas.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
}

window.addEventListener('mouseup', () => {
  if (!state.drag) return;
  const { canvasId, startX, curX } = state.drag;
  state.drag = null;
  const info = state.charts[canvasId];
  if (!info) return;
  const x0Px = Math.max(info.pad.l, Math.min(info.pad.l + info.pw, Math.min(startX, curX)));
  const x1Px = Math.max(info.pad.l, Math.min(info.pad.l + info.pw, Math.max(startX, curX)));
  if (x1Px - x0Px >= 12) {
    const a = info.xMin + ((x0Px - info.pad.l) / info.pw) * (info.xMax - info.xMin);
    const b = info.xMin + ((x1Px - info.pad.l) / info.pw) * (info.xMax - info.xMin);
    if (b > a) {
      state.customXRange = [a, b];
      render();
      return;
    }
  }
  drawChart(canvasId, info.seriesList, info.opts);
});

['c-train', 'c-val-kl', 'c-val-ppl', 'c-hid', 'c-lr', 'c-perf'].forEach(attachHover);

function buildSeries(pts, xKey, yKey, alpha, name, color, extra = {}) {
  const clean = pts.filter(p => p[xKey] !== undefined && p[yKey] !== undefined && p[yKey] !== null);
  const ys = clean.map(p => Number(p[yKey]));
  const sm = alpha > 0 ? ema(ys, alpha) : ys;
  return {
    name,
    color,
    pts: clean.map((p, i) => ({ x: Number(p[xKey]), y: sm[i] })),
    rawPts: alpha > 0 ? clean.map((p, i) => ({ x: Number(p[xKey]), y: ys[i] })) : null,
    ...extra,
  };
}

function syncRunSelectors(runs, activeId, compareId) {
  const key = runs.map(r => r.id).join('|');
  if (key === state.knownRunsKey) return;
  state.knownRunsKey = key;

  const runSel = document.getElementById('run-select');
  const prevRun = runSel.value || 'auto';
  runSel.innerHTML = '<option value="auto">Auto-follow latest</option>' +
    runs.map(r => `<option value="${escapeHtml(r.id)}">${escapeHtml(r.label)}</option>`).join('');
  runSel.value = (prevRun === 'auto' || runs.some(r => r.id === prevRun)) ? prevRun : 'auto';

  const cmpSel = document.getElementById('cmp-select');
  const prevCmp = cmpSel.value || 'default';
  cmpSel.innerHTML = '<option value="default">Default (v1 baseline)</option><option value="none">None</option>' +
    runs.map(r => `<option value="${escapeHtml(r.id)}">${escapeHtml(r.label)}</option>`).join('');
  cmpSel.value = (prevCmp === 'default' || prevCmp === 'none' || runs.some(r => r.id === prevCmp)) ? prevCmp : 'default';
}

function renderBadges(meta, events) {
  const box = document.getElementById('run-badges');
  const badges = [];
  if (meta.teacher_model) {
    badges.push(`<span class="badge cyan">Teacher: ${escapeHtml(meta.teacher_model)}${meta.teacher_int8 ? ' (int8)' : ' (bf16)'}</span>`);
  }
  if (meta.kl_type) {
    const klTxt = meta.kl_type === 'jsd' ? `JSD β=${meta.jsd_beta ?? 0.5}` : 'Forward KL';
    badges.push(`<span class="badge green">${klTxt}</span>`);
  }
  if (meta.lr_schedule) {
    badges.push(`<span class="badge purple">LR: ${escapeHtml(meta.lr_schedule.toUpperCase())}</span>`);
  }
  if (meta.dataset_mix) {
    badges.push(`<span class="badge amber">Mix: ${escapeHtml(meta.dataset_mix)}</span>`);
  }
  if (events && events.hid_cfg && events.hid_cfg !== 'OFF') {
    badges.push(`<span class="badge amber">Hid: ${escapeHtml(events.hid_cfg)}</span>`);
  }
  if (events && events.ridge_init) {
    badges.push(`<span class="badge green">DSKD Ridge: ${fmt(events.ridge_init.before, 2)}→${fmt(events.ridge_init.after, 2)}</span>`);
  }
  if (events && events.opd_cfg && events.opd_cfg !== 'OFF') {
    const nOpd = (events.opd || []).length;
    badges.push(`<span class="badge purple">OPD: ${escapeHtml(events.opd_cfg)}${nOpd ? ` (${nOpd} rollouts)` : ''}</span>`);
  }
  if (events && events.ema_start !== null && events.ema_start !== undefined) {
    badges.push(`<span class="badge cyan">CPU-EMA @${Math.round(events.ema_start * 100)}%</span>`);
  }
  if (meta.stage_boundary) {
    badges.push(`<span class="badge cyan">Stitched S1→S2</span>`);
  }
  if (meta.params_str) {
    badges.push(`<span class="badge">${escapeHtml(meta.params_str)}</span>`);
  }
  box.innerHTML = badges.join('');
}

function renderCheckpoints(ckpts) {
  const box = document.getElementById('ckpt-strip');
  if (!box) return;
  if (!ckpts || !ckpts.length) {
    box.innerHTML = `<span style="color:var(--muted)">${t('Sin checkpoints .pt guardados aún', 'No .pt checkpoints saved yet')}</span>`;
    return;
  }
  const fmtAge = m => {
    if (m < 1) return t('ahora', 'just now');
    if (m < 60) return t(`hace ${Math.round(m)}m`, `${Math.round(m)}m ago`);
    return t(`hace ${(m / 60).toFixed(1)}h`, `${(m / 60).toFixed(1)}h ago`);
  };
  box.innerHTML = ckpts.map(c => {
    const cls = c.name === 'best.pt' ? 'best' : (c.name === 'ema.pt' ? 'ema' : '');
    const pref = c.tag ? `[${escapeHtml(c.tag)}] ` : '';
    return `<span class="ckpt-pill ${cls}">${pref}<strong>${escapeHtml(c.name)}</strong> <span class="c-meta">${fmt(c.size_gb, 2)} GB · ${fmtAge(c.age_min)}</span></span>`;
  }).join('');
}

function interpAtX(pts, xKey, yKey, targetX) {
  const clean = pts.filter(p => p[xKey] !== undefined && p[yKey] !== undefined && p[yKey] !== null && Number.isFinite(Number(p[xKey])) && Number.isFinite(Number(p[yKey])));
  if (!clean.length) return null;
  if (targetX < Number(clean[0][xKey]) || targetX > Number(clean[clean.length - 1][xKey])) return null;
  for (let i = 0; i < clean.length - 1; i++) {
    const x0 = Number(clean[i][xKey]), x1 = Number(clean[i + 1][xKey]);
    if (targetX >= x0 && targetX <= x1) {
      const y0 = Number(clean[i][yKey]), y1 = Number(clean[i + 1][yKey]);
      if (x1 === x0) return y0;
      return y0 + ((targetX - x0) / (x1 - x0)) * (y1 - y0);
    }
  }
  return Number(clean[clean.length - 1][yKey]);
}

function renderDeltaBar(train, ev, cmpT, cmpE, cmpLabel) {
  const box = document.getElementById('delta-bar');
  const q = '<span class="help-q" style="margin-left:auto">?</span>';
  if (!train.length || !cmpT.length) {
    box.innerHTML = `<span style="color:var(--muted)">${t('Comparativa en los mismos tokens: selecciona un baseline en "Compare" o espera datos.', 'Same-token comparison: pick a baseline in "Compare" or wait for data.')}</span>${q}`;
    return;
  }
  const lastT = train[train.length - 1];
  const curTok = Number(lastT.tokens_m || 0);
  const chips = [`<span style="color:var(--muted);font-weight:600">${t('En', 'At')} ${fmt(curTok, 2)}M tok vs ${escapeHtml(cmpLabel)}:</span>`];

  const makeChip = (label, cur, ref, lowerIsBetter = true, digits = 2, suffix = '') => {
    if (ref === null || ref === undefined || !Number.isFinite(ref) || ref === 0) return '';
    const diffPct = ((cur - ref) / Math.abs(ref)) * 100;
    const good = lowerIsBetter ? (diffPct < -1.0) : (diffPct > 1.0);
    const bad = lowerIsBetter ? (diffPct > 1.0) : (diffPct < -1.0);
    const cls = good ? 'delta-good' : (bad ? 'delta-bad' : 'delta-flat');
    const sign = diffPct > 0 ? '+' : '';
    return `<span class="delta-chip"><span>${label}</span> <strong>${fmt(cur, digits)}${suffix}</strong> <span style="color:var(--muted)">vs ${fmt(ref, digits)}</span> <span class="${cls}">(${sign}${diffPct.toFixed(1)}%)</span></span>`;
  };

  // Average last 3 points to avoid single-point jitter
  const tail = train.slice(-3);
  const curKl = tail.reduce((a, b) => a + Number(b.kl || 0), 0) / tail.length;
  const curCe = tail.reduce((a, b) => a + Number(b.ce || 0), 0) / tail.length;
  const curTps = tail.reduce((a, b) => a + Number(b.tps || 0), 0) / tail.length;

  const refKl = interpAtX(cmpT, 'tokens_m', 'kl', curTok);
  const refCe = interpAtX(cmpT, 'tokens_m', 'ce', curTok);
  const refTps = interpAtX(cmpT, 'tokens_m', 'tps', curTok);

  chips.push(makeChip('KL', curKl, refKl, true, 3));
  chips.push(makeChip('CE', curCe, refCe, true, 3));
  const lastE = ev.length ? ev[ev.length - 1] : null;
  if (lastE && lastE.step > 0 && cmpE.length) {
    const refVkl = interpAtX(cmpE, 'tokens_m', 'mean_kl', Number(lastE.tokens_m || 0));
    chips.push(makeChip('Val KL', Number(lastE.mean_kl), refVkl, true, 3));
  }
  chips.push(makeChip(t('Vel', 'Speed'), curTps, refTps, false, 0, ' tok/s'));
  box.innerHTML = chips.filter(Boolean).join('') + q;
}

function renderHealthBanner(train, ev, gpu, events) {
  const box = document.getElementById('health-banner');
  const badge = document.getElementById('health-badge');
  const msg = document.getElementById('health-msg');

  let level = 'ok';
  let label = t('Saludable', 'Healthy');
  const notes = [];

  if (train.length < 3) {
    label = t('Iniciando', 'Warming up');
    notes.push(t('Recogiendo los primeros pasos para diagnosticar convergencia...', 'Collecting initial steps to diagnose convergence...'));
  } else {
    const recent = train.slice(-10);
    const hasNan = recent.some(p => !Number.isFinite(Number(p.kl)) || !Number.isFinite(Number(p.ce)));
    if (hasNan) {
      level = 'err'; label = t('Alerta NaN', 'NaN Alert');
      notes.push(t('Detectado NaN/Inf en la pérdida reciente.', 'NaN/Inf detected in recent loss.'));
    }

    const kls = recent.map(p => Number(p.kl)).sort((a, b) => a - b);
    const medKl = kls[Math.floor(kls.length / 2)] || 1;
    const lastKl = Number(recent[recent.length - 1].kl);
    if (level !== 'err' && recent.length >= 5 && lastKl > medKl * 2.4 && lastKl > 1.0) {
      level = 'err'; label = t('Pico de Loss', 'Loss Spike');
      notes.push(t(`KL saltó a ${fmt(lastKl, 3)} (>2.4× mediana ${fmt(medKl, 3)}).`, `KL spiked to ${fmt(lastKl, 3)} (>2.4× median ${fmt(medKl, 3)}).`));
    }

    const win = train.slice(-Math.min(train.length, 9));
    const headN = Math.max(1, Math.floor(win.length / 3));
    const avg = (arr, k) => arr.reduce((a, b) => a + Number(b[k] || 0), 0) / arr.length;
    const klStart = avg(win.slice(0, headN), 'kl');
    const klEnd = avg(win.slice(-headN), 'kl');
    const ceStart = avg(win.slice(0, headN), 'ce');
    const ceEnd = avg(win.slice(-headN), 'ce');
    const klDrop = klStart > 1e-6 ? ((klStart - klEnd) / klStart) * 100 : 0;
    const ceDrop = ceStart > 1e-6 ? ((ceStart - ceEnd) / ceStart) * 100 : 0;
    const spanSteps = win[win.length - 1].step - win[0].step;

    if (train.length >= 10 && spanSteps >= 60 && klDrop < 2.5 && ceDrop < 1.5 && klEnd > 1.2) {
      if (level === 'ok') { level = 'warn'; label = t('Posible Plateau', 'Slow Progress'); }
      notes.push(t(
        `Descenso lento en últimos ${spanSteps} pasos (KL ${klDrop >= 0 ? '-' : '+'}${Math.abs(klDrop).toFixed(1)}%, CE ${ceDrop >= 0 ? '-' : '+'}${Math.abs(ceDrop).toFixed(1)}%).`,
        `Slow slope over last ${spanSteps} steps (KL ${klDrop >= 0 ? '-' : '+'}${Math.abs(klDrop).toFixed(1)}%, CE ${ceDrop >= 0 ? '-' : '+'}${Math.abs(ceDrop).toFixed(1)}%).`
      ));
    } else if (spanSteps > 0) {
      notes.push(t(
        `Convergiendo en últimos ${spanSteps} pasos: KL ${klDrop >= 0 ? '-' : '+'}${Math.abs(klDrop).toFixed(1)}% · CE ${ceDrop >= 0 ? '-' : '+'}${Math.abs(ceDrop).toFixed(1)}%`,
        `Converging over last ${spanSteps} steps: KL ${klDrop >= 0 ? '-' : '+'}${Math.abs(klDrop).toFixed(1)}% · CE ${ceDrop >= 0 ? '-' : '+'}${Math.abs(ceDrop).toFixed(1)}%`
      ));
    }

    if (train.length >= 6) {
      const tpsArr = train.map(p => Number(p.tps || 0)).sort((a, b) => a - b);
      const medTps = tpsArr[Math.floor(tpsArr.length / 2)] || 0;
      const curTps = Number(train[train.length - 1].tps || 0);
      if (medTps > 0 && curTps < medTps * 0.72) {
        if (level === 'ok') { level = 'warn'; label = 'Throttling?'; }
        notes.push(t(`Velocidad bajó a ${Math.round(curTps)} tok/s (mediana ${Math.round(medTps)}).`, `Throughput dropped to ${Math.round(curTps)} tok/s (median ${Math.round(medTps)}).`));
      }
    }
  }

  if (gpu && gpu.temp_c !== undefined) {
    if (gpu.temp_c >= 86) {
      level = 'err'; label = t('Temp Alta', 'High Temp');
      notes.push(t(`GPU a ${gpu.temp_c}°C (riesgo de throttling térmico).`, `GPU at ${gpu.temp_c}°C (thermal throttling risk).`));
    } else if (gpu.temp_c >= 80) {
      if (level === 'ok') { level = 'warn'; label = t('GPU Caliente', 'Warm GPU'); }
      notes.push(`GPU ${gpu.temp_c}°C.`);
    }
    if (gpu.mem_total_gb > 0 && (gpu.mem_used_gb / gpu.mem_total_gb) >= 0.96) {
      level = 'err'; label = t('VRAM Límite', 'VRAM Full');
      notes.push(`VRAM ${((gpu.mem_used_gb / gpu.mem_total_gb) * 100).toFixed(0)}%.`);
    }
  }

  if (events && events.ridge_init) {
    notes.push(`DSKD Ridge: ${fmt(events.ridge_init.before, 2)}→${fmt(events.ridge_init.after, 2)}`);
  }
  if (events && events.opd && events.opd.length) {
    const lastOpd = events.opd[events.opd.length - 1];
    notes.push(`OPD: ${events.opd.length} rollouts (${t('últ.', 'last')} opd_jsd=${fmt(lastOpd.opd_jsd, 3)} @ step ${lastOpd.step})`);
  }
  if (events && events.ema_notes && events.ema_notes.length) {
    notes.push(events.ema_notes[events.ema_notes.length - 1]);
  }

  box.className = 'health-box ' + level;
  badge.className = 'h-badge ' + level;
  badge.textContent = label;
  msg.textContent = notes.join(' · ');
}

function drawGpuSparkline(hist, gpu) {
  const canvas = document.getElementById('c-gpu-spark');
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  if (rect.width < 20) return;
  canvas.width = Math.floor(rect.width * dpr);
  canvas.height = Math.floor(rect.height * dpr);
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;
  ctx.clearRect(0, 0, W, H);
  if (!hist || hist.length < 2) {
    ctx.fillStyle = '#475569';
    ctx.font = '10px ui-monospace, monospace';
    ctx.fillText('Collecting GPU history...', 6, H / 2 + 3);
    return;
  }
  const pad = 4;
  const drawNormLine = (key, minV, maxV, color) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    hist.forEach((p, i) => {
      const x = pad + (i / Math.max(1, hist.length - 1)) * (W - pad * 2);
      const norm = Math.max(0, Math.min(1, (Number(p[key] || 0) - minV) / Math.max(1e-6, maxV - minV)));
      const y = H - pad - norm * (H - pad * 2);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  };
  drawNormLine('temp_c', 30, 90, '#fbbf24');
  drawNormLine('power_w', 0, (gpu && gpu.power_limit_w) ? gpu.power_limit_w : 320, '#38bdf8');
  drawNormLine('mem_used_gb', 0, (gpu && gpu.mem_total_gb) ? gpu.mem_total_gb : 16, '#fb7185');

  const temps = hist.map(p => Number(p.temp_c || 0));
  const maxT = Math.max(...temps);
  ctx.fillStyle = '#8b9bb4';
  ctx.font = '9.5px ui-monospace, monospace';
  ctx.fillText(`max ${maxT.toFixed(0)}°C · ${hist.length}pts`, 5, 11);
}

function computePhaseVlines(meta, events, train, xKey) {
  const emaFrac = (events && events.ema_start !== null && events.ema_start !== undefined) ? events.ema_start : 0.70;
  const hasWsd = meta.lr_schedule === 'wsd';
  const mkEma = document.getElementById('mk-ema');
  const mkWsd = document.getElementById('mk-wsd');
  const mkS2 = document.getElementById('mk-stage2');

  const out = [];
  const sb = meta.stage_boundary || null;
  let pctS2 = 0;
  if (sb) {
    const sbVal = Number(sb[xKey] || 0);
    if (sbVal > 0) {
      out.push({ x: sbVal, label: sb.label || 'Stage 2', color: '#38bdf8' });
    }
    if (mkS2) {
      if (meta.max_steps > 0 && sb.step > 0) {
        pctS2 = Math.max(2, Math.min(95, (sb.step / meta.max_steps) * 100));
      } else if (meta.max_hours > 0 && sb.elapsed_min > 0) {
        pctS2 = Math.max(2, Math.min(95, (sb.elapsed_min / (meta.max_hours * 60)) * 100));
      }
      if (pctS2 > 0) {
        mkS2.style.display = 'block';
        mkS2.style.left = pctS2 + '%';
        document.getElementById('mk-s2-lbl').textContent = sb.label || 'Stage 2';
      } else {
        mkS2.style.display = 'none';
      }
    }
  } else if (mkS2) {
    mkS2.style.display = 'none';
  }

  if (meta.max_hours > 0 || meta.max_steps > 0) {
    const remPct = sb ? (100 - pctS2) : 100;
    const basePct = sb ? pctS2 : 0;
    mkEma.style.display = 'block';
    mkEma.style.left = (basePct + remPct * emaFrac) + '%';
    document.getElementById('mk-ema-lbl').textContent = `EMA ${Math.round(emaFrac * 100)}%`;
    mkWsd.style.display = hasWsd ? 'block' : 'none';
    if (hasWsd) mkWsd.style.left = (basePct + remPct * 0.85) + '%';
  } else {
    mkEma.style.display = 'none';
    mkWsd.style.display = 'none';
  }

  const lastT = train.length ? train[train.length - 1] : null;
  if (sb) {
    const sbX = Number(sb[xKey] || 0);
    const s2MaxH = Number(sb.s2_max_hours || 0);
    const s2MaxS = Number(sb.s2_max_steps || 0);
    let s2SpanX = null;
    if (xKey === 'elapsed_min' && s2MaxH > 0) {
      s2SpanX = s2MaxH * 60;
    } else if (xKey === 'step') {
      if (s2MaxS > 0) s2SpanX = s2MaxS;
      else if (lastT && (lastT.elapsed_min - sb.elapsed_min) > 0.3 && s2MaxH > 0) {
        s2SpanX = ((lastT.step - sb.step) / (lastT.elapsed_min - sb.elapsed_min)) * (s2MaxH * 60);
      }
    } else if (xKey === 'tokens_m') {
      if (lastT && (lastT.elapsed_min - sb.elapsed_min) > 0.3 && s2MaxH > 0) {
        s2SpanX = ((lastT.tokens_m - sb.tokens_m) / (lastT.elapsed_min - sb.elapsed_min)) * (s2MaxH * 60);
      }
    }
    if (s2SpanX && s2SpanX > 0) {
      out.push({ x: sbX + s2SpanX * emaFrac, label: `S2 EMA ${Math.round(emaFrac * 100)}%`, color: '#fbbf24' });
      if (hasWsd) out.push({ x: sbX + s2SpanX * 0.85, label: 'S2 WSD 85%', color: '#a78bfa' });
    }
    return out;
  }

  let totalX = null;
  if (xKey === 'elapsed_min' && meta.max_hours > 0) {
    totalX = meta.max_hours * 60;
  } else if (xKey === 'step') {
    if (meta.max_steps > 0) totalX = meta.max_steps;
    else if (lastT && lastT.elapsed_min > 0.3 && meta.max_hours > 0) totalX = (lastT.step / lastT.elapsed_min) * (meta.max_hours * 60);
  } else if (xKey === 'tokens_m') {
    if (lastT && lastT.elapsed_min > 0.3 && meta.max_hours > 0) totalX = (lastT.tokens_m / lastT.elapsed_min) * (meta.max_hours * 60);
  }
  if (!totalX) return out;
  out.push({ x: totalX * emaFrac, label: `EMA ${Math.round(emaFrac * 100)}%`, color: '#fbbf24' });
  if (hasWsd) out.push({ x: totalX * 0.85, label: 'WSD 85%', color: '#a78bfa' });
  return out;
}

function renderSamplesTimeline(samples) {
  const nav = document.getElementById('sample-nav');
  const slider = document.getElementById('samp-slider');
  const label = document.getElementById('samp-label');
  const autoChk = document.getElementById('samp-auto');
  const allChk = document.getElementById('samp-all');
  const sBox = document.getElementById('samples-box');

  if (!samples || !samples.length) {
    nav.style.display = 'none';
    sBox.className = 'empty';
    sBox.textContent = 'No evaluation samples yet — first eval appears after Step 200.';
    return;
  }

  nav.style.display = 'flex';
  slider.max = String(samples.length - 1);
  if (autoChk.checked || Number(slider.value) >= samples.length) {
    slider.value = String(samples.length - 1);
  }
  const idx = Math.max(0, Math.min(samples.length - 1, Number(slider.value)));
  const cur = samples[idx];
  label.textContent = `Step ${cur.step} (${idx + 1}/${samples.length})`;

  const toShow = allChk.checked ? samples.slice().reverse() : [cur];
  sBox.className = '';
  sBox.innerHTML = toShow.map(s =>
    `<div class="sample-item">` +
    `<div class="sample-meta">Step ${s.step} · ${fmt(s.tokens_m, 2)}M tok · ${fmt(s.elapsed_min, 1)}m · wt_ppl=${fmt(s.wt_ppl, 2)} · chat_ppl=${fmt(s.chat_ppl, 2)} · mean_kl=${fmt(s.mean_kl, 4)}</div>` +
    s.items.map(it => `<div class="q-line"><span class="q-tag">[${escapeHtml(it.tag)}]</span>${escapeHtml(it.text)}</div>`).join('') +
    `</div>`
  ).join('');
}

function syncSmoothPresets(alpha) {
  document.querySelectorAll('.preset-btn[data-smooth]').forEach(btn => {
    const v = parseFloat(btn.getAttribute('data-smooth'));
    btn.classList.toggle('active', Math.abs(v - alpha) < 0.025);
  });
}

function syncZoomAndLangControls() {
  document.querySelectorAll('.zoom-btn').forEach(btn => {
    const z = parseFloat(btn.getAttribute('data-zoom'));
    btn.classList.toggle('active', !state.customXRange && Math.abs(z - state.zoomTail) < 0.01);
  });
  const rst = document.getElementById('zoom-reset');
  if (rst) rst.style.display = (state.customXRange || state.zoomTail < 0.99) ? 'inline-block' : 'none';
  document.querySelectorAll('.lang-btn').forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('data-lang') === state.lang);
  });
  const showAllLbl = document.getElementById('lbl-show-all');
  if (showAllLbl) showAllLbl.textContent = t('Ver todos', 'Show all');
  const copyBtn = document.getElementById('btn-copy-md');
  if (copyBtn && !copyBtn.classList.contains('copied')) {
    copyBtn.textContent = t('Copiar MD', 'Copy MD');
  }
}

function exportChartPng(canvasId) {
  const src = document.getElementById(canvasId);
  const card = src ? src.closest('.card') : null;
  if (!src || !card || !state.data) return;
  const titleEl = card.querySelector('h2');
  const titleTxt = titleEl ? titleEl.textContent.replace(/\s*\?\s*$/, '').trim() : canvasId;
  const subTxt = document.getElementById('run-sub')?.textContent || state.data.active_run_id;
  const legendItems = Array.from(card.querySelectorAll('.legend span'))
    .filter(sp => sp.style.display !== 'none')
    .map(sp => ({
      color: sp.querySelector('.swatch')?.style.background || '#38bdf8',
      text: sp.textContent.trim(),
    }));

  const scale = 2;
  const padX = 24, headerH = 68, padB = 20;
  const chartW = Math.max(760, Math.round(src.getBoundingClientRect().width));
  const chartH = 300;
  const outW = chartW + padX * 2;
  const outH = headerH + chartH + padB;

  const out = document.createElement('canvas');
  out.width = outW * scale;
  out.height = outH * scale;
  const ctx = out.getContext('2d');
  ctx.scale(scale, scale);

  ctx.fillStyle = '#0b0f17';
  ctx.fillRect(0, 0, outW, outH);
  ctx.strokeStyle = '#232d42';
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, outW - 1, outH - 1);

  ctx.fillStyle = '#e6edf7';
  ctx.font = '700 15px ui-sans-serif, system-ui, sans-serif';
  ctx.fillText(titleTxt, padX, 26);

  ctx.fillStyle = '#8b9bb4';
  ctx.font = '11px ui-monospace, monospace';
  ctx.fillText(subTxt, padX, 44);

  let lx = padX;
  ctx.font = '11px ui-sans-serif, system-ui, sans-serif';
  for (const it of legendItems) {
    ctx.fillStyle = it.color;
    ctx.fillRect(lx, 54, 12, 3);
    ctx.fillStyle = '#cbd5e1';
    ctx.fillText(it.text, lx + 16, 59);
    lx += ctx.measureText(it.text).width + 32;
  }

  const info = state.charts[canvasId];
  if (info) {
    ctx.save();
    ctx.translate(padX, headerH);
    renderChartToCtx(ctx, chartW, chartH, info.seriesList, info.opts, null);
    ctx.restore();
  } else {
    ctx.drawImage(src, padX, headerH, chartW, chartH);
  }

  const a = document.createElement('a');
  const safeRun = String(state.data.active_run_id || 'rrt').replace(/[^a-zA-Z0-9_-]+/g, '_');
  a.download = `${safeRun}_${canvasId.replace(/^c-/, '')}.png`;
  a.href = out.toDataURL('image/png');
  a.click();
}

async function copyMarkdownSummary() {
  const d = state.data;
  if (!d) return;
  const meta = d.run.meta || {};
  const train = d.run.train || [];
  const ev = d.run.eval || [];
  const events = d.events || {};
  const lastT = train[train.length - 1];
  const lastE = ev[ev.length - 1];
  const cmpT = d.compare ? (d.compare.train || []) : [];
  const cmpE = d.compare ? (d.compare.eval || []) : [];

  const lines = [
    `### RRT Distillation Summary — \`${d.active_run_id}\``,
    '',
    '| Metric | Value |',
    '| :--- | :--- |',
    `| Model / Teacher | \`${meta.model || '—'}\` (${meta.teacher_int8 ? 'int8' : 'bf16'}) |`,
    `| Params | ${meta.params_str || '—'} |`,
    `| Progress | Step **${lastT ? lastT.step.toLocaleString() : '—'}** · **${lastT ? fmt(lastT.tokens_m, 2) : '—'}M** tokens · ${lastT ? fmt(lastT.elapsed_min, 1) : '—'} min |`,
    `| Train KL/JSD · CE · Hid | \`${lastT ? fmt(lastT.kl, 3) : '—'}\` · \`${lastT ? fmt(lastT.ce, 3) : '—'}\` · \`${lastT ? fmt(lastT.hid, 4) : '—'}\` |`,
    `| Val WikiText-2 PPL | **${lastE ? fmt(lastE.wt_ppl, 2) : '—'}** (Teacher: ${meta.teacher_wt_ppl ? fmt(meta.teacher_wt_ppl, 2) : '—'}) |`,
    `| Val SmolTalk PPL | **${lastE ? fmt(lastE.chat_ppl, 2) : '—'}** (Teacher: ${meta.teacher_chat_ppl ? fmt(meta.teacher_chat_ppl, 2) : '—'}) |`,
    `| Val Mean KL | **${lastE ? fmt(lastE.mean_kl, 4) : '—'}** |`,
    `| Throughput · Peak VRAM | ${lastT ? Math.round(lastT.tps).toLocaleString() : '—'} tok/s · ${lastT ? fmt(lastT.vram_gb, 2) : '—'} GB |`,
  ];
  if (events.ridge_init) {
    lines.push(`| DSKD Ridge Init | \`${fmt(events.ridge_init.before, 2)} -> ${fmt(events.ridge_init.after, 2)}\` |`);
  }
  if (events.opd && events.opd.length) {
    const lo = events.opd[events.opd.length - 1];
    lines.push(`| On-Policy Rollouts | ${events.opd.length} rollouts (latest \`opd_jsd=${fmt(lo.opd_jsd, 3)}\`) |`);
  }
  if (lastT && cmpT.length) {
    const refKl = interpAtX(cmpT, 'tokens_m', 'kl', Number(lastT.tokens_m || 0));
    const refVkl = (lastE && cmpE.length) ? interpAtX(cmpE, 'tokens_m', 'mean_kl', Number(lastE.tokens_m || 0)) : null;
    if (refKl !== null) {
      const dKl = ((lastT.kl - refKl) / Math.abs(refKl)) * 100;
      const vPart = refVkl !== null ? `, Val KL \`${fmt(lastE.mean_kl, 3)}\` vs \`${fmt(refVkl, 3)}\` (${(((lastE.mean_kl - refVkl) / Math.abs(refVkl)) * 100).toFixed(1)}%)` : '';
      lines.push(`| Same-Token Delta vs ${d.compare_label} | Train KL \`${fmt(lastT.kl, 3)}\` vs \`${fmt(refKl, 3)}\` (${dKl >= 0 ? '+' : ''}${dKl.toFixed(1)}%)${vPart} |`);
    }
  }
  const md = lines.join('\n') + '\n';
  try {
    await navigator.clipboard.writeText(md);
  } catch {
    const ta = document.createElement('textarea');
    ta.value = md;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
  }
  const btn = document.getElementById('btn-copy-md');
  if (btn) {
    btn.classList.add('copied');
    btn.textContent = t('Copiado ✓', 'Copied ✓');
    setTimeout(() => {
      btn.classList.remove('copied');
      btn.textContent = t('Copiar MD', 'Copy MD');
    }, 1800);
  }
}

function renderLogTail(lines, logPath) {
  const title = document.getElementById('log-title');
  title.innerHTML = (logPath ? `Live Log (${escapeHtml(logPath)})` : 'Live Training Log') + ' <span class="help-q">?</span>';
  const box = document.getElementById('log-box');
  if (!lines || !lines.length) {
    box.textContent = t('Sin archivo .log detectado para este run.', 'No .log file detected for this run yet.');
    return;
  }
  const atBottom = (box.scrollHeight - box.scrollTop - box.clientHeight) < 40;
  box.innerHTML = lines.map(ln => {
    let cls = 'log-line';
    if (ln.includes('[EVAL')) cls += ' log-eval';
    else if (ln.includes('[OPD')) cls += ' log-opd';
    else if (ln.includes('[CPU-EMA') || ln.includes('[DSKD Projector]')) cls += ' log-ema';
    else if (ln.includes('[checkpoint]') || ln.includes('Saved new best') || ln.includes('Promoted CPU-EMA')) cls += ' log-ckpt';
    return `<div class="${cls}">${escapeHtml(ln)}</div>`;
  }).join('');
  if (atBottom) box.scrollTop = box.scrollHeight;
}

function render() {
  const d = state.data;
  if (!d) return;

  syncRunSelectors(d.runs || [], d.active_run_id, d.compare_run_id);

  const xKey = document.getElementById('x-mode').value;
  const alpha = parseFloat(document.getElementById('smooth').value);
  const logTrain = document.getElementById('log-train').checked;
  const logY = document.getElementById('log-y').checked;
  const cropCmp = document.getElementById('crop-cmp').checked;
  document.getElementById('smooth-val').textContent = alpha.toFixed(2);
  syncSmoothPresets(alpha);
  syncZoomAndLangControls();

  const meta = d.run.meta || {};
  const train = d.run.train || [];
  const ev = d.run.eval || [];
  const cmpT = d.compare ? (d.compare.train || []) : [];
  const cmpE = d.compare ? (d.compare.eval || []) : [];
  const cmpLabel = d.compare_label || 'Compare';
  const gpu = d.gpu || {};
  const gpuHist = d.gpu_history || [];
  const events = d.events || { opd: [], ema_start: null, ema_notes: [], opd_cfg: '', hid_cfg: '', ridge_init: null };

  // Compute active zoom X window (custom drag-zoom OR tail zoom preset)
  const activeMaxX = train.length ? Math.max(...train.map(p => Number(p[xKey] || 0))) : 0;
  const activeMinX = train.length ? Math.min(...train.map(p => Number(p[xKey] || 0))) : 0;
  let viewMin = -Infinity, viewMax = Infinity;
  if (state.customXRange) {
    [viewMin, viewMax] = state.customXRange;
  } else if (state.zoomTail < 0.99 && activeMaxX > activeMinX) {
    viewMin = activeMinX + (1 - state.zoomTail) * (activeMaxX - activeMinX);
  }
  const inWindow = arr => arr.filter(p => {
    const xv = Number(p[xKey]);
    return Number.isFinite(xv) && xv >= viewMin && xv <= viewMax;
  });

  // Crop comparison overlay to active run's X range so early runs stay full-width readable
  const minFloor = xKey === 'step' ? 50 : (xKey === 'tokens_m' ? 0.5 : 2.0);
  const xCap = (cropCmp && activeMaxX > 0) ? Math.max(activeMaxX * 1.15, minFloor) : Infinity;
  const trainV = inWindow(train);
  const evV = inWindow(ev);
  const cmpTCapped = inWindow(cmpT.filter(p => Number(p[xKey] || 0) <= xCap));
  const cmpECapped = inWindow(cmpE.filter(p => Number(p[xKey] || 0) <= xCap));

  document.getElementById('run-sub').textContent = meta.model
    ? `[${d.active_run_id}] ${meta.model} | ${meta.sharing_strategy || 'middle_sequence'} R=${meta.num_recursions ?? 2} keep_first=${meta.keep_first ?? 3} r=${meta.lora_rank ?? 128}`
    : `Watching ${d.metrics_path}`;
  renderBadges(meta, events);

  const lastT = train[train.length - 1];
  const lastE = ev[ev.length - 1];

  let pct = 0, leftTxt = 'Waiting for run...', rightTxt = 'ETA: —';
  if (lastT) {
    if (meta.max_steps > 0) {
      pct = Math.min(100, (100 * lastT.step) / meta.max_steps);
      leftTxt = `[${d.active_run_id}] Step ${lastT.step.toLocaleString()} / ${meta.max_steps.toLocaleString()} (${pct.toFixed(1)}%)`;
      rightTxt = `Elapsed: ${fmt(lastT.elapsed_min, 1)}m`;
    } else if (meta.max_hours > 0) {
      const totalMin = meta.max_hours * 60;
      pct = Math.min(100, (100 * lastT.elapsed_min) / totalMin);
      leftTxt = `[${d.active_run_id}] Elapsed: ${fmt(lastT.elapsed_min, 1)}m / ${fmt(totalMin, 0)}m (${pct.toFixed(1)}%)`;
      rightTxt = `ETA: ${fmt(Math.max(0, totalMin - lastT.elapsed_min), 1)}m`;
    }
  }
  document.getElementById('prog-bar').style.width = pct + '%';
  document.getElementById('prog-left').textContent = leftTxt;
  document.getElementById('prog-right').textContent = rightTxt;

  const phaseVlines = computePhaseVlines(meta, events, train, xKey);
  renderHealthBanner(train, ev, gpu, events);
  renderDeltaBar(train, ev, cmpT, cmpE, cmpLabel);

  document.getElementById('k-step').textContent = lastT ? lastT.step.toLocaleString() : '—';
  document.getElementById('k-tok').textContent = lastT ? `${fmt(lastT.tokens_m, 2)}M tokens` : '—';
  document.getElementById('k-kl').textContent = lastT ? `${fmt(lastT.kl, 3)} / ${fmt(lastT.ce, 3)}` : '—';
  document.getElementById('k-ce').textContent = lastT ? `hid=${fmt(lastT.hid, 4)}` : '—';
  document.getElementById('k-wt').textContent = lastE ? fmt(lastE.wt_ppl, 2) : '—';
  document.getElementById('k-wt-sub').textContent = meta.teacher_wt_ppl ? `Teacher: ${fmt(meta.teacher_wt_ppl, 2)}` : '—';
  document.getElementById('k-chat').textContent = lastE ? fmt(lastE.chat_ppl, 2) : '—';
  document.getElementById('k-chat-sub').textContent = meta.teacher_chat_ppl ? `Teacher: ${fmt(meta.teacher_chat_ppl, 2)}` : '—';
  document.getElementById('k-vkl').textContent = lastE ? fmt(lastE.mean_kl, 4) : '—';
  document.getElementById('k-lora').textContent = lastE && lastE.lora_rel ? `LoRA ||BA||/||W|| = ${fmt(lastE.lora_rel, 3)}` : '—';
  document.getElementById('k-tps').textContent = lastT ? `${Math.round(lastT.tps).toLocaleString()} tok/s` : '—';
  document.getElementById('k-lr').textContent = lastT ? `LR: ${Number(lastT.lr).toExponential(2)}` : '—';
  document.getElementById('k-gpu').textContent = gpu.temp_c !== undefined ? `${gpu.temp_c}°C · ${Math.round(gpu.power_w)}W` : '—';
  document.getElementById('k-gpu-sub').textContent = gpu.mem_used_gb !== undefined ? `VRAM ${gpu.mem_used_gb}/${gpu.mem_total_gb} GB · Util ${gpu.util_pct}%` : '—';
  drawGpuSparkline(gpuHist, gpu);

  const tLabel = meta.teacher_model || 'Teacher';
  document.getElementById('h-val-kl').innerHTML = `Validation KL Divergence (lower = closer to ${escapeHtml(tLabel)}) <span class="help-q">?</span>`;

  const trainSeries = [
    buildSeries(trainV, xKey, 'kl', alpha, `${d.active_run_id} KL/JSD`, '#38bdf8'),
    buildSeries(trainV, xKey, 'ce', alpha, `${d.active_run_id} CE`, '#a78bfa'),
  ];
  if (cmpTCapped.length) trainSeries.push(buildSeries(cmpTCapped, xKey, 'kl', alpha, `${cmpLabel} KL`, '#64748b', { dashed: true, width: 1.5 }));
  drawChart('c-train', trainSeries, { logY: logTrain, vlines: phaseVlines });

  const valKlSeries = [
    buildSeries(evV, xKey, 'mean_kl', 0, 'Mean KL', '#34d399', { dots: true }),
    buildSeries(evV, xKey, 'wt_kl', 0, 'WikiText KL', '#38bdf8', { width: 1.6 }),
    buildSeries(evV, xKey, 'chat_kl', 0, 'SmolTalk KL', '#fbbf24', { width: 1.6 }),
  ];
  if (cmpECapped.length && !(d.compare_is_legacy && xKey === 'elapsed_min')) {
    valKlSeries.push(buildSeries(cmpECapped, xKey, 'mean_kl', 0, `${cmpLabel} Val KL`, '#64748b', { dashed: true, width: 1.5 }));
  }
  drawChart('c-val-kl', valKlSeries, { vlines: phaseVlines });

  const evPpl = (!logY && evV.length > 2) ? evV.filter(p => p.step > 0) : evV;
  const valPplSeries = [
    buildSeries(evPpl, xKey, 'wt_ppl', 0, 'WikiText PPL', '#38bdf8', { dots: true }),
    buildSeries(evPpl, xKey, 'chat_ppl', 0, 'SmolTalk PPL', '#34d399', { dots: true }),
  ];
  if (meta.teacher_wt_ppl) valPplSeries.push({ name: 'Teacher WikiText', color: '#fb7185', hline: meta.teacher_wt_ppl, dashed: true, width: 1.5 });
  if (cmpECapped.length && !(d.compare_is_legacy && xKey === 'elapsed_min')) {
    const cmpPplPts = (!logY && cmpECapped.length > 2) ? cmpECapped.filter(p => p.step > 0) : cmpECapped;
    const cmpKey = cmpPplPts.some(p => p.wt_ppl !== undefined) ? 'wt_ppl' : 'val_ppl';
    valPplSeries.push(buildSeries(cmpPplPts, xKey, cmpKey, 0, `${cmpLabel} PPL`, '#64748b', { dashed: true, width: 1.5 }));
  }
  drawChart('c-val-ppl', valPplSeries, { logY, vlines: phaseVlines });

  const hidSeries = [buildSeries(trainV, xKey, 'hid', alpha, 'Hidden Alignment Loss', '#fbbf24')];
  const legOpd = document.getElementById('leg-opd');
  const opdPts = inWindow(events.opd || []);
  if (opdPts.length && opdPts.some(p => p[xKey] !== undefined)) {
    legOpd.style.display = 'inline-flex';
    hidSeries.push(buildSeries(opdPts, xKey, 'opd_jsd', alpha, 'OPD Rollout JSD', '#f472b6', { dots: true, width: 1.7 }));
  } else {
    legOpd.style.display = 'none';
  }
  drawChart('c-hid', hidSeries, { logY: logTrain, vlines: phaseVlines });

  drawChart('c-lr', [
    buildSeries(trainV.map(p => ({ ...p, lr_scaled: p.lr * 1e4 })), xKey, 'lr_scaled', 0, 'Base LR (×1e-4)', '#a78bfa'),
    buildSeries(evV, xKey, 'lora_rel', 0, 'LoRA ||BA||/||W||', '#34d399', { dots: true }),
  ], { vlines: phaseVlines });

  drawChart('c-perf', [
    buildSeries(trainV.map(p => ({ ...p, ktps: p.tps / 1000 })), xKey, 'ktps', alpha, 'Throughput (k tok/s)', '#38bdf8'),
    buildSeries(trainV, xKey, 'vram_gb', 0, 'Peak VRAM (GB)', '#fb7185', { width: 1.6 }),
  ], { vlines: phaseVlines });

  const tBox = document.getElementById('teacher-box');
  if (meta.teacher_samples && meta.teacher_samples.length) {
    tBox.innerHTML = `<div class="teacher-ref"><strong>Teacher Reference (${escapeHtml(meta.teacher_model || meta.model || 'Qwen3')}):</strong>` +
      meta.teacher_samples.map(s => `<div class="q-line"><span class="q-tag">[${escapeHtml(s.tag)}]</span>${escapeHtml(s.text)}</div>`).join('') +
      `</div>`;
  } else {
    tBox.innerHTML = '';
  }
  renderSamplesTimeline(d.run.samples || []);

  renderCheckpoints(d.checkpoints || []);
  renderLogTail(d.log_tail || [], d.log_path || '');
}

async function poll() {
  try {
    const runId = document.getElementById('run-select').value || 'auto';
    const cmpId = document.getElementById('cmp-select').value || 'default';
    const url = `/api/data?run=${encodeURIComponent(runId)}&compare=${encodeURIComponent(cmpId)}`;
    const res = await fetch(url, { cache: 'no-store' });
    state.data = await res.json();
    document.getElementById('status-txt').textContent = 'Live · ' + new Date().toLocaleTimeString();
    render();
  } catch (e) {
    document.getElementById('status-txt').textContent = 'Disconnected';
  }
}

['x-mode', 'smooth', 'log-train', 'log-y', 'crop-cmp'].forEach(id => {
  document.getElementById(id).addEventListener('input', () => {
    if (id === 'x-mode') state.customXRange = null;
    render();
  });
  document.getElementById(id).addEventListener('change', () => {
    if (id === 'x-mode') state.customXRange = null;
    render();
  });
});
document.querySelectorAll('.preset-btn[data-smooth]').forEach(btn => {
  btn.addEventListener('click', () => {
    document.getElementById('smooth').value = btn.getAttribute('data-smooth');
    render();
  });
});
document.querySelectorAll('.zoom-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    state.customXRange = null;
    state.zoomTail = parseFloat(btn.getAttribute('data-zoom'));
    render();
  });
});
document.getElementById('zoom-reset').addEventListener('click', () => {
  state.customXRange = null;
  state.zoomTail = 1.0;
  render();
});
document.querySelectorAll('.png-btn[data-export]').forEach(btn => {
  btn.addEventListener('click', () => exportChartPng(btn.getAttribute('data-export')));
});
document.getElementById('btn-copy-md').addEventListener('click', copyMarkdownSummary);
document.querySelectorAll('.lang-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    state.lang = btn.getAttribute('data-lang') || 'es';
    try { localStorage.setItem('rrt_dash_lang', state.lang); } catch {}
    render();
  });
});
document.getElementById('samp-slider').addEventListener('input', () => {
  document.getElementById('samp-auto').checked = false;
  if (state.data) renderSamplesTimeline(state.data.run.samples || []);
});
document.getElementById('samp-prev').addEventListener('click', () => {
  const sl = document.getElementById('samp-slider');
  sl.value = String(Math.max(0, Number(sl.value) - 1));
  document.getElementById('samp-auto').checked = false;
  if (state.data) renderSamplesTimeline(state.data.run.samples || []);
});
document.getElementById('samp-next').addEventListener('click', () => {
  const sl = document.getElementById('samp-slider');
  sl.value = String(Math.min(Number(sl.max), Number(sl.value) + 1));
  if (Number(sl.value) === Number(sl.max)) document.getElementById('samp-auto').checked = true;
  if (state.data) renderSamplesTimeline(state.data.run.samples || []);
});
['samp-auto', 'samp-all'].forEach(id => {
  document.getElementById(id).addEventListener('change', () => {
    if (state.data) renderSamplesTimeline(state.data.run.samples || []);
  });
});
['run-select', 'cmp-select'].forEach(id => {
  document.getElementById(id).addEventListener('change', () => {
    state.customXRange = null;
    poll();
  });
});

// Floating bilingual (ES / EN) explanation tooltip for any element with data-help
(function initHelpTooltips() {
  const helpTip = document.getElementById('help-tip');
  let activeEl = null;
  function placeTip(e) {
    const pad = 14;
    const tw = helpTip.offsetWidth || 300;
    const th = helpTip.offsetHeight || 80;
    let left = e.clientX + pad;
    let top = e.clientY + pad;
    if (left + tw > window.innerWidth - 10) left = Math.max(10, e.clientX - tw - pad);
    if (top + th > window.innerHeight - 10) top = Math.max(10, e.clientY - th - pad);
    helpTip.style.left = left + 'px';
    helpTip.style.top = top + 'px';
  }
  document.addEventListener('mouseover', e => {
    const el = e.target.closest('[data-help]');
    if (!el) return;
    activeEl = el;
    let title = el.getAttribute('data-help-title') || '';
    let body = el.getAttribute('data-help') || '';
    if (state.lang === 'en' && HELP_EN[title]) {
      [title, body] = HELP_EN[title];
    }
    helpTip.innerHTML = (title ? `<strong>${escapeHtml(title)}</strong>` : '') + escapeHtml(body);
    helpTip.style.display = 'block';
    placeTip(e);
  });
  document.addEventListener('mousemove', e => {
    if (!activeEl) return;
    if (!e.target.closest('[data-help]')) {
      activeEl = null;
      helpTip.style.display = 'none';
      return;
    }
    placeTip(e);
  });
  document.addEventListener('mouseout', e => {
    if (!activeEl) return;
    const to = e.relatedTarget;
    if (!to || !to.closest || !to.closest('[data-help]')) {
      activeEl = null;
      helpTip.style.display = 'none';
    }
  });
})();

window.addEventListener('resize', render);
poll();
setInterval(poll, 3000);
</script>
</body>
</html>
"""


def make_handler(metrics_path: Path, compare_log: Path | None, checkpoints_root: Path | None = None):
    default_dir = metrics_path.parent
    root_dir = checkpoints_root if checkpoints_root is not None else default_dir.parent
    legacy_cache: dict[str, tuple[float, dict]] = {}
    events_cache: dict[str, tuple[float, dict]] = {}
    gpu_history: deque[dict] = deque(maxlen=240)
    last_gpu_ts = [0.0]

    def _load_events(log_p: Path | None) -> dict:
        if log_p is None or not log_p.exists():
            return {
                "opd": [],
                "ema_start": None,
                "ema_notes": [],
                "opd_cfg": "",
                "hid_cfg": "",
                "ridge_init": None,
                "init_from_dir": "",
            }
        key = str(log_p)
        mt = log_p.stat().st_mtime
        cached = events_cache.get(key)
        if cached is not None and cached[0] == mt:
            return cached[1]
        parsed = parse_log_events(log_p)
        events_cache[key] = (mt, parsed)
        return parsed

    def _sample_gpu() -> tuple[dict, list[dict]]:
        gpu = query_gpu_telemetry()
        now = time.time()
        if gpu and (now - last_gpu_ts[0] >= 2.0 or not gpu_history):
            last_gpu_ts[0] = now
            gpu_history.append(
                {
                    "t": round(now, 1),
                    "temp_c": gpu["temp_c"],
                    "power_w": gpu["power_w"],
                    "mem_used_gb": gpu["mem_used_gb"],
                    "util_pct": gpu["util_pct"],
                }
            )
        return gpu, list(gpu_history)

    def _load_by_entry(entry: dict, by_id: dict[str, dict]) -> tuple[dict, dict, list[tuple[str, Path]]]:
        if entry["kind"] == "pipeline":
            s1_e = by_id[entry["s1_id"]]
            s2_e = by_id[entry["s2_id"]]
            s1_d = load_run_jsonl(Path(s1_e["metrics_path"]))
            s2_d = load_run_jsonl(Path(s2_e["metrics_path"]))
            s1_ev = _load_events(Path(s1_e["log_path"]) if s1_e["log_path"] else None)
            s2_ev = _load_events(Path(s2_e["log_path"]) if s2_e["log_path"] else None)
            stitched_data, stitched_events = stitch_pipeline_runs(
                s1_d, s2_d, s1_ev, s2_ev, entry["s1_id"], entry["s2_id"]
            )
            ckpt_dirs = [("S1", Path(s1_e["dir"])), ("S2", Path(s2_e["dir"]))]
            return stitched_data, stitched_events, ckpt_dirs

        p = Path(entry["metrics_path"])
        lp = Path(entry["log_path"]) if entry.get("log_path") else None
        if entry["kind"] == "legacy":
            mt = p.stat().st_mtime if p.exists() else -1.0
            cached = legacy_cache.get(entry["id"])
            if cached is not None and cached[0] == mt:
                parsed = cached[1]
            else:
                parsed = parse_legacy_v1_log(p)
                legacy_cache[entry["id"]] = (mt, parsed)
            ckpt_dirs = [("", Path(entry["dir"]))] if entry.get("dir") else []
            return parsed, _load_events(lp), ckpt_dirs

        data = load_run_jsonl(p)
        ev = enrich_opd_coords(_load_events(lp), data.get("train", []))
        ckpt_dirs = [("", Path(entry["dir"]))] if entry.get("dir") else []
        return data, ev, ckpt_dirs

    class DashboardHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                body = HTML_PAGE.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if parsed.path == "/api/data":
                qs = parse_qs(parsed.query)
                req_run = qs.get("run", ["auto"])[0]
                req_cmp = qs.get("compare", ["default"])[0]

                runs = discover_runs(root_dir, default_dir, compare_log)
                by_id = {r["id"]: r for r in runs}

                # Resolve active run (auto -> most recently modified JSONL run, fallback default metrics_path)
                active_entry: dict | None = None
                if req_run == "auto":
                    jsonl_runs = [r for r in runs if r["kind"] == "jsonl"]
                    active_entry = jsonl_runs[0] if jsonl_runs else (runs[0] if runs else None)
                elif req_run in by_id:
                    active_entry = by_id[req_run]

                if active_entry is not None:
                    active_data, active_events, ckpt_dirs = _load_by_entry(active_entry, by_id)
                    active_id = active_entry["id"]
                    active_metrics = active_entry["metrics_path"]
                    active_log = Path(active_entry["log_path"]) if active_entry["log_path"] else None
                else:
                    active_data = load_run_jsonl(metrics_path)
                    active_id = default_dir.name
                    active_metrics = str(metrics_path)
                    guess_log = default_dir.parent / f"{default_dir.name}.log"
                    active_log = guess_log if guess_log.exists() else None
                    active_events = enrich_opd_coords(_load_events(active_log), active_data.get("train", []))
                    ckpt_dirs = [("", default_dir)]

                # Resolve comparison run
                cmp_entry: dict | None = None
                if req_cmp == "default":
                    legacy_runs = [r for r in runs if r["kind"] == "legacy"]
                    cmp_entry = legacy_runs[0] if legacy_runs else None
                elif req_cmp != "none" and req_cmp in by_id and req_cmp != active_id:
                    cmp_entry = by_id[req_cmp]

                if cmp_entry is not None:
                    cmp_data, _, _ = _load_by_entry(cmp_entry, by_id)
                    cmp_id = cmp_entry["id"]
                    cmp_label = cmp_entry["label"]
                    cmp_is_legacy = cmp_entry["kind"] == "legacy"
                else:
                    cmp_data = {"train": [], "eval": []}
                    cmp_id = "none"
                    cmp_label = ""
                    cmp_is_legacy = False

                gpu_now, gpu_hist = _sample_gpu()
                payload = {
                    "timestamp": time.time(),
                    "active_run_id": active_id,
                    "compare_run_id": cmp_id,
                    "compare_label": cmp_label,
                    "compare_is_legacy": cmp_is_legacy,
                    "metrics_path": active_metrics,
                    "log_path": str(active_log) if active_log else "",
                    "log_tail": read_log_tail(active_log, max_lines=35),
                    "events": active_events,
                    "checkpoints": list_checkpoints(ckpt_dirs),
                    "runs": [{"id": r["id"], "label": r["label"], "kind": r["kind"]} for r in runs],
                    "run": active_data,
                    "compare": cmp_data,
                    "gpu": gpu_now,
                    "gpu_history": gpu_hist,
                }
                body = json.dumps(payload).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            self.send_error(HTTPStatus.NOT_FOUND)

    return DashboardHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time TensorBoard-style web dashboard for RRT distillation")
    parser.add_argument("--dir", type=str, default="checkpoints/qwen3_1.7b_rrt_s1", help="Default training output dir")
    parser.add_argument("--checkpoints-root", type=str, default="checkpoints", help="Root dir to auto-discover all runs")
    parser.add_argument("--compare-log", type=str, default="checkpoints/qwen3_1.7b_rrt.log", help="Optional v1 log to overlay")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    metrics_path = Path(args.dir) / "metrics.jsonl"
    compare_log = Path(args.compare_log) if args.compare_log else None
    checkpoints_root = Path(args.checkpoints_root)

    server = ThreadingHTTPServer((args.host, args.port), make_handler(metrics_path, compare_log, checkpoints_root))
    print(f"RRT Live Dashboard running at http://{args.host}:{args.port}", flush=True)
    print(f"  Auto-discover root: {checkpoints_root} (auto-follows newest run)", flush=True)
    if compare_log and compare_log.exists():
        print(f"  Default compare   : {compare_log} (v1 10h baseline)", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
