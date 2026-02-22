#!/usr/bin/env python3
"""
Compile LLaDA2.1 SSD-policy sweep results into markdown tables.

Data sources:
- GSM8K: summary/gsm8k.jsonl (appended by eval_gsm8k_llada.py)
- MBPP: results/{cached,ssd}/{config_str}/MODEL_DIR_NAME/results_*.json

Table layout:
- Baseline table: cached results, cols = threshold settings x block sizes
- Per-policy tables: SSD results, cols = threshold settings
- Sub-cols: gsm8k, mbpp, avg
- Each cell: "acc (speedup)" where speedup = time(cached_b1_tm1_te0.9) / time(this)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

MODEL_DIR_NAME = "inclusionAI__LLaDA2.1-mini"
MBPP_METRIC = "pass_at_1,none"
SPEEDUP_BASELINE_CFG = "cached_b1_tm1_te0.9"

POLICY_ORDER = ["mask_span_length", "score_threshold", "score_hysteresis"]
POLICY_TITLE = {
    "mask_span_length": "mask_span_length",
    "score_threshold": "score_threshold",
    "score_hysteresis": "score_hysteresis",
}

# Columns: (threshold_tag, block_length) pairs
BASELINE_COLUMNS: List[Tuple[str, int]] = [
    ("tm1_te0.9", 1),
    ("tm1_te0.9", 32),
    ("tm0.95_te0.9", 32),
    ("tm0.7_te0.5", 32),
]
SSD_COLUMNS: List[Tuple[str, int]] = [
    ("tm0.95_te0.9", 32),
    ("tm0.7_te0.5", 32),
]
TASKS = ["gsm8k", "mbpp", "avg"]


@dataclass(frozen=True)
class Cell:
    acc: Optional[float] = None
    time_s: Optional[float] = None
    nfe: Optional[float] = None
    ts: Optional[str] = None


# ── Helpers ──


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _mean(xs) -> Optional[float]:
    vals = [x for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


def _nearly_equal(a, b, eps=1e-12):
    if a is None or b is None:
        return False
    return abs(a - b) <= eps


def _fmt_acc(acc: Optional[float], digits: int = 1) -> str:
    if acc is None:
        return "-"
    return f"{acc * 100:.{digits}f}"


def _speedup(base_t: Optional[float], t: Optional[float]) -> Optional[float]:
    if base_t is None or t is None or t <= 0:
        return None
    return base_t / t


def _fmt_speedup(spd: Optional[float], digits: int = 1) -> str:
    if spd is None:
        return "-"
    return f"{spd:.{digits}f}x"


def _choose_newer(prev: Optional[Cell], cand: Cell) -> Cell:
    if prev is None:
        return cand
    if (cand.ts or "") > (prev.ts or ""):
        return cand
    return prev


# ── Config parsing ──


def _parse_config(cfg: str):
    """
    Parse config string -> (generate_fn, block_length, th_tag, policy_suffix | None).

    Examples:
      "cached_b1_tm1_te0.9"         -> ("cached", 1, "tm1_te0.9", None)
      "ssd_b32_tm0.95_te0.9_policy=span_span=1"
                                     -> ("ssd", 32, "tm0.95_te0.9", "policy=span_span=1")
    """
    m = re.match(r"^cached_b(\d+)_(tm[\d.]+_te[\d.]+)$", cfg)
    if m:
        return ("cached", int(m.group(1)), m.group(2), None)
    m = re.match(r"^ssd_b(\d+)_(tm[\d.]+_te[\d.]+)_(policy=.+)$", cfg)
    if m:
        return ("ssd", int(m.group(1)), m.group(2), m.group(3))
    return None


def _policy_from_suffix(suffix: str) -> Optional[str]:
    if suffix.startswith("policy=span_"):
        return "mask_span_length"
    if suffix.startswith("policy=score_"):
        return "score_threshold"
    if suffix.startswith("policy=hysteresis_"):
        return "score_hysteresis"
    return None


def _row_label(suffix: str) -> str:
    """Strip 'policy=' prefix for display."""
    return suffix[len("policy="):] if suffix.startswith("policy=") else suffix


def _sort_policy_rows(policy: str, rows: List[str]) -> List[str]:
    if policy == "mask_span_length":
        def key_fn(r):
            m = re.match(r"^span_span=(\d+)$", r)
            return int(m.group(1)) if m else 10**9
        return sorted(rows, key=key_fn)
    return sorted(rows)


# ── Data loading ──


def load_gsm8k(summary_paths: Sequence[Path]) -> Dict[str, Cell]:
    """Load GSM8K results from summary JSONL files. Returns {config_str: Cell}."""
    data: Dict[str, Cell] = {}
    for p in summary_paths:
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                cfg = str(row.get("config", ""))
                if not cfg:
                    continue
                cand = Cell(
                    acc=_safe_float(row.get("acc")),
                    time_s=_safe_float(row.get("eval_seconds")),
                    nfe=_safe_float(row.get("avg_nfe")),
                    ts=str(row.get("ts")) if row.get("ts") else None,
                )
                data[cfg] = _choose_newer(data.get(cfg), cand)
    return data


def load_mbpp(result_roots: Sequence[Path], model_dir_name: str) -> Dict[str, Cell]:
    """Load MBPP results from lm_eval output dirs. Returns {config_str: Cell}."""
    data: Dict[str, Cell] = {}
    for root in result_roots:
        for subdir in ["cached", "ssd"]:
            d = root / subdir
            if not d.is_dir():
                continue
            for cfg_dir in d.iterdir():
                if not cfg_dir.is_dir():
                    continue
                cfg = cfg_dir.name
                model_dir = cfg_dir / model_dir_name
                if not model_dir.is_dir():
                    continue
                files = sorted(model_dir.glob("results_*.json"))
                if not files:
                    continue
                latest = files[-1]
                try:
                    j = json.loads(latest.read_text(encoding="utf-8"))
                except Exception:
                    continue
                acc = None
                try:
                    acc = _safe_float(j["results"]["mbpp"][MBPP_METRIC])
                except Exception:
                    pass
                t = _safe_float(j.get("total_evaluation_time_seconds"))
                cand = Cell(acc=acc, time_s=t)
                data[cfg] = _choose_newer(data.get(cfg), cand)
    return data


# ── Table rendering ──


def _cell_values(
    gsm8k: Optional[Cell], mbpp: Optional[Cell], task: str,
    baseline_g_t: Optional[float], baseline_m_t: Optional[float],
):
    """Return (acc, speedup) for a given task."""
    g_acc = gsm8k.acc if gsm8k else None
    m_acc = mbpp.acc if mbpp else None
    g_t = gsm8k.time_s if gsm8k else None
    m_t = mbpp.time_s if mbpp else None

    if task == "gsm8k":
        return g_acc, _speedup(baseline_g_t, g_t)
    elif task == "mbpp":
        return m_acc, _speedup(baseline_m_t, m_t)
    else:  # avg
        return _mean([g_acc, m_acc]), _speedup(
            _mean([baseline_g_t, baseline_m_t]),
            _mean([g_t, m_t]),
        )


def render_baseline_table(
    gsm8k_data: Dict[str, Cell],
    mbpp_data: Dict[str, Cell],
    baseline_g_t: Optional[float],
    baseline_m_t: Optional[float],
    acc_digits: int = 1,
    speedup_digits: int = 1,
) -> str:
    columns = BASELINE_COLUMNS
    lines = ["### Baseline (cached)", ""]
    lines.append("<table>")
    lines.append("  <thead>")

    # Row 1: threshold settings (grouped)
    lines.append("    <tr>")
    lines.append('      <th rowspan="3">method</th>')
    prev_th = None
    for th_tag, _bl in columns:
        if th_tag != prev_th:
            span = sum(1 for t, _ in columns if t == th_tag)
            lines.append(f'      <th colspan="{span * 3}">{th_tag}</th>')
            prev_th = th_tag
    lines.append("    </tr>")

    # Row 2: block sizes
    lines.append("    <tr>")
    for _th_tag, bl in columns:
        lines.append(f'      <th colspan="3">b{bl}</th>')
    lines.append("    </tr>")

    # Row 3: tasks
    lines.append("    <tr>")
    for _ in columns:
        for task in TASKS:
            lines.append(f"      <th>{task}</th>")
    lines.append("    </tr>")

    lines.append("  </thead>")
    lines.append("  <tbody>")

    # Single data row
    lines.append("    <tr>")
    lines.append("      <td>cached</td>")
    for th_tag, bl in columns:
        cfg = f"cached_b{bl}_{th_tag}"
        g = gsm8k_data.get(cfg)
        m = mbpp_data.get(cfg)
        for task in TASKS:
            acc, spd = _cell_values(g, m, task, baseline_g_t, baseline_m_t)
            acc_s = _fmt_acc(acc, acc_digits)
            spd_s = _fmt_speedup(spd, speedup_digits)
            if acc is None and spd is None:
                lines.append("      <td>-</td>")
            else:
                lines.append(f"      <td>{acc_s} ({spd_s})</td>")
    lines.append("    </tr>")

    lines.append("  </tbody>")
    lines.append("</table>")
    lines.append("")
    return "\n".join(lines)


def render_policy_table(
    policy: str,
    gsm8k_data: Dict[str, Cell],
    mbpp_data: Dict[str, Cell],
    baseline_g_t: Optional[float],
    baseline_m_t: Optional[float],
    acc_digits: int = 1,
    speedup_digits: int = 1,
) -> str:
    columns = SSD_COLUMNS

    # Collect row labels for this policy
    row_labels_set: set = set()
    for cfg in list(gsm8k_data) + list(mbpp_data):
        parsed = _parse_config(cfg)
        if parsed is None:
            continue
        gen_fn, _bl, _th_tag, suffix = parsed
        if gen_fn != "ssd" or suffix is None:
            continue
        if _policy_from_suffix(suffix) == policy:
            row_labels_set.add(_row_label(suffix))

    if not row_labels_set:
        return ""

    row_labels = _sort_policy_rows(policy, list(row_labels_set))

    # Compute best acc/speedup per column+task for bolding
    best_acc: Dict[Tuple, Optional[float]] = {}
    best_spd: Dict[Tuple, Optional[float]] = {}
    for th_tag, bl in columns:
        for task in TASKS:
            accs, spds = [], []
            for rl in row_labels:
                cfg = f"ssd_b{bl}_{th_tag}_policy={rl}"
                g = gsm8k_data.get(cfg)
                m = mbpp_data.get(cfg)
                acc, spd = _cell_values(g, m, task, baseline_g_t, baseline_m_t)
                if acc is not None:
                    accs.append(acc)
                if spd is not None:
                    spds.append(spd)
            best_acc[(th_tag, bl, task)] = max(accs) if accs else None
            best_spd[(th_tag, bl, task)] = max(spds) if spds else None

    lines = [f"### {POLICY_TITLE[policy]}", ""]
    lines.append("<table>")
    lines.append("  <thead>")

    # Header row 1: threshold settings
    lines.append("    <tr>")
    lines.append('      <th rowspan="2">config</th>')
    for th_tag, bl in columns:
        lines.append(f'      <th colspan="3">{th_tag} b{bl}</th>')
    lines.append("    </tr>")

    # Header row 2: tasks
    lines.append("    <tr>")
    for _ in columns:
        for task in TASKS:
            lines.append(f"      <th>{task}</th>")
    lines.append("    </tr>")

    lines.append("  </thead>")
    lines.append("  <tbody>")

    for rl in row_labels:
        lines.append("    <tr>")
        lines.append(f"      <td>{rl}</td>")
        for th_tag, bl in columns:
            cfg = f"ssd_b{bl}_{th_tag}_policy={rl}"
            g = gsm8k_data.get(cfg)
            m = mbpp_data.get(cfg)
            for task in TASKS:
                acc, spd = _cell_values(g, m, task, baseline_g_t, baseline_m_t)
                acc_s = _fmt_acc(acc, acc_digits)
                spd_s = _fmt_speedup(spd, speedup_digits)
                ba = best_acc.get((th_tag, bl, task))
                bs = best_spd.get((th_tag, bl, task))
                if ba is not None and _nearly_equal(acc, ba):
                    acc_s = f"<b>{acc_s}</b>"
                if bs is not None and _nearly_equal(spd, bs):
                    spd_s = f"<b>{spd_s}</b>"
                if acc is None and spd is None:
                    lines.append("      <td>-</td>")
                else:
                    lines.append(f"      <td>{acc_s} ({spd_s})</td>")
        lines.append("    </tr>")

    lines.append("  </tbody>")
    lines.append("</table>")
    lines.append("")
    return "\n".join(lines)


# ── Main ──


def main():
    parser = argparse.ArgumentParser(
        description="Compile LLaDA2.1 SSD-policy sweep results into markdown tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gsm8k_summary", type=str, nargs="+",
                        default=["summary/gsm8k.jsonl"],
                        help="GSM8K summary JSONL file(s)")
    parser.add_argument("--mbpp_result_roots", type=str, nargs="+",
                        default=["results"],
                        help="Root dir(s) containing cached/ and ssd/ MBPP results")
    parser.add_argument("--out", type=str, default="compiled/sweep_results.md",
                        help="Output markdown path")
    parser.add_argument("--model_dir_name", type=str, default=MODEL_DIR_NAME,
                        help="Model directory name in lm_eval output (pretrained with / -> __)")
    parser.add_argument("--acc_digits", type=int, default=1)
    parser.add_argument("--speedup_digits", type=int, default=1)

    args = parser.parse_args()

    gsm8k_data = load_gsm8k([Path(p) for p in args.gsm8k_summary])
    mbpp_data = load_mbpp([Path(p) for p in args.mbpp_result_roots], args.model_dir_name)

    baseline_g = gsm8k_data.get(SPEEDUP_BASELINE_CFG)
    baseline_m = mbpp_data.get(SPEEDUP_BASELINE_CFG)
    baseline_g_t = baseline_g.time_s if baseline_g else None
    baseline_m_t = baseline_m.time_s if baseline_m else None

    md: List[str] = []
    md.append(f"Compiled on {dt.datetime.now().isoformat(timespec='seconds')}")
    md.append("")
    md.append(f"- speedup baseline: `{SPEEDUP_BASELINE_CFG}`")
    md.append(f"- baseline times: gsm8k={baseline_g_t or 'NA'}s, mbpp={baseline_m_t or 'NA'}s")
    md.append("")

    md.append("## Baseline")
    md.append("")
    md.append(render_baseline_table(
        gsm8k_data, mbpp_data, baseline_g_t, baseline_m_t,
        acc_digits=args.acc_digits, speedup_digits=args.speedup_digits,
    ))

    md.append("## SSD Policy Sweeps")
    md.append("")
    for policy in POLICY_ORDER:
        table = render_policy_table(
            policy, gsm8k_data, mbpp_data, baseline_g_t, baseline_m_t,
            acc_digits=args.acc_digits, speedup_digits=args.speedup_digits,
        )
        if table:
            md.append(table)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
