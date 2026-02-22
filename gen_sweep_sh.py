#!/usr/bin/env python3
"""
Generate per-node per-GPU bash scripts for LLaDA2.1 SSD-policy sweeps.

Experiment grid:
- Baseline (cached):
  - tm1_te0.9: b1, b32  (speedup reference = b1)
  - tm0.95_te0.9: b32
  - tm0.7_te0.5: b32
- SSD policies x {tm0.95_te0.9, tm0.7_te0.5} x b32:
  - mask_span_length: span = 1, 2, 4, 8, 16, 31
  - score_threshold: th x {dyn c1, sta c1/c2/c4}
  - score_hysteresis: (on, off) pairs

Each run produces two commands: GSM8K eval + MBPP eval (lm_eval).
"""

from __future__ import annotations

import argparse
import json
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

MODEL = "inclusionAI/LLaDA2.1-mini"
CUSTOM_MODEL_CLASS = "./modeling_llada2_moe_cache.py:LLaDA2MoeModelLM"
BLOCK_LENGTH = 32


def _fmt_num(v: float) -> str:
    if float(v).is_integer():
        return str(int(v))
    return str(v)


def _th_tag(threshold: float, editing_threshold: float) -> str:
    return f"tm{_fmt_num(threshold)}_te{_fmt_num(editing_threshold)}"


@dataclass(frozen=True)
class RunConfig:
    generate_fn: str  # "cached" or "ssd_policy"
    block_length: int
    threshold: float
    editing_threshold: float
    max_post_steps: int
    policy_cli_args: Tuple[str, ...]
    policy_model_args: Tuple[str, ...]
    config_str: str


# ── Build experiment grid ──


def _mask_span_values(block_length: int) -> List[int]:
    """Powers of 2 from 1 up to block_length, plus block_length-1 if not already."""
    vals: List[int] = []
    x = 1
    while x < block_length:
        vals.append(x)
        x *= 2
    tail = block_length - 1
    if tail >= 1 and tail not in vals:
        vals.append(tail)
    return vals


def build_runs() -> List[RunConfig]:
    runs: List[RunConfig] = []

    # ── Baseline (cached) ──
    # tm1_te0.9 at b1 and b32
    for bl in [1, BLOCK_LENGTH]:
        runs.append(RunConfig(
            generate_fn="cached",
            block_length=bl,
            threshold=1.0, editing_threshold=0.9,
            max_post_steps=0,
            policy_cli_args=(), policy_model_args=(),
            config_str=f"cached_b{bl}_{_th_tag(1.0, 0.9)}",
        ))
    # tm0.95_te0.9 and tm0.7_te0.5 at b32
    for th, eth in [(0.95, 0.9), (0.7, 0.5)]:
        runs.append(RunConfig(
            generate_fn="cached",
            block_length=BLOCK_LENGTH,
            threshold=th, editing_threshold=eth,
            max_post_steps=0,
            policy_cli_args=(), policy_model_args=(),
            config_str=f"cached_b{BLOCK_LENGTH}_{_th_tag(th, eth)}",
        ))

    # ── SSD Policies ──
    for th, eth in [(0.95, 0.9), (0.7, 0.5)]:
        tag = _th_tag(th, eth)

        # mask_span_length
        for span in _mask_span_values(BLOCK_LENGTH):
            ptag = f"policy=span_span={span}"
            runs.append(RunConfig(
                generate_fn="ssd_policy",
                block_length=BLOCK_LENGTH,
                threshold=th, editing_threshold=eth,
                max_post_steps=0,
                policy_cli_args=(
                    "--do_verify_policy", "mask_span_length",
                    "--min_ssd_span_length", str(span),
                ),
                policy_model_args=(
                    "do_verify_policy=mask_span_length",
                    f"min_ssd_span_length={span}",
                ),
                config_str=f"ssd_b{BLOCK_LENGTH}_{tag}_{ptag}",
            ))

        # score_threshold
        for sth in [-5, -1, 0, 1, 5]:
            for score_full, score_tag, coef in [
                ("difference_dynamic", "dyn", 1.0),
                ("difference_static", "sta", 1.0),
                ("difference_static", "sta", 2.0),
                ("difference_static", "sta", 4.0),
            ]:
                ptag = f"policy=score_th={sth}_est=entropy_score={score_tag}_c={_fmt_num(coef)}"
                runs.append(RunConfig(
                    generate_fn="ssd_policy",
                    block_length=BLOCK_LENGTH,
                    threshold=th, editing_threshold=eth,
                    max_post_steps=0,
                    policy_cli_args=(
                        "--do_verify_policy", "score_threshold",
                        "--do_verify_score_threshold", _fmt_num(float(sth)),
                        "--token_acceptance_estimator", "soft_entropy_negexp",
                        "--do_verify_score_type", score_full,
                        "--score_penalty_coef", _fmt_num(coef),
                    ),
                    policy_model_args=(
                        "do_verify_policy=score_threshold",
                        f"do_verify_score_threshold={_fmt_num(float(sth))}",
                        "token_acceptance_estimator=soft_entropy_negexp",
                        f"do_verify_score_type={score_full}",
                        f"score_penalty_coef={_fmt_num(coef)}",
                    ),
                    config_str=f"ssd_b{BLOCK_LENGTH}_{tag}_{ptag}",
                ))

        # score_hysteresis
        for h_on, h_off in [(0, -1), (1, -1), (0, -5), (1, -5), (5, -1), (5, -5)]:
            ptag = f"policy=hysteresis_on={h_on}_off={h_off}_est=entropy_score=dyn_c=1"
            runs.append(RunConfig(
                generate_fn="ssd_policy",
                block_length=BLOCK_LENGTH,
                threshold=th, editing_threshold=eth,
                max_post_steps=0,
                policy_cli_args=(
                    "--do_verify_policy", "score_hysteresis",
                    "--hysteresis_threshold_on", _fmt_num(float(h_on)),
                    "--hysteresis_threshold_off", _fmt_num(float(h_off)),
                    "--token_acceptance_estimator", "soft_entropy_negexp",
                    "--do_verify_score_type", "difference_dynamic",
                    "--score_penalty_coef", "1",
                ),
                policy_model_args=(
                    "do_verify_policy=score_hysteresis",
                    f"hysteresis_threshold_on={_fmt_num(float(h_on))}",
                    f"hysteresis_threshold_off={_fmt_num(float(h_off))}",
                    "token_acceptance_estimator=soft_entropy_negexp",
                    "do_verify_score_type=difference_dynamic",
                    "score_penalty_coef=1",
                ),
                config_str=f"ssd_b{BLOCK_LENGTH}_{tag}_{ptag}",
            ))

    return runs


# ── Command builders ──


def build_gsm8k_cmd(
    gpu_id: str,
    run: RunConfig,
    python_bin: str,
    eval_script: str,
    sample_n: int,
    gen_length: int,
    summary_file: str,
) -> str:
    parts = [
        f"CUDA_VISIBLE_DEVICES={gpu_id}",
        python_bin, eval_script,
        "--sample_n", str(sample_n),
        "--generate_fn", run.generate_fn,
        "--gen_length", str(gen_length),
        "--block_length", str(run.block_length),
        "--threshold", _fmt_num(run.threshold),
        "--editing_threshold", _fmt_num(run.editing_threshold),
        "--max_post_steps", str(run.max_post_steps),
    ]
    parts.extend(run.policy_cli_args)
    parts.extend([
        "--summary_file", summary_file,
        "--config_str", f'"{run.config_str}"',
    ])
    return " ".join(parts)


def build_mbpp_cmd(
    gpu_id: str,
    run: RunConfig,
    model: str,
    custom_model_class: str,
    max_gen_toks: int,
    output_root: str,
) -> str:
    fn_name = {"cached": "generate_cached", "ssd_policy": "generate_ssd_policy"}[run.generate_fn]
    custom_generate = f"./generate_utils.py:{fn_name}"
    model_args = [
        f"pretrained={model}",
        "trust_remote_code=True",
        f"custom_model_class={custom_model_class}",
        f"custom_generate={custom_generate}",
        f"block_length={run.block_length}",
        f"threshold={_fmt_num(run.threshold)}",
        f"editing_threshold={_fmt_num(run.editing_threshold)}",
        f"max_post_steps={run.max_post_steps}",
        f"max_gen_toks={max_gen_toks}",
        "return_stats=false",
    ]
    model_args.extend(run.policy_model_args)

    output_subdir = "cached" if run.generate_fn == "cached" else "ssd"
    parts = [
        f"CUDA_VISIBLE_DEVICES={gpu_id}",
        "HF_ALLOW_CODE_EVAL=1",
        "lm_eval",
        "--model", "hf",
        "--model_args", ",".join(model_args),
        "--batch_size", "1",
        "--tasks", "mbpp",
        "--confirm_run_unsafe_code",
        "--log_samples",
        "--output_path", f"{output_root}/{output_subdir}/{run.config_str}",
    ]
    return " ".join(parts)


# ── Node/GPU parsing and script writing ──


def parse_node_gpu_ids(s: str) -> List[Tuple[str, str]]:
    """Parse 'rh01:2,3;rh02:0,1' into [('rh01','2'), ('rh01','3'), ...]."""
    slots: List[Tuple[str, str]] = []
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Invalid node_gpu_ids chunk: {chunk}")
        node, gids = chunk.split(":", 1)
        for gid in gids.split(","):
            gid = gid.strip()
            if gid:
                slots.append((node.strip(), gid))
    if not slots:
        raise ValueError("No GPU slots parsed")
    return slots


def write_sh(path: Path, cmds: Sequence[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -euo pipefail\n\n")
        f.write('echo "Running on $(hostname)"; date\n\n')
        for i, (cmd, label) in enumerate(cmds, 1):
            f.write(f'echo "===== [{i}/{len(cmds)}] {label} ====="\n')
            f.write(cmd + "\n\n")
        f.write('echo "Done."; date\n')
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


# ── Scan existing results ──

MODEL_DIR_NAME = "inclusionAI__LLaDA2.1-mini"


def scan_existing_gsm8k(roots: Sequence[str], summary_file: str) -> Set[str]:
    """Scan directories for existing GSM8K configs in summary JSONL files."""
    configs: Set[str] = set()
    for root in roots:
        for suffix in [".jsonl", "_ssd.jsonl", "_cached.jsonl"]:
            p = Path(root) / (summary_file + suffix)
            if not p.exists():
                continue
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        cfg = json.loads(line).get("config", "")
                        if cfg:
                            configs.add(cfg)
                    except Exception:
                        continue
    return configs


def scan_existing_mbpp(roots: Sequence[str], output_root: str, model_dir_name: str = MODEL_DIR_NAME) -> Set[str]:
    """Scan directories for existing MBPP configs (has results_*.json)."""
    configs: Set[str] = set()
    for root in roots:
        for subdir in ["cached", "ssd"]:
            d = Path(root) / output_root / subdir
            if not d.is_dir():
                continue
            for cfg_dir in d.iterdir():
                if not cfg_dir.is_dir():
                    continue
                model_dir = cfg_dir / model_dir_name
                if model_dir.is_dir() and list(model_dir.glob("results_*.json")):
                    configs.add(cfg_dir.name)
    return configs


# ── Main ──


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-node per-GPU sweep scripts for LLaDA2.1 SSD-policy experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--name", type=str, required=True,
                        help="Script prefix: {out_dir}/{name}_{node}_{gpu}.sh")
    parser.add_argument("--node_gpu_ids", type=str, required=True,
                        help='Node/GPU mapping, e.g. "rh01:2,3,4,5,6,7;rh02:0,1,2,3,4,5,6,7"')
    parser.add_argument("--out_dir", type=str, default="runs",
                        help="Output directory for generated scripts")

    # GSM8K settings
    parser.add_argument("--python_bin", type=str, default="python")
    parser.add_argument("--eval_script", type=str, default="eval_gsm8k_llada.py")
    parser.add_argument("--gsm8k_sample_n", type=int, default=500)
    parser.add_argument("--gsm8k_gen_length", type=int, default=512)
    parser.add_argument("--gsm8k_summary_file", type=str, default="summary/gsm8k")

    # MBPP settings
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--custom_model_class", type=str, default=CUSTOM_MODEL_CLASS)
    parser.add_argument("--mbpp_max_gen_toks", type=int, default=512)
    parser.add_argument("--mbpp_output_root", type=str, default="results")

    # Skip existing results
    parser.add_argument("--existing_roots", type=str, nargs="*", default=None,
                        help="Directories with existing results to skip (scans summary JSONL and MBPP result dirs)")
    parser.add_argument("--rerun_mbpp", action="store_true",
                        help="Always generate MBPP commands even if results exist")

    args = parser.parse_args()
    slots = parse_node_gpu_ids(args.node_gpu_ids)
    runs = build_runs()

    # Scan existing results
    existing_gsm8k: Set[str] = set()
    existing_mbpp: Set[str] = set()
    if args.existing_roots:
        existing_gsm8k = scan_existing_gsm8k(args.existing_roots, args.gsm8k_summary_file)
        if not args.rerun_mbpp:
            existing_mbpp = scan_existing_mbpp(args.existing_roots, args.mbpp_output_root)
        if existing_gsm8k:
            print(f"Found {len(existing_gsm8k)} existing GSM8K configs (will skip)")
        if existing_mbpp:
            print(f"Found {len(existing_mbpp)} existing MBPP configs (will skip)")

    # Build flat command list, skipping existing
    # Each entry is (command_str, label_for_echo)
    all_cmds: List[Tuple[str, str]] = []
    skipped_gsm8k = 0
    skipped_mbpp = 0
    for run in runs:
        if run.config_str in existing_gsm8k:
            skipped_gsm8k += 1
        else:
            all_cmds.append((
                build_gsm8k_cmd(
                    gpu_id="__GPU__", run=run,
                    python_bin=args.python_bin, eval_script=args.eval_script,
                    sample_n=args.gsm8k_sample_n, gen_length=args.gsm8k_gen_length,
                    summary_file=args.gsm8k_summary_file,
                ),
                f"gsm8k {run.config_str}",
            ))
        if run.config_str in existing_mbpp:
            skipped_mbpp += 1
        else:
            all_cmds.append((
                build_mbpp_cmd(
                    gpu_id="__GPU__", run=run,
                    model=args.model, custom_model_class=args.custom_model_class,
                    max_gen_toks=args.mbpp_max_gen_toks, output_root=args.mbpp_output_root,
                ),
                f"mbpp {run.config_str}",
            ))

    # Distribute commands round-robin across GPU slots
    per_slot: Dict[Tuple[str, str], List[Tuple[str, str]]] = {s: [] for s in slots}
    for j, (cmd, label) in enumerate(all_cmds):
        slot = slots[j % len(slots)]
        _, gpu_id = slot
        per_slot[slot].append((cmd.replace("__GPU__", gpu_id), label))

    out_dir = Path(args.out_dir)
    for (node, gpu_id), cmds in per_slot.items():
        if not cmds:
            continue
        write_sh(out_dir / f"{args.name}_{node}_{gpu_id}.sh", cmds)

    total_cmds = len(all_cmds)
    print(f"Total runs: {len(runs)}, skipped GSM8K: {skipped_gsm8k}, skipped MBPP: {skipped_mbpp}")
    print(f"Generated {total_cmds} commands across {len(slots)} GPU slots")
    print(f"Output dir: {out_dir}")
    for (node, gpu_id), cmds in per_slot.items():
        if cmds:
            print(f"  {args.name}_{node}_{gpu_id}.sh: {len(cmds)} commands")


if __name__ == "__main__":
    main()
