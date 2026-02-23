# Commands

## Single-run examples

### GSM8K
```bash
python eval_gsm8k_llada.py --sample_n 500 --generate_fn ssd_policy --gen_length 512 --block_length 32 --threshold 0.95 --editing_threshold 0.9 --max_post_steps 0 --do_verify_policy mask_span_length --min_ssd_span_length 8 --summary_file summary/gsm8k --config_str "ssd_b32_tm0.95_te0.9_policy=span_span=8"
```

### MBPP — SSD
```bash
python eval_mbpp_llada.py --sample_n 500 --generate_fn ssd_policy --gen_length 512 --block_length 32 --threshold 0.95 --editing_threshold 0.9 --max_post_steps 0 --do_verify_policy mask_span_length --min_ssd_span_length 8 --summary_file summary/mbpp --config_str "ssd_b32_tm0.95_te0.9_policy=span_span=8"
```

### MBPP — cached baseline
```bash
python eval_mbpp_llada.py --sample_n 500 --generate_fn cached --gen_length 512 --block_length 32 --threshold 0.95 --editing_threshold 0.9 --max_post_steps 0 --summary_file summary/mbpp --config_str "cached_b32_tm0.95_te0.9"
```

## Sweep experiments

### 1. Generate sweep scripts

```bash
python gen_sweep_sh.py \
  --name sweep/run \
  --node_gpu_ids "rh01:2,3,4,5,6,7;rh02:0,1,2,3,4,5,6,7;rh04:0,1,2,3,4,5,6,7"
```

```bash
python gen_sweep_sh.py \
    --name sweep/run \
    --node_gpu_ids "rh01:2,3,4,5,6,7;rh02:0,1,2,3,4,5,6,7;rh04:0,1,2,3,4,5,6,7" \
    --existing_roots res_rh01 res_rh02 res_rh04
```

This generates bash scripts under `runs/sweep/` with one script per GPU:
- `runs/sweep/run_rh01_2.sh`, `runs/sweep/run_rh01_3.sh`, ...
- `runs/sweep/run_rh02_0.sh`, ..., `runs/sweep/run_rh04_7.sh`

68 runs (4 baseline + 64 SSD) × 2 tasks = 136 commands across 22 GPUs.

### 2. Run on each node

On each node, launch the scripts for that node's GPUs in parallel:

```bash
# On rh01:
bash runs/sweep/run_rh01_2.sh &
bash runs/sweep/run_rh01_3.sh &
bash runs/sweep/run_rh01_4.sh &
bash runs/sweep/run_rh01_5.sh &
bash runs/sweep/run_rh01_6.sh &
bash runs/sweep/run_rh01_7.sh &

# On rh02:
bash runs/sweep/run_rh02_0.sh &
bash runs/sweep/run_rh02_1.sh &
bash runs/sweep/run_rh02_2.sh &
bash runs/sweep/run_rh02_3.sh &
bash runs/sweep/run_rh02_4.sh &
bash runs/sweep/run_rh02_5.sh &
bash runs/sweep/run_rh02_6.sh &
bash runs/sweep/run_rh02_7.sh &

# On rh04:
bash runs/sweep/run_rh04_0.sh &
bash runs/sweep/run_rh04_1.sh &
bash runs/sweep/run_rh04_2.sh &
bash runs/sweep/run_rh04_3.sh &
bash runs/sweep/run_rh04_4.sh &
bash runs/sweep/run_rh04_5.sh &
bash runs/sweep/run_rh04_6.sh &
bash runs/sweep/run_rh04_7.sh &
```

Results are written to:
- GSM8K: `summary/gsm8k.jsonl` (all runs append to the same file)
- MBPP: `summary/mbpp.jsonl` (all runs append to the same file)

### 3. Compile results into tables

```bash
python compile_sweep_tables.py \
  --gsm8k_summary summary/gsm8k.jsonl \
  --mbpp_summary summary/mbpp.jsonl \
  --out compiled/sweep_results.md
```

Output: `compiled/sweep_results.md` with HTML tables showing `acc (speedup)` for each config. Speedup is relative to `cached_b1_tm1_te0.9`.

### Experiment grid summary

**Baseline (cached):**
| setting | block sizes |
|---------|-------------|
| tm1_te0.9 (speedup ref) | b1, b32 |
| tm0.95_te0.9 | b32 |
| tm0.7_te0.5 | b32 |

**SSD policies** (each × {tm0.95_te0.9, tm0.7_te0.5} × b32):
| policy | configs |
|--------|---------|
| mask_span_length | span=1, 2, 4, 8, 16, 31 |
| score_threshold | th={-5,-1,0,1,5} × {dyn c1, sta c1/c2/c4} |
| score_hysteresis | (on,off)={(0,-1),(1,-1),(0,-5),(1,-5),(5,-1),(5,-5)} |
