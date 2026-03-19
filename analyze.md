# Decoding Analysis for LLaDA2.1

- **Local AR-ness@k**: fraction of steps where the last k+1 decoded positions form a consecutive increasing sequence (i.e., next-token pattern).
- **Global AR-ness@k**: fraction of steps where the decoded position is among the k earliest remaining (still-masked) positions.

## 1) Collect AR-ness data

Uses `threshold=1.0` with `num_to_transfer=1` (one token decoded per step, no editing), across block lengths.

```bash
# LLaDA2.1-mini, GSM8K (default)
CUDA_VISIBLE_DEVICES=0 python analyze_arness.py \
  --model_dir inclusionAI/LLaDA2.1-mini \
  --output arness_results_llada_gsm8k.json

# LLaDA2.1-mini, MBPP
CUDA_VISIBLE_DEVICES=0 python analyze_arness.py \
  --model_dir inclusionAI/LLaDA2.1-mini \
  --task mbpp \
  --output arness_results_llada_mbpp.json
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--model_dir` | `inclusionAI/LLaDA2.1-mini` | Model path or HuggingFace repo |
| `--dtype` | `bfloat16` | Model dtype: `bfloat16`, `float16`, `float32` |
| `--mask_id` | `156895` | Mask token ID |
| `--task` | `gsm8k` | Dataset to use: `gsm8k`, `mbpp` |
| `--block_lengths` | `4 8 16 32 64` | Block lengths to sweep |
| `--gen_length` | `256` | Number of tokens to generate |
| `--max_k` | `4` | Compute AR-ness @k for k=1..max_k |
| `--n_samples` | `100` | Number of samples |
| `--seed` | `42` | Random seed |
| `--verbose` | `false` | Print per-sample details |

## 2) Plot AR-ness

```bash
python plot_arness.py \
  --input arness_results_llada_gsm8k.json \
  --output_prefix arness_llada_gsm8k \
  --format pdf
```

Produces:
- `arness_llada_gsm8k_local.pdf` -- Local AR-ness vs block length (4 curves: @k=1,2,3,4)
- `arness_llada_gsm8k_global.pdf` -- Global AR-ness vs block length (4 curves: @k=1,2,3,4)

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--format` | `pdf` | Output format: pdf, png, svg |
| `--max_k` | from data | Override number of k curves to plot |

---

## Confidence Analysis

Analyze per-step decoding confidence across block lengths, for both static and dynamic strategies.

### 3) Collect confidence data

Two strategies:
- **Static** (`threshold=1.0`): 1 token/step via `num_to_transfer=1` fallback
- **Dynamic** (`threshold=0.9`): all tokens above threshold unmasked at once

Both use `editing_threshold=1.0`, `max_post_steps=0`, `num_to_transfer=1`.

```bash
# LLaDA2.1-mini, GSM8K (default)
CUDA_VISIBLE_DEVICES=0 python analyze_confidence.py \
  --model_dir inclusionAI/LLaDA2.1-mini \
  --output confidence_results_llada_gsm8k.json

# LLaDA2.1-mini, MBPP
CUDA_VISIBLE_DEVICES=0 python analyze_confidence.py \
  --model_dir inclusionAI/LLaDA2.1-mini \
  --task mbpp \
  --output confidence_results_llada_mbpp.json

# Custom dynamic threshold
CUDA_VISIBLE_DEVICES=0 python analyze_confidence.py \
  --model_dir inclusionAI/LLaDA2.1-mini \
  --confidence_threshold 0.85 \
  --output confidence_results_llada_gsm8k_t85.json
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--model_dir` | `inclusionAI/LLaDA2.1-mini` | Model path or HuggingFace repo |
| `--dtype` | `bfloat16` | Model dtype: `bfloat16`, `float16`, `float32` |
| `--mask_id` | `156895` | Mask token ID |
| `--task` | `gsm8k` | Dataset: `gsm8k`, `mbpp` |
| `--block_lengths` | `4 8 16 32` | Block lengths to sweep |
| `--gen_length` | `256` | Number of tokens to generate |
| `--confidence_threshold` | `0.9` | Threshold for dynamic strategy |
| `--n_samples` | `100` | Number of samples |
| `--seed` | `42` | Random seed |

### 4) Plot confidence

```bash
python plot_confidence.py \
  --input confidence_results_llada_gsm8k.json \
  --output_prefix confidence_llada_gsm8k \
  --format pdf
```

Produces 4 figures (curves for B=4,8,16,32):
- `confidence_llada_gsm8k_static_step.pdf` -- Static, x = step index
- `confidence_llada_gsm8k_static_normalized.pdf` -- Static, x = normalized progress
- `confidence_llada_gsm8k_dynamic_step.pdf` -- Dynamic, x = step index
- `confidence_llada_gsm8k_dynamic_normalized.pdf` -- Dynamic, x = normalized progress

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--format` | `pdf` | Output format: pdf, png, svg |
| `--n_bins` | `100` | Bins for normalized progress plots |
| `--smooth` | `0` | Smoothing window for step-based plots (0 = off) |
