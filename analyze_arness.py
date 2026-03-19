#!/usr/bin/env python3
"""
Analyze decoding AR-ness of LLaDA2.1 across block lengths.

For each block_length in [4, 8, 16, 32, 64]:
  - threshold=1.0, editing_threshold=1.0, max_post_steps=0, num_to_transfer=1
    (forces exactly 1 token per step via topk fallback)
  - Generate n_samples GSM8K/MBPP samples with record_decoding_order=True
  - Compute local and global AR-ness @k for k=1..max_k
  - Average across samples

Saves results to JSON (same format as references/analyze_arness.py).
"""

import warnings

warnings.filterwarnings("ignore")

import argparse
import json
import random

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from utils import set_seed, str2bool
from utils_arness import local_ar_ness, global_ar_ness
from generate_analysis import generate_cached, load_model_and_tokenizer


SYSTEM_PROMPT_GSM8K = "Solve the following math problem concisely and clearly and put your final answer within \\boxed{}."

SYSTEM_PROMPT_MBPP = (
    "You are an expert Python programmer. "
    "Write only the function code, no explanations."
)


def extract_decoding_order(decoding_order, prompt_length, gen_length):
    """
    Extract flat list of decoded positions (relative to generation start).

    decoding_order: list of lists of ints (absolute positions per step),
                    as produced by generate_analysis.generate_cached.
    Returns: list of ints in [0, gen_length), in decode order.
    """
    positions = []
    seen = set()
    for step in decoding_order:
        for pos in step:
            rel = pos - prompt_length
            if 0 <= rel < gen_length and rel not in seen:
                positions.append(rel)
                seen.add(rel)
    return positions


def main():
    parser = argparse.ArgumentParser(
        description="Analyze decoding AR-ness of LLaDA2.1 across block lengths",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_dir", type=str, default="inclusionAI/LLaDA2.1-mini")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--cuda_visible_devices", "--cvd", type=str, default=None)
    parser.add_argument("--mask_id", type=int, default=156895)

    parser.add_argument("--task", type=str, default="gsm8k", choices=["gsm8k", "mbpp"])
    parser.add_argument("--block_lengths", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--max_k", type=int, default=4)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt_length", type=int, default=4096)

    parser.add_argument("--output", type=str, default="arness_results.json")
    parser.add_argument("--verbose", type=str2bool, default=False)
    args = parser.parse_args()

    import os
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    set_seed(args.seed)

    # Load model
    print(f"\nLoading model: {args.model_dir}")
    model, tokenizer = load_model_and_tokenizer(args.model_dir, dtype_str=args.dtype, device_map=args.device_map)

    # Load dataset and sample
    if args.task == "gsm8k":
        print("Loading GSM8K...")
        dataset = load_dataset("gsm8k", "main", split="test")
    elif args.task == "mbpp":
        print("Loading MBPP...")
        dataset = load_dataset("google-research-datasets/mbpp", "full", split="test")

    rng = random.Random(args.seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    indices = sorted(indices[:args.n_samples])
    dataset = dataset.select(indices)
    print(f"Using {len(dataset)} {args.task} samples")

    # Prepare prompts
    prompts = []
    for ex in dataset:
        if args.task == "gsm8k":
            user_content = f"Question: {ex['question']}"
            system_prompt = SYSTEM_PROMPT_GSM8K
        elif args.task == "mbpp":
            tests_str = "\n".join(ex["test_list"][:3])
            user_content = f"{ex['text']}\nYour code should pass these tests:\n{tests_str}"
            system_prompt = SYSTEM_PROMPT_MBPP

        prompt_text = f"{system_prompt}\n\n{user_content}"
        messages = [{"role": "user", "content": prompt_text}]
        formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        tokens = tokenizer(formatted, return_tensors="pt", truncation=True,
                           add_special_tokens=False, max_length=args.prompt_length)
        prompts.append(tokens)

    # Run analysis
    results = {}
    for bl in args.block_lengths:
        print(f"\n=== block_length={bl} (static, 1 token/step) ===")
        local_scores = {k: [] for k in range(1, args.max_k + 1)}
        global_scores = {k: [] for k in range(1, args.max_k + 1)}

        pbar = tqdm(range(len(prompts)), desc=f"B={bl}")
        for i in pbar:
            set_seed(args.seed + i)
            input_ids = prompts[i]["input_ids"].to(model.device)
            prompt_len = int(input_ids.shape[1])

            out_ids, stats = generate_cached(
                model,
                input_ids=input_ids,
                mask_id=args.mask_id,
                gen_length=args.gen_length,
                block_length=bl,
                threshold=1.0,
                editing_threshold=1.0,
                max_post_steps=0,
                num_to_transfer=1,
                record_decoding_order=True,
            )

            order = extract_decoding_order(stats["decoding_order"], prompt_len, args.gen_length)

            if len(order) < 2:
                continue

            loc = local_ar_ness(order, args.max_k)
            glo = global_ar_ness(order, args.max_k)

            for k in range(1, args.max_k + 1):
                local_scores[k].append(loc[k])
                global_scores[k].append(glo[k])

            if args.verbose and i < 3:
                print(f"  sample {i}: order[:20]={order[:20]}, local={loc}, global={glo}")

        # Average
        local_mean = {k: float(np.mean(local_scores[k])) if local_scores[k] else 0.0
                      for k in range(1, args.max_k + 1)}
        global_mean = {k: float(np.mean(global_scores[k])) if global_scores[k] else 0.0
                       for k in range(1, args.max_k + 1)}
        local_std = {k: float(np.std(local_scores[k])) if local_scores[k] else 0.0
                     for k in range(1, args.max_k + 1)}
        global_std = {k: float(np.std(global_scores[k])) if global_scores[k] else 0.0
                      for k in range(1, args.max_k + 1)}

        results[str(bl)] = {
            "local_mean": {str(k): local_mean[k] for k in range(1, args.max_k + 1)},
            "local_std": {str(k): local_std[k] for k in range(1, args.max_k + 1)},
            "global_mean": {str(k): global_mean[k] for k in range(1, args.max_k + 1)},
            "global_std": {str(k): global_std[k] for k in range(1, args.max_k + 1)},
            "n_valid": len(local_scores[1]),
        }

        print(f"  local  mean: {local_mean}")
        print(f"  global mean: {global_mean}")

    # Save
    output = {
        "task": args.task,
        "block_lengths": args.block_lengths,
        "max_k": args.max_k,
        "n_samples": args.n_samples,
        "gen_length": args.gen_length,
        "model_dir": args.model_dir,
        "seed": args.seed,
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
