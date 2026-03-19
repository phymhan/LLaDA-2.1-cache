#!/usr/bin/env python3
"""
Analyze per-step decoding confidence of LLaDA2.1 across block lengths.

For each block_length in [4, 8, 16, 32]:
  - Static:  threshold=1.0 (1 token/step via num_to_transfer=1 fallback)
  - Dynamic: threshold=confidence_threshold (all tokens above threshold at once)
  - Both use: editing_threshold=1.0, max_post_steps=0, num_to_transfer=1

Records per-step confidence values and decoding orders.
Saves raw per-sample data to JSON for flexible plotting.
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
from generate_analysis import generate_cached, load_model_and_tokenizer


SYSTEM_PROMPT_GSM8K = "Solve the following math problem concisely and clearly and put your final answer within \\boxed{}."

SYSTEM_PROMPT_MBPP = (
    "You are an expert Python programmer. "
    "Write only the function code, no explanations."
)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze per-step decoding confidence of LLaDA2.1 across block lengths",
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
    parser.add_argument("--confidence_threshold", type=float, default=0.9,
                        help="Confidence threshold for dynamic strategy")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt_length", type=int, default=4096)

    parser.add_argument("--output", type=str, default="confidence_results.json")
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

    # Run analysis: two strategies
    strategies = {
        "static": 1.0,                      # threshold=1.0 → always falls back to topk(1)
        "dynamic": args.confidence_threshold,  # threshold=0.9 → unmask all above 0.9
    }
    results = {}

    for bl in args.block_lengths:
        results[str(bl)] = {}
        for strategy_name, thresh in strategies.items():
            print(f"\n=== B={bl}, {strategy_name} (threshold={thresh}) ===")
            all_samples = []
            all_orders = []

            pbar = tqdm(range(len(prompts)), desc=f"B={bl} {strategy_name}")
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
                    threshold=thresh,
                    editing_threshold=1.0,
                    max_post_steps=0,
                    num_to_transfer=1,
                    record_decoding_order=True,
                )

                # Filter to generation-only positions
                conf_hist_raw = stats["confidence_history"]
                dec_order_raw = stats["decoding_order"]

                conf_filtered = []
                order_filtered = []
                for step_confs, step_order in zip(conf_hist_raw, dec_order_raw):
                    step_c = []
                    step_o = []
                    for pos, conf in zip(step_order, step_confs):
                        rel = pos - prompt_len
                        if 0 <= rel < args.gen_length:
                            step_c.append(conf)
                            step_o.append(rel)
                    if step_c:
                        conf_filtered.append(step_c)
                        order_filtered.append(step_o)

                all_samples.append(conf_filtered)
                all_orders.append(order_filtered)

                if args.verbose and i < 3:
                    n_steps = len(conf_filtered)
                    n_toks = sum(len(s) for s in conf_filtered)
                    mean_conf = np.mean([c for s in conf_filtered for c in s]) if n_toks > 0 else 0
                    print(f"  sample {i}: {n_steps} steps, {n_toks} tokens, mean_conf={mean_conf:.4f}")

            # Print summary
            all_n_steps = [len(s) for s in all_samples]
            all_mean_conf = [np.mean([c for step in s for c in step]) for s in all_samples if s]
            if all_n_steps:
                print(f"  steps: mean={np.mean(all_n_steps):.1f}, "
                      f"min={min(all_n_steps)}, max={max(all_n_steps)}")
            if all_mean_conf:
                print(f"  overall mean confidence: {np.mean(all_mean_conf):.4f}")

            results[str(bl)][strategy_name] = {
                "confidences": all_samples,
                "decoding_orders": all_orders,
            }

    # Save
    output = {
        "task": args.task,
        "block_lengths": args.block_lengths,
        "gen_length": args.gen_length,
        "confidence_threshold": args.confidence_threshold,
        "n_samples": args.n_samples,
        "model_dir": args.model_dir,
        "seed": args.seed,
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
