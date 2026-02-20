"""
Minimal LLaDA2.1 inference example.

Usage:
  python example_llada.py
  python example_llada.py --model inclusionAI/LLaDA2.1-mini --prompt "Hello!"
  python example_llada.py --prompt-file path/to/prompt.txt --gen-length 1024
"""

import argparse
import sys
import time
from typing import Optional

import torch

from generate_utils import generate_cached, generate_ssd_policy, load_model_and_tokenizer
from utils import set_seed, str2bool


DEFAULT_PROMPT = (
    "You are an expert Python programmer, and here is your task: Write a python function to remove "
    "first and last occurrence of a given character from the string. Your code should pass these tests:\n\n"
    'assert remove_Occ("hello","l") == "heo"\nassert remove_Occ("abcda","a") == "bcd"\n'
    'assert remove_Occ("PHP","P") == "H"\n'
)


def build_text_input(tokenizer, prompt: str) -> str:
    """
    Prefer chat-template formatting if the tokenizer provides it;
    otherwise fall back to raw prompt.
    """
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return prompt


def read_prompt_from_sources(prompt_arg, prompt_file, default_prompt):
    """Read prompt from CLI arg, file, stdin, or use default."""
    if prompt_arg is not None:
        return prompt_arg
    if prompt_file is not None:
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    if not sys.stdin.isatty():
        data = sys.stdin.read()
        if data:
            return data
    return default_prompt


@torch.no_grad()
def main(args):
    set_seed(args.seed)

    print(f"Loading model from: {args.model}")
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        dtype_str=args.dtype,
        device_map=args.device_map,
    )

    prompt = read_prompt_from_sources(args.prompt, args.prompt_file, DEFAULT_PROMPT)
    text = build_text_input(tokenizer, prompt)

    # Tokenize input
    input_ids = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
    )["input_ids"]
    input_ids = input_ids.to(model.device)

    print(f"\n=== Prompt ===")
    print(prompt)
    print(f"\n=== Generating ({args.gen_length} tokens max) ===")

    # Generate
    t0 = time.perf_counter()
    stats = None

    if args.generate_fn == "cached":
        generated_tokens = generate_cached(
            model=model,
            inputs=input_ids,
            temperature=args.temperature,
            block_length=args.block_length,
            steps=args.steps,
            gen_length=args.gen_length,
            top_p=args.top_p,
            top_k=args.top_k,
            eos_early_stop=args.eos_early_stop,
            minimal_topk=args.minimal_topk,
            threshold=args.threshold,
            editing_threshold=args.editing_threshold,
            max_post_steps=args.max_post_steps,
            eos_id=args.eos_id,
            mask_id=args.mask_id,
            num_to_transfer=args.num_to_transfer,
        )
    elif args.generate_fn == "ssd_policy":
        result = generate_ssd_policy(
            model=model,
            inputs=input_ids,
            temperature=args.temperature,
            block_length=args.block_length,
            steps=args.steps,
            gen_length=args.gen_length,
            top_p=args.top_p,
            top_k=args.top_k,
            eos_early_stop=args.eos_early_stop,
            eos_id=args.eos_id,
            mask_id=args.mask_id,
            threshold=args.threshold,
            editing_threshold=args.editing_threshold,
            min_ssd_span_length=args.min_ssd_span_length,
            ssd_ratio_tempering_factor=args.ssd_ratio_tempering_factor,
            return_forward_stats=args.return_forward_stats,
        )
        if args.return_forward_stats:
            generated_tokens, stats = result
        else:
            generated_tokens = result
    else:
        raise ValueError(f"Unknown generate function: {args.generate_fn}")

    t1 = time.perf_counter()

    # Decode output
    output_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    num_generated = generated_tokens.shape[1]

    print(f"\n=== Output ===")
    print(output_text)
    print(f"\n=== Stats ===")
    print(f"Tokens generated: {num_generated}")
    print(f"Time: {t1 - t0:.2f}s")
    print(f"Speed: {num_generated / (t1 - t0):.1f} tok/s")

    if stats is not None:
        print(f"Total forward steps: {stats['total_forward_steps']}")
        print(f"Decoding steps: {len(stats['decoding_order'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal LLaDA2.1 generation example")
    parser.add_argument("--model", type=str, default="inclusionAI/LLaDA2.1-mini", help="HuggingFace model repo or local path")
    parser.add_argument("-p", "--prompt", type=str, default=None, help="User prompt text. If omitted, read from stdin/default.")
    parser.add_argument("--prompt-file", type=str, default=None, help="Path to a file containing the prompt.")
    parser.add_argument("--dtype", type=str, choices=["bfloat16", "float16", "float32"], default="bfloat16", help="Torch dtype for model.")
    parser.add_argument("--device-map", type=str, default="auto", help="Transformers device_map value.")
    parser.add_argument("--gen-length", type=int, default=512, help="Number of tokens to generate.")
    parser.add_argument("--block-length", type=int, default=32, help="Block length used by model.generate.")
    parser.add_argument("--steps", type=int, default=32, help="Refinement steps per block.")
    parser.add_argument("--top-p", type=float, default=None, help="Optional nucleus sampling threshold.")
    parser.add_argument("--top-k", type=int, default=None, help="Optional top-k sampling cutoff.")
    parser.add_argument("--minimal-topk", type=int, default=1, help="Caps effective steps via gen_length // minimal_topk.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Acceptance threshold for generation.")
    parser.add_argument("--editing-threshold", type=float, default=0.0, help="Editing threshold for generation.")
    parser.add_argument("--max-post-steps", type=int, default=16, help="Post-mask global editing steps per block.")
    parser.add_argument("--eos-id", type=int, default=156892, help="EOS token id for early stopping.")
    parser.add_argument("--mask-id", type=int, default=156895, help="Mask token id used during iterative refinement.")
    parser.add_argument("--num-to-transfer", type=int, default=1, help="Minimum number of masked positions to resolve per iteration.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--eos-early-stop", type=str2bool, default=True, help="Enable/disable early stopping at EOS.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    # Generation function selection
    parser.add_argument("--generate-fn", type=str, choices=["cached", "ssd_policy"], default="cached", help="Generation function to use.")

    # SSD-specific arguments (used when --generate-fn=ssd_policy)
    parser.add_argument("--min-ssd-span-length", type=int, default=1, help="Minimum mask span length to trigger 2L verification.")
    parser.add_argument("--ssd-ratio-tempering-factor", type=float, default=1.0, help="Exponent applied to SSD acceptance ratios.")
    parser.add_argument("--return-forward-stats", action="store_true", help="Return and print forward statistics (SSD only).")

    args = parser.parse_args()
    main(args)
