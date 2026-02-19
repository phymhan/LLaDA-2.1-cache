import argparse
import os
import sys
import time

import torch
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils import set_seed, str2bool

from configuration_llada2_moe import LLaDA2MoeConfig
from modeling_llada2_moe_cache import LLaDA2MoeModelLM


# DEFAULT_PROMPT = "Calculate 1+5-28*0.5-200=?"
DEFAULT_PROMPT = (
    "You are an expert Python programmer, and here is your task: Write a python function to remove "
    "first and last occurrence of a given character from the string. Your code should pass these tests:\n\n"
    'assert remove_Occ("hello","l") == "heo"\nassert remove_Occ("abcda","a") == "bcd"\n'
    'assert remove_Occ("PHP","P") == "H"\n'
)


def parse_args():
    parser = argparse.ArgumentParser(description="LLaDA 2.1 generation CLI (with KV cache)")
    parser.add_argument("-p", "--prompt", type=str, default=None, help="User prompt text. If omitted, read from stdin/default.")
    parser.add_argument("--prompt-file", type=str, default=None, help="Path to a file containing the prompt.")
    parser.add_argument("--model-path", type=str, default="inclusionAI/LLaDA2.1-mini", help="HF model repo or local path.")
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
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--cuda-visible-devices", "--cvd", type=str, default=None, help="Set CUDA_VISIBLE_DEVICES before loading model.")
    return parser.parse_args()


def read_prompt_from_sources(prompt_arg, prompt_file, default_prompt):
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


def load_model_and_tokenizer(model_path, dtype_str, device_map):
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[dtype_str]

    config = LLaDA2MoeConfig.from_pretrained(model_path)
    model = LLaDA2MoeModelLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer


@torch.no_grad()
def generate_cached(
    model,
    inputs,
    temperature=0.0,
    block_length=32,
    steps=32,
    gen_length=2048,
    top_p=None,
    top_k=None,
    eos_early_stop=False,
    minimal_topk=1,
    threshold=0.95,
    editing_threshold=0.9,
    max_post_steps=16,
    eos_id=156892,
    mask_id=156895,
    num_to_transfer=1,
):
    steps = min(steps, gen_length // minimal_topk)
    input_ids = inputs.to(model.device)

    prompt_length = input_ids.shape[1]
    num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length

    # Block-causal attention mask (for prefill stage only)
    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=model.device))
    block_diffusion_attention_mask = (
        block_mask.repeat_interleave(block_length, dim=0)
        .repeat_interleave(block_length, dim=1)
        .unsqueeze(0)
        .unsqueeze(0)
    ).to(model.dtype)

    position_ids = torch.arange(total_length, device=model.device).unsqueeze(0)
    x = torch.full((1, total_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :prompt_length] = input_ids.clone()

    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length

    past_key_values = DynamicCache()

    # ── Prefill stage: process all complete prompt blocks and store KV ──
    if prefill_length > 0:
        cur_x = x[:, :prefill_length]
        cur_attn_mask = block_diffusion_attention_mask[
            :, :, :prefill_length, :prefill_length
        ]
        cur_position_ids = position_ids[:, :prefill_length]
        model(
            cur_x,
            attention_mask=cur_attn_mask,
            position_ids=cur_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            store_kv=True,
        )

    # ── Decode stage: process each block with KV cache ──
    for num_block in range(prefill_blocks, num_blocks):
        block_start_pos = num_block * block_length
        block_end_pos = (num_block + 1) * block_length

        cur_x = x[:, block_start_pos:block_end_pos].clone()
        cur_position_ids = position_ids[:, block_start_pos:block_end_pos]

        # Prompt tokens within this block (if any)
        prompt_mask_in_block = torch.zeros(
            block_length, dtype=torch.bool, device=model.device
        )
        if block_start_pos < prompt_length:
            prompt_end_in_block = min(prompt_length - block_start_pos, block_length)
            prompt_mask_in_block[:prompt_end_in_block] = True

        post_steps = 0
        while True:
            old_block_tokens = cur_x.clone()
            active_block_mask = cur_x == mask_id
            if not torch.any(active_block_mask):
                post_steps += 1
            if post_steps > max_post_steps:
                break

            # Forward with cache (don't store KV during denoising iterations)
            # attention_mask=None means full attention to all cached + current tokens
            outputs = model(
                cur_x,
                attention_mask=None,
                position_ids=cur_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                store_kv=False,
            )
            logits = outputs.logits  # (1, block_length, vocab_size)

            x0, x0_p = model._sample_with_temperature_topk_topp(
                logits, temperature=temperature, top_k=top_k, top_p=top_p
            )

            # ── Mask-filling: select which masked positions to unmask ──
            mask_transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            if active_block_mask.sum() > 0:
                mask_confidence = torch.where(active_block_mask, x0_p, -torch.inf)
                high_conf_mask = (
                    mask_confidence[0] > threshold
                ) & active_block_mask[0]
                num_high_confidence = high_conf_mask.sum().item()

                if num_high_confidence >= num_to_transfer:
                    mask_transfer_index[0] = high_conf_mask
                else:
                    num_available = active_block_mask.sum().item()
                    if num_available > 0:
                        _, idx = torch.topk(
                            mask_confidence[0],
                            k=min(num_to_transfer, num_available),
                        )
                        mask_transfer_index[0, idx] = True

            # ── Editing: optionally replace non-mask, non-prompt tokens ──
            editing_transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            non_mask_positions = ~active_block_mask
            non_prompt_positions = ~prompt_mask_in_block
            editable_positions = non_mask_positions & non_prompt_positions[None, :]
            editing_confidence = torch.where(editable_positions, x0_p, -torch.inf)
            high_conf_editing = (
                editing_confidence[0] > editing_threshold
            ) & editable_positions[0]

            token_changed = x0[0] != old_block_tokens[0]
            editing_transfer_index[0] = high_conf_editing & token_changed
            final_transfer_index = mask_transfer_index | editing_transfer_index

            if final_transfer_index.any():
                cur_x[final_transfer_index] = x0[final_transfer_index]

            if active_block_mask.sum() == 0 and not editing_transfer_index.any():
                break

        x[:, block_start_pos:block_end_pos] = cur_x

        # Check if we should stop before committing KV (commit is only needed
        # if there are subsequent blocks that will read from the cache).
        if eos_early_stop:
            generated_part = x[0, prompt_length:block_end_pos]
            if (generated_part == mask_id).sum() == 0:
                eos_positions = (generated_part == eos_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    break

        if num_block < num_blocks - 1:
            # ── Commit finalized block: store KV for this block in cache ──
            model(
                cur_x,
                attention_mask=None,
                position_ids=cur_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                store_kv=True,
            )

    # ── Extract generated tokens ──
    generated_answer = x[:, : prompt_length + gen_length]
    eos_positions = (generated_answer[0][input_ids.shape[1] :] == eos_id).nonzero(
        as_tuple=True
    )[0]
    if len(eos_positions) > 0:
        first_eos_position = eos_positions[0].item()
    else:
        first_eos_position = gen_length

    return generated_answer[
        :, input_ids.shape[1] : input_ids.shape[1] + first_eos_position + 1
    ]


def main():
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if args.seed is not None:
        set_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args.model_path, args.dtype, args.device_map)
    prompt = read_prompt_from_sources(args.prompt, args.prompt_file, DEFAULT_PROMPT)

    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )

    t0 = time.perf_counter()
    generated_tokens = generate_cached(
        model=model,
        inputs=input_ids,
        eos_early_stop=args.eos_early_stop,
        gen_length=args.gen_length,
        block_length=args.block_length,
        steps=args.steps,
        top_p=args.top_p,
        top_k=args.top_k,
        minimal_topk=args.minimal_topk,
        threshold=args.threshold,
        editing_threshold=args.editing_threshold,
        max_post_steps=args.max_post_steps,
        eos_id=args.eos_id,
        mask_id=args.mask_id,
        num_to_transfer=args.num_to_transfer,
        temperature=args.temperature,
    )
    t1 = time.perf_counter()

    generated_answer = tokenizer.decode(
        generated_tokens[0],
        skip_special_tokens=True,
    )
    print(generated_answer)
    num_generated = generated_tokens.shape[1]
    print(f"\n--- {num_generated} tokens in {t1 - t0:.2f}s ({num_generated / (t1 - t0):.1f} tok/s) ---")


if __name__ == "__main__":
    main()
