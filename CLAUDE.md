# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a PyTorch implementation of **LLaDA2.1**, a Mixture-of-Experts (MoE) language model with KV cache optimization. The model uses iterative masked refinement for text generation.

## Running Demos

```bash
# Recommended: Clean example using modular generate_utils
python example_llada.py
python example_llada.py --prompt "Your question here"
python example_llada.py --model inclusionAI/LLaDA2.1-mini --gen-length 1024

# Legacy demos (monolithic scripts)
python demo_llada.py          # Without KV cache (uses HuggingFace AutoModel)
python demo_llada_cache.py    # With KV cache (uses custom LLaDA2MoeModelLM)

# Pass custom prompts
python example_llada.py --prompt "Your question here"
python example_llada.py --prompt-file path/to/prompt.txt

# Adjust generation parameters
python example_llada.py --gen-length 1024 --block-length 32 --steps 32 --threshold 0.5
```

## Code Organization

- **example_llada.py**: Clean, minimal inference example (recommended)
- **generate_utils.py**: Reusable generation utilities
  - `generate_cached()`: Main generation function with KV cache and iterative refinement
  - `load_model_and_tokenizer()`: Model/tokenizer loading helper
- **demo_llada.py** / **demo_llada_cache.py**: Legacy monolithic demo scripts
- **modeling_llada2_moe_cache.py**: Core model implementation
- **configuration_llada2_moe.py**: Model configuration

## Key Architecture Components

### Core Model Classes (modeling_llada2_moe_cache.py)

- **LLaDA2MoeModelLM** (line 978): Main language model class with generation capabilities. Inherits from both PreTrainedModel and GenerationMixin. This is the primary entry point for inference.

- **LLaDA2MoeModel** (line 772): Base transformer model without language modeling head.

- **LLaDA2MoeDecoderLayer** (line 550): Single transformer layer combining attention and MoE FFN.

- **LLaDA2MoeSparseMoeBlock** (line 277): Mixture-of-Experts block with expert routing. Uses grouped expert selection (n_group and topk_group parameters).

- **LLaDA2MoeAttention** (line 412): Multi-head attention with support for:
  - Grouped query attention (GQA)
  - Sliding window attention
  - Flash Attention 2 and SDPA backends
  - QK normalization

- **LLaDA2MoeGate** (line 205): Expert routing mechanism for MoE layers.

### Configuration (configuration_llada2_moe.py)

Key MoE-specific parameters:
- `num_experts`: Number of expert networks (default: 16)
- `num_experts_per_tok`: Experts activated per token (default: 2)
- `n_group`: Number of expert groups (default: 8)
- `topk_group`: Top-k groups to select (default: 4)
- `first_k_dense_replace`: Number of initial layers to keep as dense (default: 0)

### Generation Algorithm

The model uses iterative masked refinement:
1. Generate a block of masked tokens
2. Iteratively refine masked positions based on confidence threshold
3. Transfer highest-confidence predictions in each step
4. Apply post-mask editing steps for global refinement
5. Move to next block

Key generation parameters:
- `block_length`: Tokens per generation block
- `steps`: Refinement iterations per block
- `threshold`: Acceptance threshold for unmasking tokens
- `editing_threshold`: Threshold for post-mask editing
- `max_post_steps`: Maximum global editing steps per block

### KV Cache Implementation

The cache-enabled version (`demo_llada_cache.py` + `modeling_llada2_moe_cache.py`) saves the forward pass results of the last model call for reuse, significantly improving generation efficiency. The model stores intermediate states to avoid recomputing unchanged prefix tokens.

## Reference Scripts

The `references/` directory contains SDAR (Speculative Decoding with Adaptive Refinement) implementations:
- `sdar_generate_ssd.py`: Standard SDAR generation
- `sdar_generate_ssd_policy.py`: SDAR with policy-based refinement
- `eval_gsm8k_sdar.py`: Evaluation on GSM8K benchmark

## Dependencies

Core dependencies:
- PyTorch
- Transformers (HuggingFace)

The codebase expects models to be loaded from HuggingFace Hub or local paths (default: `inclusionAI/LLaDA2.1-mini`).

## Architecture Notes

- The model extends standard transformer architecture with sparse MoE layers
- Uses RMSNorm for layer normalization
- Implements custom RoPE (Rotary Position Embeddings) with optional scaling
- Supports both standard and grouped expert routing
- First `first_k_dense_replace` layers can be dense instead of MoE
- Partial rotary factor (default 0.5) applies RoPE to half of the head dimensions
