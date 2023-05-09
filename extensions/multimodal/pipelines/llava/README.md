## LLaVA pipeline

This module provides 2 pipelines:
- `llava-7b` - for use with LLaVA v0 7B model (finetuned LLaMa 7B)
- `llava-13b` - for use with LLaVA v0 13B model (finetuned LLaMa 13B)

[LLaVA](https://github.com/haotian-liu/LLaVA) uses CLIP `openai/clip-vit-large-patch14` as the vision model, and then a single linear layer. For 13B the projector weights are in `liuhaotian/LLaVA-13b-delta-v0`, and for 7B they are in `liuhaotian/LLaVA-7b-delta-v0`.

The supported parameter combinations for both the vision model, and the projector are: CUDA/32bit, CUDA/16bit, CPU/32bit
