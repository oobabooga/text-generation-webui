# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import json
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import fire
import torch
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from llama import LLaMA, ModelArgs, Tokenizer, Transformer

os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MP'] = '1'
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '2223'

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("gloo")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size

def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


class LLaMAModel:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(self, path, max_seq_len=2048, max_batch_size=1):
        tokenizer_path = path / "tokenizer.model"
        path = os.path.abspath(path)
        tokenizer_path = os.path.abspath(tokenizer_path)
        
        local_rank, world_size = setup_model_parallel()
        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        generator = load(
            path, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
        )

        result = self()
        result.pipeline = generator
        return result

    def generate(self, prompt, token_count=512, temperature=0.8, top_p=0.95):

        results = self.pipeline.generate(
            [prompt], max_gen_len=token_count, temperature=temperature, top_p=top_p
        )

        return results[0]
