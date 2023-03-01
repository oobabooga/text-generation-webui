import os
import time
import types
from pathlib import Path

import numpy as np
import torch

import modules.shared as shared

np.set_printoptions(precision=4, suppress=True, linewidth=200)

os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '0' #  '1' : use CUDA kernel for seq mode (much faster)

from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

class RWKVModel:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(self, path, dtype="fp16", device="cuda"):
        tokenizer_path = Path(f"{path.parent}/20B_tokenizer.json")

        model = RWKV(model=path.as_posix(), strategy=f'{device} {dtype}')
        pipeline = PIPELINE(model, tokenizer_path.as_posix())

        result = self()
        result.model = pipeline
        return result

    def generate(self, context, token_count=20, temperature=1, top_p=1, alpha_frequency=0.25, alpha_presence=0.25, token_ban=[0], token_stop=[], callback=None):
        args = PIPELINE_ARGS(
            temperature = temperature,
            top_p = top_p,
            alpha_frequency = 0.25, # Frequency Penalty (as in GPT-3)
            alpha_presence = 0.25, # Presence Penalty (as in GPT-3)
            token_ban = [0], # ban the generation of some tokens
            token_stop = []
        )

        return self.model.generate(context, token_count=token_count, args=args, callback=callback)
