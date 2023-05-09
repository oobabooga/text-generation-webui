import os
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

import modules.shared as shared
from modules.callbacks import Iteratorize

np.set_printoptions(precision=4, suppress=True, linewidth=200)

os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '1' if shared.args.rwkv_cuda_on else '0'  # use CUDA kernel for seq mode (much faster)

from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS


class RWKVModel:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(self, path, dtype="fp16", device="cuda"):
        tokenizer_path = Path(f"{path.parent}/20B_tokenizer.json")

        if shared.args.rwkv_strategy is None:
            model = RWKV(model=str(path), strategy=f'{device} {dtype}')
        else:
            model = RWKV(model=str(path), strategy=shared.args.rwkv_strategy)
        pipeline = PIPELINE(model, str(tokenizer_path))

        result = self()
        result.pipeline = pipeline
        return result

    def generate(self, context="", token_count=20, temperature=1, top_p=1, top_k=50, repetition_penalty=None, alpha_frequency=0.1, alpha_presence=0.1, token_ban=None, token_stop=None, callback=None):
        args = PIPELINE_ARGS(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            alpha_frequency=alpha_frequency,  # Frequency Penalty (as in GPT-3)
            alpha_presence=alpha_presence,  # Presence Penalty (as in GPT-3)
            token_ban=token_ban or [0],  # ban the generation of some tokens
            token_stop=token_stop or []
        )

        return self.pipeline.generate(context, token_count=token_count, args=args, callback=callback)

    def generate_with_streaming(self, **kwargs):
        with Iteratorize(self.generate, kwargs, callback=None) as generator:
            reply = ''
            for token in generator:
                reply += token
                yield reply


class RWKVTokenizer:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(self, path):
        tokenizer_path = path / "20B_tokenizer.json"
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

        result = self()
        result.tokenizer = tokenizer
        return result

    def encode(self, prompt):
        return self.tokenizer.encode(prompt).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)
