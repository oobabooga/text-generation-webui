import os
from pathlib import Path
from queue import Queue
from threading import Thread

import numpy as np
from tokenizers import Tokenizer

import modules.shared as shared

np.set_printoptions(precision=4, suppress=True, linewidth=200)

os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '1' if shared.args.rwkv_cuda_on else '0' # use CUDA kernel for seq mode (much faster)

from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS


class RWKVModel:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(self, path, dtype="fp16", device="cuda"):
        tokenizer_path = Path(f"{path.parent}/20B_tokenizer.json")

        if shared.args.rwkv_strategy is None:
            model = RWKV(model=os.path.abspath(path), strategy=f'{device} {dtype}')
        else:
            model = RWKV(model=os.path.abspath(path), strategy=shared.args.rwkv_strategy)
        pipeline = PIPELINE(model, os.path.abspath(tokenizer_path))

        result = self()
        result.pipeline = pipeline
        return result

    def generate(self, context="", token_count=20, temperature=1, top_p=1, top_k=50, alpha_frequency=0.1, alpha_presence=0.1, token_ban=[0], token_stop=[], callback=None):
        args = PIPELINE_ARGS(
            temperature = temperature,
            top_p = top_p,
            top_k = top_k,
            alpha_frequency = alpha_frequency, # Frequency Penalty (as in GPT-3)
            alpha_presence = alpha_presence, # Presence Penalty (as in GPT-3)
            token_ban = token_ban, # ban the generation of some tokens
            token_stop = token_stop
        )

        return context+self.pipeline.generate(context, token_count=token_count, args=args, callback=callback)

    def generate_with_streaming(self, **kwargs):
        iterable = Iteratorize(self.generate, kwargs, callback=None)
        reply = kwargs['context']
        for token in iterable:
            reply += token
            yield reply

class RWKVTokenizer:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(self, path):
        tokenizer_path = path / "20B_tokenizer.json"
        tokenizer = Tokenizer.from_file(os.path.abspath(tokenizer_path))

        result = self()
        result.tokenizer = tokenizer
        return result

    def encode(self, prompt):
        return self.tokenizer.encode(prompt).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)

class Iteratorize:

    """
    Transforms a function that takes a callback
    into a lazy iterator (generator).
    """

    def __init__(self, func, kwargs={}, callback=None):
        self.mfunc=func
        self.c_callback=callback
        self.q = Queue(maxsize=1)
        self.sentinel = object()
        self.kwargs = kwargs

        def _callback(val):
            self.q.put(val)

        def gentask():
            ret = self.mfunc(callback=_callback, **self.kwargs)
            self.q.put(self.sentinel)
            if self.c_callback:
                self.c_callback(ret)

        Thread(target=gentask).start()

    def __iter__(self):
        return self

    def __next__(self):
        obj = self.q.get(True,None)
        if obj is self.sentinel:
            raise StopIteration
        else:
            return obj
