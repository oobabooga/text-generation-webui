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


def load_RWKV_model(path):
    print(f'strategy={"cpu" if shared.args.cpu else "cuda"} {"fp32" if shared.args.cpu else "bf16" if shared.args.bf16 else "fp16"}')

    model = RWKV(model=path.as_posix(), strategy=f'{"cpu" if shared.args.cpu else "cuda"} {"fp32" if shared.args.cpu else "bf16" if shared.args.bf16 else "fp16"}')
    pipeline = PIPELINE(model, Path("models/20B_tokenizer.json").as_posix())

    return pipeline
