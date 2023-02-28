import os, time, types, torch
from pathlib import Path
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)

os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '0' #  '1' : use CUDA kernel for seq mode (much faster)

import repositories.ChatRWKV.v2.rwkv as rwkv
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

def load_RWKV_model(path):
    os.system("ls")
    model = RWKV(model=path.as_posix(), strategy='cuda fp16')

    out, state = model.forward([187, 510, 1563, 310, 247], None)   # use 20B_tokenizer.json
    print(out.detach().cpu().numpy())                   # get logits
    out, state = model.forward([187, 510], None)
    out, state = model.forward([1563], state)           # RNN has state (use deepcopy if you want to clone it)
    out, state = model.forward([310, 247], state)
    print(out.detach().cpu().numpy())                   # same result as above

    pipeline = PIPELINE(model, Path("repositories/ChatRWKV/20B_tokenizer.json").as_posix())

    return pipeline
