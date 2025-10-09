import gc

import torch
from accelerate.utils import is_npu_available, is_xpu_available

from modules import shared


def get_device():
    return getattr(shared.model, 'device', None)


def clear_torch_cache():
    gc.collect()
    if not shared.args.cpu:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif is_xpu_available():
            torch.xpu.empty_cache()
        elif is_npu_available():
            torch.npu.empty_cache()
        elif torch.backends.mps.is_available():
            if hasattr(torch.backends.mps, 'empty_cache'):
                torch.backends.mps.empty_cache()
