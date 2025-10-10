import gc

import torch
from accelerate.utils import is_npu_available, is_xpu_available
from transformers import is_torch_npu_available, is_torch_xpu_available

from modules import shared


def get_device():
    if hasattr(shared.model, 'device'):
        return shared.model.device
    elif torch.cuda.is_available():
        return torch.device('cuda')
    elif shared.args.deepspeed:
        import deepspeed
        return deepspeed.get_accelerator().current_device_name()
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    elif is_torch_xpu_available():
        return torch.device('xpu:0')
    elif is_torch_npu_available():
        return torch.device('npu:0')
    else:
        return None


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
