'''

Converts a transformers model to a format compatible with flexgen.

'''

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=54))
parser.add_argument('MODEL', type=str, default=None, nargs='?', help="Path to the input model.")
args = parser.parse_args()


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    global torch_linear_init_backup
    global torch_layer_norm_init_backup

    torch_linear_init_backup = torch.nn.Linear.reset_parameters
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)

    torch_layer_norm_init_backup = torch.nn.LayerNorm.reset_parameters
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def restore_torch_init():
    """Rollback the change made by disable_torch_init."""
    import torch
    setattr(torch.nn.Linear, "reset_parameters", torch_linear_init_backup)
    setattr(torch.nn.LayerNorm, "reset_parameters", torch_layer_norm_init_backup)


if __name__ == '__main__':
    path = Path(args.MODEL)
    model_name = path.name

    print(f"Loading {model_name}...")
    # disable_torch_init()
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    # restore_torch_init()

    tokenizer = AutoTokenizer.from_pretrained(path)

    out_folder = Path(f"models/{model_name}-np")
    if not Path(out_folder).exists():
        os.mkdir(out_folder)

    print(f"Saving the converted model to {out_folder}...")
    for name, param in tqdm(list(model.model.named_parameters())):
        name = name.replace("decoder.final_layer_norm", "decoder.layer_norm")
        param_path = os.path.join(out_folder, name)
        with open(param_path, "wb") as f:
            np.save(f, param.cpu().detach().numpy())
