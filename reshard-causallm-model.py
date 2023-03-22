#!/usr/bin/env python

# Based on: https://github.com/oobabooga/text-generation-webui/blob/main/convert-to-torch.py
# License: GNU Affero General Public License v3.0
#
#
# This script converts a transformers model using a custom shard size.
#
# Load a model from a directory and shard it into 2GB chunks:
# python reshard-causallm-model.py --src-model gpt-j-6B --out-path gpt-j-6B-sharded --torch_dtype float16 --max-shard-size 2GB
# 
# Download a model from Hugging Face and shard it using `safetensors`:
# python reshard-causallm-model.py --src-model EleutherAI/gpt-neo-2.7B --out-path gpt-neo-2.7B-sharded --torch_dtype float16 --max-shard-size 500MB --safetensors


import os
import sys
import argparse
import re
from pathlib import Path
import torch
import diffusers
from transformers import AutoModelForCausalLM, AutoTokenizer

def sanitize_name(name):
    pattern = re.compile(r'^(/?[a-zA-Z0-9_.-]+/?)*$')
    if pattern.match(name):
        return name
    else:
        raise ValueError(f"'{name}' is not a valid name or path.")

def sanitize_size(size):
    pattern = re.compile(r'^(\d+)(MB|GB)$')
    if pattern.match(size):
        return size
    else:
        raise ValueError(f"'{size}' is not valid, size should be in MB or GB.")

parser = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog,max_help_position=60))
parser.add_argument("--src-model", type=str, help="Path to a directory or a Hugging Face model identifier.", required=True)
parser.add_argument("--out-path", type=str, help="Path for output. Must be a directory.", required=True)
parser.add_argument("--torch_dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto" , help="Precision dtype.")
parser.add_argument("--max-shard-size", type=str, default="5GB", help="Maximum size of a shard in MB or GB (default: %(default)s).")
parser.add_argument("--cuda", action='store_true', help="Load the model onto the GPU.")
parser.add_argument("--safetensors", action='store_true', help="Save the model using `safetensors`.")
args = parser.parse_args()

if __name__ == '__main__':
    src_model = args.src_model
    out_path = args.out_path
    max_shard_size = args.max_shard_size
    try:
        src_model = sanitize_name(src_model)
        out_path = sanitize_name(out_path)
        max_shard_size = sanitize_size(max_shard_size)
    except ValueError as err:
        print(f"Error: {err}")
        sys.exit()
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if args.cuda:
        device = "cuda"
    else:
        device = "cpu"
    torch_dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if args.torch_dtype != "auto":
        torch_dtype = torch_dtype_map[args.torch_dtype]
    else:
        torch_dtype = "auto"

    try:
        print(f"Loading model from '{src_model}'...")
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=src_model, torch_dtype=torch_dtype, low_cpu_mem_usage=True).to(device)
        print(f"Model loaded.\nLoading tokenizer from '{src_model}'...")
        tokenizer = AutoTokenizer.from_pretrained(src_model)
        print(f"Tokenizer loaded.\nSaving model to '{out_path}' with a maximum shard size of {max_shard_size}...")
        model.save_pretrained(Path(f"{out_path}"), max_shard_size=f"{max_shard_size}", safe_serialization=args.safetensors)
        print(f"Model saved.\nSaving tokenizer to '{out_path}'...")
        tokenizer.save_pretrained(f"{out_path}")
        print(f"Tokenizer saved.\nSaving vocabulary to '{out_path}'...")
        tokenizer.save_vocabulary(f"{out_path}")
        print(f"Vocabulary saved.")
    except Exception as e:
        print(f"Error: {e}")
