'''

Converts a transformers model to safetensors format and shards it.

This makes it faster to load (because of safetensors) and lowers its RAM usage
while loading (because of sharding).

Based on the original script by 81300:

https://gist.github.com/81300/fe5b08bff1cba45296a829b9d6b0f303

'''

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=54))
parser.add_argument('MODEL', type=str, default=None, nargs='?', help="输入模型的路径。")
parser.add_argument('--output', type=str, default=None, help='输出模型的路径 (默认: models/{model_name}_safetensors)。')
parser.add_argument("--max-shard-size", type=str, default="2GB", help="模型碎片的最大大小，以GB或MB为单位 (默认: %(default)s)。")
parser.add_argument('--bf16', action='store_true', help='以bfloat16精度加载模型。需要NVIDIA Ampere GPU。')
args = parser.parse_args()

if __name__ == '__main__':
    path = Path(args.MODEL)
    model_name = path.name

    print(f"正在加载 {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16 if args.bf16 else torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(path)

    out_folder = args.output or Path(f"models/{model_name}_safetensors")
    print(f"正在以最大碎片大小{args.max_shard_size}保存转换后的模型至{out_folder}...")
    model.save_pretrained(out_folder, max_shard_size=args.max_shard_size, safe_serialization=True)
    tokenizer.save_pretrained(out_folder)
