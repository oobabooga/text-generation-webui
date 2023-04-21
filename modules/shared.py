import argparse
from pathlib import Path

import yaml

model = None
tokenizer = None
model_name = "None"
lora_names = []
soft_prompt_tensor = None
soft_prompt = False
is_RWKV = False
is_llamacpp = False

# Chat variables
history = {'internal': [], 'visible': []}
character = 'None'
stop_everything = False
processing_message = '*Is typing...*'

# UI elements (buttons, sliders, HTML, etc)
gradio = {}

# Generation input parameters
input_params = []

# For restarting the interface
need_restart = False

settings = {
    'max_new_tokens': 200,
    'max_new_tokens_min': 1,
    'max_new_tokens_max': 2000,
    'seed': -1,
    'name1': 'You',
    'name2': 'Assistant',
    'context': 'This is a conversation with your Assistant. The Assistant is very helpful and is eager to chat with you and answer your questions.',
    'greeting': '',
    'end_of_turn': '',
    'custom_stopping_strings': '',
    'stop_at_newline': False,
    'add_bos_token': True,
    'ban_eos_token': False,
    'skip_special_tokens': True,
    'truncation_length': 2048,
    'truncation_length_min': 0,
    'truncation_length_max': 8192,
    'mode': 'cai-chat',
    'instruction_template': 'None',
    'chat_prompt_size': 2048,
    'chat_prompt_size_min': 0,
    'chat_prompt_size_max': 2048,
    'chat_generation_attempts': 1,
    'chat_generation_attempts_min': 1,
    'chat_generation_attempts_max': 5,
    'default_extensions': [],
    'chat_default_extensions': ["gallery"],
    'moderation': False,
    "transformer_model_name":"paraphrase-albert-small-v2", 
    "lancedb_uri":"~/.lancedb",
    "lancedb_table_name":"jigsaw_small",
    'presets': {
        'default': 'Default',
        '.*(alpaca|llama)': "LLaMA-Precise",
        '.*pygmalion': 'NovelAI-Storywriter',
        '.*RWKV': 'Naive',
    },
    'prompts': {
        'default': 'QA',
        '.*(gpt4chan|gpt-4chan|4chan)': 'GPT-4chan',
        '.*oasst': 'Open Assistant',
        '.*alpaca': "Alpaca",
    },
    'lora_prompts': {
        'default': 'QA',
        '.*alpaca': "Alpaca",
    }
}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=54))

# Basic settings
parser.add_argument('--notebook', action='store_true', help='Launch the web UI in notebook mode, where the output is written to the same text box as the input.')
parser.add_argument('--chat', action='store_true', help='Launch the web UI in chat mode with a style similar to the Character.AI website.')
parser.add_argument('--cai-chat', action='store_true', help='DEPRECATED: use --chat instead.')
parser.add_argument('--model', type=str, help='Name of the model to load by default.')
parser.add_argument('--lora', type=str, help='Name of the LoRA to apply to the model by default.')
parser.add_argument("--model-dir", type=str, default='models/', help="Path to directory with all the models")
parser.add_argument("--lora-dir", type=str, default='loras/', help="Path to directory with all the loras")
parser.add_argument('--model-menu', action='store_true', help='Show a model menu in the terminal when the web UI is first launched.')
parser.add_argument('--no-stream', action='store_true', help='Don\'t stream the text output in real time.')
parser.add_argument('--settings', type=str, help='Load the default interface settings from this json file. See settings-template.json for an example. If you create a file called settings.json, this file will be loaded by default without the need to use the --settings flag.')
parser.add_argument('--extensions', type=str, nargs="+", help='The list of extensions to load. If you want to load more than one extension, write the names separated by spaces.')
parser.add_argument('--verbose', action='store_true', help='Print the prompts to the terminal.')

# Accelerate/transformers
parser.add_argument('--cpu', action='store_true', help='Use the CPU to generate text. Warning: Training on CPU is extremely slow.')
parser.add_argument('--auto-devices', action='store_true', help='Automatically split the model across the available GPU(s) and CPU.')
parser.add_argument('--gpu-memory', type=str, nargs="+", help='Maxmimum GPU memory in GiB to be allocated per GPU. Example: --gpu-memory 10 for a single GPU, --gpu-memory 10 5 for two GPUs. You can also set values in MiB like --gpu-memory 3500MiB.')
parser.add_argument('--cpu-memory', type=str, help='Maximum CPU memory in GiB to allocate for offloaded weights. Same as above.')
parser.add_argument('--disk', action='store_true', help='If the model is too large for your GPU(s) and CPU combined, send the remaining layers to the disk.')
parser.add_argument('--disk-cache-dir', type=str, default="cache", help='Directory to save the disk cache to. Defaults to "cache".')
parser.add_argument('--load-in-8bit', action='store_true', help='Load the model with 8-bit precision.')
parser.add_argument('--bf16', action='store_true', help='Load the model with bfloat16 precision. Requires NVIDIA Ampere GPU.')
parser.add_argument('--no-cache', action='store_true', help='Set use_cache to False while generating text. This reduces the VRAM usage a bit at a performance cost.')
parser.add_argument('--xformers', action='store_true', help="Use xformer's memory efficient attention. This should increase your tokens/s.")
parser.add_argument('--sdp-attention', action='store_true', help="Use torch 2.0's sdp attention.")
parser.add_argument('--trust-remote-code', action='store_true', help="Set trust_remote_code=True while loading a model. Necessary for ChatGLM.")

# Moderation parameters
parser.add_argument('--moderation',  type=bool, default=False, help="Turn on/off (True/False) the moderation feature. Default False")
parser.add_argument("--transformer_model_name", type=str, required=False, default="paraphrase-albert-small-v2", help="The transformer model name in Hugging face user/model.")
parser.add_argument("--lancedb_uri", type=str, required=False, default="~/.lancedb", help="The local path of the Lance DB storage.")
parser.add_argument("--lancedb_table_name", type=str, required=False, default="jigsaw_small", help="The name of the Lance DB table.")

# llama.cpp
parser.add_argument('--threads', type=int, default=0, help='Number of threads to use in llama.cpp.')

# GPTQ
parser.add_argument('--wbits', type=int, default=0, help='Load a pre-quantized model with specified precision in bits. 2, 3, 4 and 8 are supported.')
parser.add_argument('--model_type', type=str, help='Model type of pre-quantized model. Currently LLaMA, OPT, and GPT-J are supported.')
parser.add_argument('--groupsize', type=int, default=-1, help='Group size.')
parser.add_argument('--pre_layer', type=int, default=0, help='The number of layers to allocate to the GPU. Setting this parameter enables CPU offloading for 4-bit models.')
parser.add_argument('--monkey-patch', action='store_true', help='Apply the monkey patch for using LoRAs with quantized models.')
parser.add_argument('--no-quant_attn', action='store_true', help='(triton) Disable quant attention. If you encounter incoherent results try disabling this.')
parser.add_argument('--no-warmup_autotune', action='store_true', help='(triton) Disable warmup autotune.')
parser.add_argument('--no-fused_mlp', action='store_true', help='(triton) Disable fused mlp. If you encounter "Unexpected mma -> mma layout conversion" try disabling this.')

# FlexGen
parser.add_argument('--flexgen', action='store_true', help='Enable the use of FlexGen offloading.')
parser.add_argument('--percent', type=int, nargs="+", default=[0, 100, 100, 0, 100, 0], help='FlexGen: allocation percentages. Must be 6 numbers separated by spaces (default: 0, 100, 100, 0, 100, 0).')
parser.add_argument("--compress-weight", action="store_true", help="FlexGen: activate weight compression.")
parser.add_argument("--pin-weight", type=str2bool, nargs="?", const=True, default=True, help="FlexGen: whether to pin weights (setting this to False reduces CPU memory by 20%%).")

# DeepSpeed
parser.add_argument('--deepspeed', action='store_true', help='Enable the use of DeepSpeed ZeRO-3 for inference via the Transformers integration.')
parser.add_argument('--nvme-offload-dir', type=str, help='DeepSpeed: Directory to use for ZeRO-3 NVME offloading.')
parser.add_argument('--local_rank', type=int, default=0, help='DeepSpeed: Optional argument for distributed setups.')

# RWKV
parser.add_argument('--rwkv-strategy', type=str, default=None, help='RWKV: The strategy to use while loading the model. Examples: "cpu fp32", "cuda fp16", "cuda fp16i8".')
parser.add_argument('--rwkv-cuda-on', action='store_true', help='RWKV: Compile the CUDA kernel for better performance.')

# Gradio
parser.add_argument('--listen', action='store_true', help='Make the web UI reachable from your local network.')
parser.add_argument('--listen-host', type=str, help='The hostname that the server will use.')
parser.add_argument('--listen-port', type=int, help='The listening port that the server will use.')
parser.add_argument('--share', action='store_true', help='Create a public URL. This is useful for running the web UI on Google Colab or similar.')
parser.add_argument('--auto-launch', action='store_true', default=False, help='Open the web UI in the default browser upon launch.')
parser.add_argument("--gradio-auth-path", type=str, help='Set the gradio authentication file path. The file should contain one or more user:password pairs in this format: "u1:p1,u2:p2,u3:p3"', default=None)

args = parser.parse_args()
args_defaults = parser.parse_args([])

# Deprecation warnings for parameters that have been renamed
deprecated_dict = {}
for k in deprecated_dict:
    if getattr(args, k) != deprecated_dict[k][1]:
        print(f"Warning: --{k} is deprecated and will be removed. Use --{deprecated_dict[k][0]} instead.\n")
        setattr(args, deprecated_dict[k][0], getattr(args, k))

# Deprecation warnings for parameters that have been removed
if args.cai_chat:
    print("Warning: --cai-chat is deprecated. Use --chat instead.\n")
    args.chat = True

# Security warnings
if args.trust_remote_code:
    print("Warning: trust_remote_code is enabled. This is dangerous.\n")
if args.share:
    print("Warning: the gradio \"share link\" feature downloads a proprietary and\nunaudited blob to create a reverse tunnel. This is potentially dangerous.\n")


def is_chat():
    return args.chat


# Loading model-specific settings (default)
with Path(f'{args.model_dir}/config.yaml') as p:
    if p.exists():
        model_config = yaml.safe_load(open(p, 'r').read())
    else:
        model_config = {}

# Applying user-defined model settings
with Path(f'{args.model_dir}/config-user.yaml') as p:
    if p.exists():
        user_config = yaml.safe_load(open(p, 'r').read())
        for k in user_config:
            if k in model_config:
                model_config[k].update(user_config[k])
            else:
                model_config[k] = user_config[k]
