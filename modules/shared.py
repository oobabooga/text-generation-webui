import argparse
from collections import OrderedDict
from pathlib import Path

import yaml

from modules.logging_colors import logger


# Model variables
model = None
tokenizer = None
model_name = "None"
is_seq2seq = False
model_dirty_from_training = False
lora_names = []

# Generation variables
stop_everything = False
generation_lock = None
processing_message = '*Is typing...*'

# UI variables
gradio = {}
persistent_interface_state = {}
need_restart = False
session_is_loading = False

# UI defaults
settings = {
    'dark_theme': True,
    'show_controls': True,
    'start_with': '',
    'mode': 'chat',
    'chat_style': 'cai-chat',
    'character': 'None',
    'prompt-default': 'QA',
    'prompt-notebook': 'QA',
    'preset': 'simple-1',
    'max_new_tokens': 200,
    'max_new_tokens_min': 1,
    'max_new_tokens_max': 4096,
    'seed': -1,
    'negative_prompt': '',
    'truncation_length': 2048,
    'truncation_length_min': 0,
    'truncation_length_max': 16384,
    'custom_stopping_strings': '',
    'auto_max_new_tokens': False,
    'max_tokens_second': 0,
    'ban_eos_token': False,
    'custom_token_bans': '',
    'add_bos_token': True,
    'skip_special_tokens': True,
    'stream': True,
    'name1': 'You',
    'name2': 'Assistant',
    'context': 'This is a conversation with your Assistant. It is a computer program designed to help you with various tasks such as answering questions, providing recommendations, and helping with decision making. You can ask it anything you want and it will do its best to give you accurate and relevant information.',
    'greeting': '',
    'instruction_template': 'Alpaca',
    'chat-instruct_command': 'Continue the chat dialogue below. Write a single reply for the character "<|character|>".\n\n<|prompt|>',
    'autoload_model': False,
    'default_extensions': ['gallery'],
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
parser.add_argument('--notebook', action='store_true', help='DEPRECATED')
parser.add_argument('--chat', action='store_true', help='DEPRECATED')
parser.add_argument('--multi-user', action='store_true', help='Multi-user mode. Chat histories are not saved or automatically loaded. WARNING: this is highly experimental.')
parser.add_argument('--character', type=str, help='The name of the character to load in chat mode by default.')
parser.add_argument('--model', type=str, help='Name of the model to load by default.')
parser.add_argument('--lora', type=str, nargs="+", help='The list of LoRAs to load. If you want to load more than one LoRA, write the names separated by spaces.')
parser.add_argument("--model-dir", type=str, default='models/', help="Path to directory with all the models")
parser.add_argument("--lora-dir", type=str, default='loras/', help="Path to directory with all the loras")
parser.add_argument('--model-menu', action='store_true', help='Show a model menu in the terminal when the web UI is first launched.')
parser.add_argument('--no-stream', action='store_true', help='DEPRECATED')
parser.add_argument('--settings', type=str, help='Load the default interface settings from this yaml file. See settings-template.yaml for an example. If you create a file called settings.yaml, this file will be loaded by default without the need to use the --settings flag.')
parser.add_argument('--extensions', type=str, nargs="+", help='The list of extensions to load. If you want to load more than one extension, write the names separated by spaces.')
parser.add_argument('--verbose', action='store_true', help='Print the prompts to the terminal.')
parser.add_argument('--chat-buttons', action='store_true', help='Show buttons on chat tab instead of hover menu.')

# Model loader
parser.add_argument('--loader', type=str, help='Choose the model loader manually, otherwise, it will get autodetected. Valid options: transformers, autogptq, gptq-for-llama, exllama, exllama_hf, llamacpp, rwkv')

# Accelerate/transformers
parser.add_argument('--cpu', action='store_true', help='Use the CPU to generate text. Warning: Training on CPU is extremely slow.')
parser.add_argument('--auto-devices', action='store_true', help='Automatically split the model across the available GPU(s) and CPU.')
parser.add_argument('--gpu-memory', type=str, nargs="+", help='Maximum GPU memory in GiB to be allocated per GPU. Example: --gpu-memory 10 for a single GPU, --gpu-memory 10 5 for two GPUs. You can also set values in MiB like --gpu-memory 3500MiB.')
parser.add_argument('--cpu-memory', type=str, help='Maximum CPU memory in GiB to allocate for offloaded weights. Same as above.')
parser.add_argument('--disk', action='store_true', help='If the model is too large for your GPU(s) and CPU combined, send the remaining layers to the disk.')
parser.add_argument('--disk-cache-dir', type=str, default="cache", help='Directory to save the disk cache to. Defaults to "cache".')
parser.add_argument('--load-in-8bit', action='store_true', help='Load the model with 8-bit precision (using bitsandbytes).')
parser.add_argument('--bf16', action='store_true', help='Load the model with bfloat16 precision. Requires NVIDIA Ampere GPU.')
parser.add_argument('--no-cache', action='store_true', help='Set use_cache to False while generating text. This reduces the VRAM usage a bit at a performance cost.')
parser.add_argument('--xformers', action='store_true', help="Use xformer's memory efficient attention. This should increase your tokens/s.")
parser.add_argument('--sdp-attention', action='store_true', help="Use torch 2.0's sdp attention.")
parser.add_argument('--trust-remote-code', action='store_true', help="Set trust_remote_code=True while loading a model. Necessary for ChatGLM and Falcon.")

# Accelerate 4-bit
parser.add_argument('--load-in-4bit', action='store_true', help='Load the model with 4-bit precision (using bitsandbytes).')
parser.add_argument('--compute_dtype', type=str, default="float16", help="compute dtype for 4-bit. Valid options: bfloat16, float16, float32.")
parser.add_argument('--quant_type', type=str, default="nf4", help='quant_type for 4-bit. Valid options: nf4, fp4.')
parser.add_argument('--use_double_quant', action='store_true', help='use_double_quant for 4-bit.')

# llama.cpp
parser.add_argument('--threads', type=int, default=0, help='Number of threads to use.')
parser.add_argument('--n_batch', type=int, default=512, help='Maximum number of prompt tokens to batch together when calling llama_eval.')
parser.add_argument('--no-mmap', action='store_true', help='Prevent mmap from being used.')
parser.add_argument('--low-vram', action='store_true', help='Low VRAM Mode')
parser.add_argument('--mlock', action='store_true', help='Force the system to keep the model in RAM.')
parser.add_argument('--mul_mat_q', action='store_true', help='Activate new mulmat kernels.')
parser.add_argument('--cache-capacity', type=str, help='Maximum cache capacity. Examples: 2000MiB, 2GiB. When provided without units, bytes will be assumed.')
parser.add_argument('--n-gpu-layers', type=int, default=0, help='Number of layers to offload to the GPU.')
parser.add_argument('--tensor_split', type=str, default=None, help="Split the model across multiple GPUs, comma-separated list of proportions, e.g. 18,17")
parser.add_argument('--n_ctx', type=int, default=2048, help='Size of the prompt context.')
parser.add_argument('--llama_cpp_seed', type=int, default=0, help='Seed for llama-cpp models. Default 0 (random)')

# GPTQ
parser.add_argument('--wbits', type=int, default=0, help='Load a pre-quantized model with specified precision in bits. 2, 3, 4 and 8 are supported.')
parser.add_argument('--model_type', type=str, help='Model type of pre-quantized model. Currently LLaMA, OPT, and GPT-J are supported.')
parser.add_argument('--groupsize', type=int, default=-1, help='Group size.')
parser.add_argument('--pre_layer', type=int, nargs="+", help='The number of layers to allocate to the GPU. Setting this parameter enables CPU offloading for 4-bit models. For multi-gpu, write the numbers separated by spaces, eg --pre_layer 30 60.')
parser.add_argument('--checkpoint', type=str, help='The path to the quantized checkpoint file. If not specified, it will be automatically detected.')
parser.add_argument('--monkey-patch', action='store_true', help='Apply the monkey patch for using LoRAs with quantized models.')

# AutoGPTQ
parser.add_argument('--triton', action='store_true', help='Use triton.')
parser.add_argument('--no_inject_fused_attention', action='store_true', help='Do not use fused attention (lowers VRAM requirements).')
parser.add_argument('--no_inject_fused_mlp', action='store_true', help='Triton mode only: Do not use fused MLP (lowers VRAM requirements).')
parser.add_argument('--no_use_cuda_fp16', action='store_true', help='This can make models faster on some systems.')
parser.add_argument('--desc_act', action='store_true', help='For models that don\'t have a quantize_config.json, this parameter is used to define whether to set desc_act or not in BaseQuantizeConfig.')
parser.add_argument('--disable_exllama', action='store_true', help='Disable ExLlama kernel, which can improve inference speed on some systems.')

# ExLlama
parser.add_argument('--gpu-split', type=str, help="Comma-separated list of VRAM (in GB) to use per GPU device for model layers, e.g. 20,7,7")
parser.add_argument('--max_seq_len', type=int, default=2048, help="Maximum sequence length.")
parser.add_argument('--cfg-cache', action='store_true', help="ExLlama_HF: Create an additional cache for CFG negative prompts. Necessary to use CFG with that loader, but not necessary for CFG with base ExLlama.")

# DeepSpeed
parser.add_argument('--deepspeed', action='store_true', help='Enable the use of DeepSpeed ZeRO-3 for inference via the Transformers integration.')
parser.add_argument('--nvme-offload-dir', type=str, help='DeepSpeed: Directory to use for ZeRO-3 NVME offloading.')
parser.add_argument('--local_rank', type=int, default=0, help='DeepSpeed: Optional argument for distributed setups.')

# RWKV
parser.add_argument('--rwkv-strategy', type=str, default=None, help='RWKV: The strategy to use while loading the model. Examples: "cpu fp32", "cuda fp16", "cuda fp16i8".')
parser.add_argument('--rwkv-cuda-on', action='store_true', help='RWKV: Compile the CUDA kernel for better performance.')

# RoPE
parser.add_argument('--alpha_value', type=float, default=1, help="Positional embeddings alpha factor for NTK RoPE scaling. Use either this or compress_pos_emb, not both.")
parser.add_argument('--rope_freq_base', type=int, default=0, help="If greater than 0, will be used instead of alpha_value. Those two are related by rope_freq_base = 10000 * alpha_value ^ (64 / 63).")
parser.add_argument('--compress_pos_emb', type=int, default=1, help="Positional embeddings compression factor. Should be set to (context length) / (model\'s original context length). Equal to 1/rope_freq_scale.")

# Gradio
parser.add_argument('--listen', action='store_true', help='Make the web UI reachable from your local network.')
parser.add_argument('--listen-host', type=str, help='The hostname that the server will use.')
parser.add_argument('--listen-port', type=int, help='The listening port that the server will use.')
parser.add_argument('--share', action='store_true', help='Create a public URL. This is useful for running the web UI on Google Colab or similar.')
parser.add_argument('--auto-launch', action='store_true', default=False, help='Open the web UI in the default browser upon launch.')
parser.add_argument("--gradio-auth", type=str, help='set gradio authentication like "username:password"; or comma-delimit multiple like "u1:p1,u2:p2,u3:p3"', default=None)
parser.add_argument("--gradio-auth-path", type=str, help='Set the gradio authentication file path. The file should contain one or more user:password pairs in this format: "u1:p1,u2:p2,u3:p3"', default=None)
parser.add_argument("--ssl-keyfile", type=str, help='The path to the SSL certificate key file.', default=None)
parser.add_argument("--ssl-certfile", type=str, help='The path to the SSL certificate cert file.', default=None)

# API
parser.add_argument('--api', action='store_true', help='Enable the API extension.')
parser.add_argument('--api-blocking-port', type=int, default=5000, help='The listening port for the blocking API.')
parser.add_argument('--api-streaming-port', type=int, default=5005, help='The listening port for the streaming API.')
parser.add_argument('--public-api', action='store_true', help='Create a public URL for the API using Cloudfare.')
parser.add_argument('--public-api-id', type=str, help='Tunnel ID for named Cloudflare Tunnel. Use together with public-api option.', default=None)

# Multimodal
parser.add_argument('--multimodal-pipeline', type=str, default=None, help='The multimodal pipeline to use. Examples: llava-7b, llava-13b.')

args = parser.parse_args()
args_defaults = parser.parse_args([])

# Deprecation warnings
for k in ['chat', 'notebook', 'no_stream']:
    if getattr(args, k):
        logger.warning(f'The --{k} flag has been deprecated and will be removed soon. Please remove that flag.')

# Security warnings
if args.trust_remote_code:
    logger.warning("trust_remote_code is enabled. This is dangerous.")
if args.share:
    logger.warning("The gradio \"share link\" feature uses a proprietary executable to create a reverse tunnel. Use it with care.")
if args.multi_user:
    logger.warning("The multi-user mode is highly experimental. DO NOT EXPOSE IT TO THE INTERNET.")


def fix_loader_name(name):
    if not name:
        return name

    name = name.lower()
    if name in ['llamacpp', 'llama.cpp', 'llama-cpp', 'llama cpp']:
        return 'llama.cpp'
    if name in ['llamacpp_hf', 'llama.cpp_hf', 'llama-cpp-hf', 'llamacpp-hf', 'llama.cpp-hf']:
        return 'llamacpp_HF'
    elif name in ['transformers', 'huggingface', 'hf', 'hugging_face', 'hugging face']:
        return 'Transformers'
    elif name in ['autogptq', 'auto-gptq', 'auto_gptq', 'auto gptq']:
        return 'AutoGPTQ'
    elif name in ['gptq-for-llama', 'gptqforllama', 'gptqllama', 'gptq for llama', 'gptq_for_llama']:
        return 'GPTQ-for-LLaMa'
    elif name in ['exllama', 'ex-llama', 'ex_llama', 'exlama']:
        return 'ExLlama'
    elif name in ['exllama-hf', 'exllama_hf', 'exllama hf', 'ex-llama-hf', 'ex_llama_hf']:
        return 'ExLlama_HF'
    elif name in ['exllamav2', 'exllama-v2', 'ex_llama-v2', 'exlamav2', 'exlama-v2', 'exllama2', 'exllama-2']:
        return 'ExLlamav2'
    elif name in ['exllamav2-hf', 'exllamav2_hf', 'exllama-v2-hf', 'exllama_v2_hf', 'exllama-v2_hf', 'exllama2-hf', 'exllama2_hf', 'exllama-2-hf', 'exllama_2_hf', 'exllama-2_hf']:
        return 'ExLlamav2_HF'
    elif name in ['ctransformers', 'ctranforemrs', 'ctransformer']:
        return 'ctransformers'


def add_extension(name):
    if args.extensions is None:
        args.extensions = [name]
    elif 'api' not in args.extensions:
        args.extensions.append(name)


def is_chat():
    return True


args.loader = fix_loader_name(args.loader)

# Activate the API extension
if args.api or args.public_api:
    add_extension('api')

# Activate the multimodal extension
if args.multimodal_pipeline is not None:
    add_extension('multimodal')

# Load model-specific settings
with Path(f'{args.model_dir}/config.yaml') as p:
    if p.exists():
        model_config = yaml.safe_load(open(p, 'r').read())
    else:
        model_config = {}

# Load custom model-specific settings
with Path(f'{args.model_dir}/config-user.yaml') as p:
    if p.exists():
        user_config = yaml.safe_load(open(p, 'r').read())
    else:
        user_config = {}

model_config = OrderedDict(model_config)
user_config = OrderedDict(user_config)
