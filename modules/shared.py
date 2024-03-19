import argparse
import copy
import os
import sys
from collections import OrderedDict
from pathlib import Path

import yaml

from modules.logging_colors import logger

# Model variables
model = None
tokenizer = None
model_name = 'None'
is_seq2seq = False
model_dirty_from_training = False
lora_names = []

# Generation variables
stop_everything = False
generation_lock = None
processing_message = '*正在输入...*'

# UI variables
gradio = {}
persistent_interface_state = {}
need_restart = False

# UI defaults
settings = {
    'dark_theme': True,
    'show_controls': True,
    'start_with': '',
    'mode': 'chat',
    'chat_style': 'cai-chat',
    'prompt-default': 'QA',
    'prompt-notebook': 'QA',
    'preset': 'simple-1',
    'max_new_tokens': 512,
    'max_new_tokens_min': 1,
    'max_new_tokens_max': 4096,
    'negative_prompt': '',
    'seed': -1,
    'truncation_length': 2048,
    'truncation_length_min': 0,
    'truncation_length_max': 200000,
    'max_tokens_second': 0,
    'max_updates_second': 0,
    'prompt_lookup_num_tokens': 0,
    'custom_stopping_strings': '',
    'custom_token_bans': '',
    'auto_max_new_tokens': False,
    'ban_eos_token': False,
    'add_bos_token': True,
    'skip_special_tokens': True,
    'stream': True,
    'character': 'Assistant',
    'name1': 'You',
    'user_bio': '',
    'custom_system_message': '',
    'instruction_template_str': "{%- set ns = namespace(found=false) -%}\n{%- for message in messages -%}\n    {%- if message['role'] == 'system' -%}\n        {%- set ns.found = true -%}\n    {%- endif -%}\n{%- endfor -%}\n{%- if not ns.found -%}\n    {{- '' + 'Below is an instruction that describes a task. Write a response that appropriately completes the request.' + '\\n\\n' -}}\n{%- endif %}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' -%}\n        {{- '' + message['content'] + '\\n\\n' -}}\n    {%- else -%}\n        {%- if message['role'] == 'user' -%}\n            {{-'### Instruction:\\n' + message['content'] + '\\n\\n'-}}\n        {%- else -%}\n            {{-'### Response:\\n' + message['content'] + '\\n\\n' -}}\n        {%- endif -%}\n    {%- endif -%}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n    {{-'### Response:\\n'-}}\n{%- endif -%}",
    'chat_template_str': "{%- for message in messages %}\n    {%- if message['role'] == 'system' -%}\n        {%- if message['content'] -%}\n            {{- message['content'] + '\\n\\n' -}}\n        {%- endif -%}\n        {%- if user_bio -%}\n            {{- user_bio + '\\n\\n' -}}\n        {%- endif -%}\n    {%- else -%}\n        {%- if message['role'] == 'user' -%}\n            {{- name1 + ': ' + message['content'] + '\\n'-}}\n        {%- else -%}\n            {{- name2 + ': ' + message['content'] + '\\n' -}}\n        {%- endif -%}\n    {%- endif -%}\n{%- endfor -%}",
    'chat-instruct_command': 'Continue the chat dialogue below. Write a single reply for the character "<|character|>".\n\n<|prompt|>',
    'autoload_model': False,
    'gallery-items_per_page': 50,
    'gallery-open': False,
    'default_extensions': ['gallery'],
}

default_settings = copy.deepcopy(settings)

# Parser copied from https://github.com/vladmandic/automatic
parser = argparse.ArgumentParser(description="Text generation web UI", conflict_handler='resolve', add_help=True, formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=55, indent_increment=2, width=200))

# Basic settings
group = parser.add_argument_group('基本设置')
group.add_argument('--multi-user', action='store_true', help='多用户模式。聊天历史不会被保存或自动加载。警告：这可能不适合公开分享。')
group.add_argument('--character', type=str, help='默认情况下在聊天模式中加载的角色名。')
group.add_argument('--model', type=str, help='默认情况下要加载的模型名。')
group.add_argument('--lora', type=str, nargs='+', help='要加载的LoRA列表。如果你想加载多于一个LoRA，将名字用空格分隔。')
group.add_argument('--model-dir', type=str, default='models/', help='所有模型的目录路径。')
group.add_argument('--lora-dir', type=str, default='loras/', help='所有LoRAs的目录路径。')
group.add_argument('--model-menu', action='store_true', help='当web UI首次启动时，在终端显示模型菜单。')
group.add_argument('--settings', type=str, help='从这个yaml文件加载默认界面设置。参见settings-template.yaml的示例。如果你创建了一个叫做settings.yaml的文件，这个文件将会默认加载，无需使用--settings命令行参数。')
group.add_argument('--extensions', type=str, nargs='+', help='要加载的扩展列表。如果你想加载多于一个扩展，将名字用空格分隔。')
group.add_argument('--verbose', action='store_true', help='在终端打印提示。')
group.add_argument('--chat-buttons', action='store_true', help='在聊天标签页显示按钮，而不是悬浮菜单。')

# Model loader
group = parser.add_argument_group('模型加载器')
group.add_argument('--loader', type=str, help='手动选择模型加载器，否则将自动检测。有效选项包括：Transformers, llama.cpp, llamacpp_HF, ExLlamav2_HF, ExLlamav2, AutoGPTQ, AutoAWQ, GPTQ-for-LLaMa, ctransformers, QuIP#。')

# Transformers/Accelerate
group = parser.add_argument_group('Transformers/Accelerate')
group.add_argument('--cpu', action='store_true', help='使用CPU生成文本。警告：在CPU上训练速度极慢。')
group.add_argument('--auto-devices', action='store_true', help='自动将模型分布到可用的GPU和CPU上。')
group.add_argument('--gpu-memory', type=str, nargs='+', help='每个GPU分配的最大GPU内存（以GiB为单位）。例如：单个GPU使用--gpu-memory 10，两个GPU使用--gpu-memory 10 5。您也可以像这样设置MiB值--gpu-memory 3500MiB。')
group.add_argument('--cpu-memory', type=str, help='分配给卸载权重的最大CPU内存（以GiB为单位）。同上。')
group.add_argument('--disk', action='store_true', help='如果模型对于您的GPU和CPU的组合来说太大，将剩余的层发送到磁盘。')
group.add_argument('--disk-cache-dir', type=str, default='cache', help='保存磁盘缓存的目录。默认为"cache"。')
group.add_argument('--load-in-8bit', action='store_true', help='以8位精度加载模型（使用bitsandbytes）。')
group.add_argument('--bf16', action='store_true', help='以bfloat16精度加载模型。需要NVIDIA Ampere GPU。')
group.add_argument('--no-cache', action='store_true', help='生成文本时将use_cache设置为False。这可以稍微减少VRAM使用，但会降低性能。')
group.add_argument('--trust-remote-code', action='store_true', help='加载模型时将trust_remote_code设置为True。对于某些模型来说是必要的。')
group.add_argument('--force-safetensors', action='store_true', help='加载模型时将use_safetensors设置为True。这可以防止任意代码执行。')
group.add_argument('--no_use_fast', action='store_true', help='加载分词器时将use_fast设置为False（默认为True）。如果您遇到与use_fast相关的问题，请使用此选项。')
group.add_argument('--use_flash_attention_2', action='store_true', help='加载模型时将use_flash_attention_2设置为True。')

# bitsandbytes 4-bit
group = parser.add_argument_group('bitsandbytes 4-bit')
group.add_argument('--load-in-4bit', action='store_true', help='以4位精度加载模型（使用bitsandbytes）。')
group.add_argument('--use_double_quant', action='store_true', help='对4位使用use_double_quant。')
group.add_argument('--compute_dtype', type=str, default='float16', help='4位的计算数据类型。有效选项：bfloat16, float16, float32。')
group.add_argument('--quant_type', type=str, default='nf4', help='4位的量化类型。有效选项：nf4, fp4。')

# llama.cpp
group = parser.add_argument_group('llama.cpp')
group.add_argument('--tensorcores', action='store_true', help='使用支持tensor cores的llama-cpp-python编译版本。这可以提高RTX卡的性能。仅限NVIDIA。')
group.add_argument('--n_ctx', type=int, default=2048, help='提示词上下文的大小。')
group.add_argument('--threads', type=int, default=0, help='使用的线程数。')
group.add_argument('--threads-batch', type=int, default=0, help='用于批处理/提示词处理的线程数。')
group.add_argument('--no_mul_mat_q', action='store_true', help='禁用mulmat内核。')
group.add_argument('--n_batch', type=int, default=512, help='在调用llama_eval时批量处理的最大提示词令牌数。')
group.add_argument('--no-mmap', action='store_true', help='防止使用mmap。')
group.add_argument('--mlock', action='store_true', help='强制系统将模型保留在RAM中。')
group.add_argument('--n-gpu-layers', type=int, default=0, help='卸载到GPU的层数。')
group.add_argument('--tensor_split', type=str, default=None, help='将模型分布在多个GPU上。逗号分隔的比例列表。例如：18,17。')
group.add_argument('--numa', action='store_true', help='为llama.cpp激活NUMA任务分配。')
group.add_argument('--logits_all', action='store_true', help='需要设置以便困惑度评估能够工作。否则，忽略它，因为它会使提示词处理变慢。')
group.add_argument('--no_offload_kqv', action='store_true', help='不要将K, Q, V卸载到GPU。这样可以节省VRAM，但会降低性能。')
group.add_argument('--cache-capacity', type=str, help='最大缓存容量（llama-cpp-python）。例如：2000MiB, 2GiB。如果没有提供单位，默认为字节。')
group.add_argument('--row_split', action='store_true', help='在GPUs之间按行分割模型。这可能会提高多GPU性能。')
group.add_argument('--streaming-llm', action='store_true', help='激活StreamingLLM以避免在删除旧消息时重新评估整个提示词。')
group.add_argument('--attention-sink-size', type=int, default=5, help='StreamingLLM：sink token的数量。仅在修剪后的提示词不与旧提示词前缀相同时使用。')

# ExLlamaV2
group = parser.add_argument_group('ExLlamaV2')
group.add_argument('--gpu-split', type=str, help='用逗号分隔的VRAM（以GB为单位）列表，指定每个GPU设备用于模型层的内存。示例：20,7,7。')
group.add_argument('--autosplit', action='store_true', help='自动将模型张量分布在可用的GPU上。这会导致忽略--gpu-split参数。')
group.add_argument('--max_seq_len', type=int, default=2048, help='最大序列长度。')
group.add_argument('--cfg-cache', action='store_true', help='ExLlamav2_HF：为CFG负提示词创建额外的缓存。使用该加载器进行CFG时必需。')
group.add_argument('--no_flash_attn', action='store_true', help='强制不使用flash-attention。')
group.add_argument('--cache_8bit', action='store_true', help='使用8位缓存以节省VRAM。')
group.add_argument('--cache_4bit', action='store_true', help='使用Q4缓存以节省VRAM。')
group.add_argument('--num_experts_per_token', type=int, default=2, help='用于生成的专家数量。适用于像Mixtral这样的MoE模型。')

# AutoGPTQ
group = parser.add_argument_group('AutoGPTQ')
group.add_argument('--triton', action='store_true', help='使用triton。')
group.add_argument('--no_inject_fused_attention', action='store_true', help='禁用融合注意力机制，这将减少VRAM的使用，但会导致推理速度变慢。')
group.add_argument('--no_inject_fused_mlp', action='store_true', help='仅Triton模式：禁用融合MLP，这将减少VRAM的使用，但会导致推理速度变慢。')
group.add_argument('--no_use_cuda_fp16', action='store_true', help='这可以在某些系统上加快模型的速度。')
group.add_argument('--desc_act', action='store_true', help='对于没有quantize_config.json的模型，此参数用于定义是否在BaseQuantizeConfig中设置desc_act。')
group.add_argument('--disable_exllama', action='store_true', help='禁用ExLlama内核，这可以在某些系统上提高推理速度。')
group.add_argument('--disable_exllamav2', action='store_true', help='禁用ExLlamav2内核。')

# GPTQ-for-LLaMa
group = parser.add_argument_group('GPTQ-for-LLaMa')
group.add_argument('--wbits', type=int, default=0, help='以指定的位精度加载预量化模型。支持2、3、4和8位。')
group.add_argument('--model_type', type=str, help='预量化模型的类型。目前支持LLaMA、OPT和GPT-J。')
group.add_argument('--groupsize', type=int, default=-1, help='组大小。')
group.add_argument('--pre_layer', type=int, nargs='+', help='分配给GPU的层数。设置此参数可启用4位模型的CPU卸载。对于多GPU，将数字用空格分隔，例如 --pre_layer 30 60。')
group.add_argument('--checkpoint', type=str, help='量化检查点文件的路径。如果未指定，将自动检测。')
group.add_argument('--monkey-patch', action='store_true', help='应用monkey patch以便与量化模型一起使用LoRAs。')

# HQQ
group = parser.add_argument_group('HQQ')
group.add_argument('--hqq-backend', type=str, default='PYTORCH_COMPILE', help='HQQ加载器的后端。有效选项：PYTORCH, PYTORCH_COMPILE, ATEN。')

# DeepSpeed
group = parser.add_argument_group('DeepSpeed')
group.add_argument('--deepspeed', action='store_true', help='通过Transformers集成启用DeepSpeed ZeRO-3进行推理。')
group.add_argument('--nvme-offload-dir', type=str, help='DeepSpeed：用于ZeRO-3 NVME卸载的目录。')
group.add_argument('--local_rank', type=int, default=0, help='DeepSpeed：分布式设置的可选参数。')

# RoPE
group = parser.add_argument_group('RoPE')
group.add_argument('--alpha_value', type=float, default=1, help='NTK RoPE缩放的位置嵌入alpha因子。使用这个或compress_pos_emb，不要同时使用。')
group.add_argument('--rope_freq_base', type=int, default=0, help='如果大于0，将代替alpha_value使用。这两个参数的关系为rope_freq_base = 10000 * alpha_value ^ (64 / 63)。')
group.add_argument('--compress_pos_emb', type=int, default=1, help="位置嵌入的压缩因子。应该设置为（上下文长度）/（模型的原始上下文长度）。等于1/rope_freq_scale。")

# Gradio
group = parser.add_argument_group('Gradio')
group.add_argument('--listen', action='store_true', help='使Web界面能够从您的本地网络访问。')
group.add_argument('--listen-port', type=int, help='服务器将使用的监听端口。')
group.add_argument('--listen-host', type=str, help='服务器将使用的主机名。')
group.add_argument('--share', action='store_true', help='创建一个公共URL。这对于在Google Colab或类似环境上运行Web界面很有用。')
group.add_argument('--auto-launch', action='store_true', default=False, help='启动时在默认浏览器中打开Web界面。')
group.add_argument('--gradio-auth', type=str, help='设置Gradio认证密码，格式为"用户名:密码"。也可以提供多个凭证，如"u1:p1,u2:p2,u3:p3"。', default=None)
group.add_argument('--gradio-auth-path', type=str, help='设置Gradio认证文件路径。文件应包含一个或多个上述格式的用户:密码对。', default=None)
group.add_argument('--ssl-keyfile', type=str, help='SSL证书密钥文件的路径。', default=None)
group.add_argument('--ssl-certfile', type=str, help='SSL证书文件的路径。', default=None)

# API
group = parser.add_argument_group('API')
group.add_argument('--api', action='store_true', help='启用API扩展。')
group.add_argument('--public-api', action='store_true', help='使用Cloudfare创建一个公共API URL。')
group.add_argument('--public-api-id', type=str, help='指定Cloudflare Tunnel的隧道ID。与public-api选项一起使用。', default=None)
group.add_argument('--api-port', type=int, default=5000, help='API的监听端口。')
group.add_argument('--api-key', type=str, default='', help='API认证密钥。')
group.add_argument('--admin-key', type=str, default='', help='用于管理任务的API认证密钥，如加载和卸载模型。如果未设置，将与--api-key相同。')
group.add_argument('--nowebui', action='store_true', help='不启动Gradio UI。适用于以独立模式启动API。')

# Multimodal
group = parser.add_argument_group('Multimodal')
group.add_argument('--multimodal-pipeline', type=str, default=None, help='要使用的多模态管道。示例：llava-7b, llava-13b。')

# Deprecated parameters
# group = parser.add_argument_group('Deprecated')

args = parser.parse_args()
args_defaults = parser.parse_args([])
provided_arguments = []
for arg in sys.argv[1:]:
    arg = arg.lstrip('-').replace('-', '_')
    if hasattr(args, arg):
        provided_arguments.append(arg)

deprecated_args = []


def do_cmd_flags_warnings():

    # Deprecation warnings
    for k in deprecated_args:
        if getattr(args, k):
            logger.warning(f'--{k}命令行参数已被弃用，即将被移除。请移除该参数。')

    # Security warnings
    if args.trust_remote_code:
        logger.warning('trust_remote_code已启用。这有危险。')
    if 'COLAB_GPU' not in os.environ and not args.nowebui:
        if args.share:
            logger.warning("Gradio的“共享链接”功能使用专有的可执行文件创建反向隧道。请谨慎使用。")
        if any((args.listen, args.share)) and not any((args.gradio_auth, args.gradio_auth_path)):
            logger.warning("\n您可能正在将Web界面暴露给整个互联网，而没有任何访问密码。\n您可以使用“--gradio-auth”标志创建一个，如下所示：\n\n--gradio-auth 用户名:密码\n\n确保将用户名:密码替换为您自己的。")
            if args.multi_user:
                logger.warning('\n多用户模式处于高度实验阶段，不应公开分享。')


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
    elif name in ['exllamav2', 'exllama-v2', 'ex_llama-v2', 'exlamav2', 'exlama-v2', 'exllama2', 'exllama-2']:
        return 'ExLlamav2'
    elif name in ['exllamav2-hf', 'exllamav2_hf', 'exllama-v2-hf', 'exllama_v2_hf', 'exllama-v2_hf', 'exllama2-hf', 'exllama2_hf', 'exllama-2-hf', 'exllama_2_hf', 'exllama-2_hf']:
        return 'ExLlamav2_HF'
    elif name in ['ctransformers', 'ctranforemrs', 'ctransformer']:
        return 'ctransformers'
    elif name in ['autoawq', 'awq', 'auto-awq']:
        return 'AutoAWQ'
    elif name in ['quip#', 'quip-sharp', 'quipsharp', 'quip_sharp']:
        return 'QuIP#'
    elif name in ['hqq']:
        return 'HQQ'


def add_extension(name, last=False):
    if args.extensions is None:
        args.extensions = [name]
    elif last:
        args.extensions = [x for x in args.extensions if x != name]
        args.extensions.append(name)
    elif name not in args.extensions:
        args.extensions.append(name)


def is_chat():
    return True


def load_user_config():
    '''
    Loads custom model-specific settings
    '''
    if Path(f'{args.model_dir}/config-user.yaml').exists():
        file_content = open(f'{args.model_dir}/config-user.yaml', 'r').read().strip()

        if file_content:
            user_config = yaml.safe_load(file_content)
        else:
            user_config = {}
    else:
        user_config = {}

    return user_config


args.loader = fix_loader_name(args.loader)

# Activate the multimodal extension
if args.multimodal_pipeline is not None:
    add_extension('multimodal')

# Activate the API extension
if args.api or args.public_api:
    add_extension('openai', last=True)

# Load model-specific settings
with Path(f'{args.model_dir}/config.yaml') as p:
    if p.exists():
        model_config = yaml.safe_load(open(p, 'r').read())
    else:
        model_config = {}

# Load custom model-specific settings
user_config = load_user_config()

model_config = OrderedDict(model_config)
user_config = OrderedDict(user_config)
