import argparse
import copy
import os
import shlex
import sys
from collections import OrderedDict
from pathlib import Path

import yaml

from modules.logging_colors import logger
from modules.paths import resolve_user_data_dir
from modules.presets import default_preset, default_preset_values

# Resolve user_data directory early (before argparse defaults are set)
user_data_dir = resolve_user_data_dir()

# Text model variables
model = None
tokenizer = None
model_name = 'None'
is_seq2seq = False
is_multimodal = False
model_dirty_from_training = False
lora_names = []
bos_token = '<s>'
eos_token = '</s>'

# Image model variables
image_model = None
image_model_name = 'None'
image_pipeline_type = None

# Generation variables
stop_everything = False
generation_lock = None
processing_message = ''

# UI variables
gradio = {}
persistent_interface_state = {}
need_restart = False

# Parser copied from https://github.com/vladmandic/automatic
parser = argparse.ArgumentParser(description="Text Generation Web UI", conflict_handler='resolve', add_help=True, formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=55, indent_increment=2, width=200))

# Basic settings
group = parser.add_argument_group('Basic settings')
group.add_argument('--user-data-dir', type=str, default=str(user_data_dir), help='Path to the user data directory. Default: auto-detected.')
group.add_argument('--multi-user', action='store_true', help='Multi-user mode. Chat histories are not saved or automatically loaded. Best suited for small trusted teams.')
group.add_argument('--model', type=str, help='Name of the model to load by default.')
group.add_argument('--lora', type=str, nargs='+', help='The list of LoRAs to load. If you want to load more than one LoRA, write the names separated by spaces.')
group.add_argument('--model-dir', type=str, default=str(user_data_dir / 'models'), help='Path to directory with all the models.')
group.add_argument('--lora-dir', type=str, default=str(user_data_dir / 'loras'), help='Path to directory with all the loras.')
group.add_argument('--model-menu', action='store_true', help='Show a model menu in the terminal when the web UI is first launched.')
group.add_argument('--settings', type=str, help='Load the default interface settings from this yaml file. See user_data/settings-template.yaml for an example. If you create a file called user_data/settings.yaml, this file will be loaded by default without the need to use the --settings flag.')
group.add_argument('--extensions', type=str, nargs='+', help='The list of extensions to load. If you want to load more than one extension, write the names separated by spaces.')
group.add_argument('--verbose', action='store_true', help='Print the prompts to the terminal.')
group.add_argument('--idle-timeout', type=int, default=0, help='Unload model after this many minutes of inactivity. It will be automatically reloaded when you try to use it again.')

# Image generation
group = parser.add_argument_group('Image model')
group.add_argument('--image-model', type=str, help='Name of the image model to select on startup (overrides saved setting).')
group.add_argument('--image-model-dir', type=str, default=str(user_data_dir / 'image_models'), help='Path to directory with all the image models.')
group.add_argument('--image-dtype', type=str, default=None, choices=['bfloat16', 'float16'], help='Data type for image model.')
group.add_argument('--image-attn-backend', type=str, default=None, choices=['flash_attention_2', 'sdpa'], help='Attention backend for image model.')
group.add_argument('--image-cpu-offload', action='store_true', help='Enable CPU offloading for image model.')
group.add_argument('--image-compile', action='store_true', help='Compile the image model for faster inference.')
group.add_argument('--image-quant', type=str, default=None,
                   choices=['none', 'bnb-8bit', 'bnb-4bit', 'torchao-int8wo', 'torchao-fp4', 'torchao-float8wo'],
                   help='Quantization method for image model.')

# Model loader
group = parser.add_argument_group('Model loader')
group.add_argument('--loader', type=str, help='Choose the model loader manually, otherwise, it will get autodetected. Valid options: Transformers, llama.cpp, ExLlamav3_HF, ExLlamav3, TensorRT-LLM.')

# Cache
group = parser.add_argument_group('Context and cache')
group.add_argument('--ctx-size', '--n_ctx', '--max_seq_len', type=int, default=0, metavar='N', help='Context size in tokens. 0 = auto for llama.cpp (requires gpu-layers=-1), 8192 for other loaders.')
group.add_argument('--cache-type', '--cache_type', type=str, default='fp16', metavar='N', help='KV cache type; valid options: llama.cpp - fp16, q8_0, q4_0; ExLlamaV3 - fp16, q2 to q8 (can specify k_bits and v_bits separately, e.g. q4_q8).')

# Speculative decoding
group = parser.add_argument_group('Speculative decoding')
group.add_argument('--model-draft', type=str, default=None, help='Path to the draft model for speculative decoding.')
group.add_argument('--draft-max', type=int, default=4, help='Number of tokens to draft for speculative decoding.')
group.add_argument('--gpu-layers-draft', type=int, default=256, help='Number of layers to offload to the GPU for the draft model.')
group.add_argument('--device-draft', type=str, default=None, help='Comma-separated list of devices to use for offloading the draft model. Example: CUDA0,CUDA1')
group.add_argument('--ctx-size-draft', type=int, default=0, help='Size of the prompt context for the draft model. If 0, uses the same as the main model.')
group.add_argument('--spec-type', type=str, default='none', choices=['none', 'ngram-mod', 'ngram-simple', 'ngram-map-k', 'ngram-map-k4v', 'ngram-cache'], help='Draftless speculative decoding type. Recommended: ngram-mod.')
group.add_argument('--spec-ngram-size-n', type=int, default=24, help='N-gram lookup size for ngram speculative decoding.')
group.add_argument('--spec-ngram-size-m', type=int, default=48, help='Draft n-gram size for ngram speculative decoding.')
group.add_argument('--spec-ngram-min-hits', type=int, default=1, help='Minimum n-gram hits for ngram-map speculative decoding.')

# llama.cpp
group = parser.add_argument_group('llama.cpp')
group.add_argument('--gpu-layers', '--n-gpu-layers', type=int, default=-1, metavar='N', help='Number of layers to offload to the GPU. -1 = auto.')
group.add_argument('--cpu-moe', action='store_true', help='Move the experts to the CPU (for MoE models).')
group.add_argument('--mmproj', type=str, default=None, help='Path to the mmproj file for vision models.')
group.add_argument('--streaming-llm', action='store_true', help='Activate StreamingLLM to avoid re-evaluating the entire prompt when old messages are removed.')
group.add_argument('--tensor-split', type=str, default=None, help='Split the model across multiple GPUs. Comma-separated list of proportions. Example: 60,40.')
group.add_argument('--row-split', action='store_true', help='Split the model by rows across GPUs. This may improve multi-gpu performance.')
group.add_argument('--no-mmap', action='store_true', help='Prevent mmap from being used.')
group.add_argument('--mlock', action='store_true', help='Force the system to keep the model in RAM.')
group.add_argument('--no-kv-offload', action='store_true', help='Do not offload the  K, Q, V to the GPU. This saves VRAM but reduces the performance.')
group.add_argument('--batch-size', type=int, default=1024, help='Maximum number of prompt tokens to batch together when calling llama-server. This is the application level batch size.')
group.add_argument('--ubatch-size', type=int, default=1024, help='Maximum number of prompt tokens to batch together when calling llama-server. This is the max physical batch size for computation (device level).')
group.add_argument('--threads', type=int, default=0, help='Number of threads to use.')
group.add_argument('--threads-batch', type=int, default=0, help='Number of threads to use for batches/prompt processing.')
group.add_argument('--numa', action='store_true', help='Activate NUMA task allocation for llama.cpp.')
group.add_argument('--parallel', type=int, default=1, help='Number of parallel request slots. The context size is divided equally among slots. For example, to have 4 slots with 8192 context each, set ctx_size to 32768.')
group.add_argument('--fit-target', type=str, default='512', help='Target VRAM margin per device for auto GPU layers, comma-separated list of values in MiB. A single value is broadcast across all devices.')
group.add_argument('--extra-flags', type=str, default=None, help='Extra flags to pass to llama-server. Format: "flag1=value1,flag2,flag3=value3". Example: "override-tensor=exps=CPU"')

# Transformers/Accelerate
group = parser.add_argument_group('Transformers/Accelerate')
group.add_argument('--cpu', action='store_true', help='Use the CPU to generate text. Warning: Training on CPU is extremely slow.')
group.add_argument('--cpu-memory', type=float, default=0, help='Maximum CPU memory in GiB. Use this for CPU offloading.')
group.add_argument('--disk', action='store_true', help='If the model is too large for your GPU(s) and CPU combined, send the remaining layers to the disk.')
group.add_argument('--disk-cache-dir', type=str, default=str(user_data_dir / 'cache'), help='Directory to save the disk cache to.')
group.add_argument('--load-in-8bit', action='store_true', help='Load the model with 8-bit precision (using bitsandbytes).')
group.add_argument('--bf16', action='store_true', help='Load the model with bfloat16 precision. Requires NVIDIA Ampere GPU.')
group.add_argument('--no-cache', action='store_true', help='Set use_cache to False while generating text. This reduces VRAM usage slightly, but it comes at a performance cost.')
group.add_argument('--trust-remote-code', action='store_true', help='Set trust_remote_code=True while loading the model. Necessary for some models.')
group.add_argument('--force-safetensors', action='store_true', help='Set use_safetensors=True while loading the model. This prevents arbitrary code execution.')
group.add_argument('--no_use_fast', action='store_true', help='Set use_fast=False while loading the tokenizer (it\'s True by default). Use this if you have any problems related to use_fast.')
group.add_argument('--attn-implementation', type=str, default='sdpa', metavar="IMPLEMENTATION", help='Attention implementation. Valid options: sdpa, eager, flash_attention_2.')

# bitsandbytes 4-bit
group = parser.add_argument_group('bitsandbytes 4-bit')
group.add_argument('--load-in-4bit', action='store_true', help='Load the model with 4-bit precision (using bitsandbytes).')
group.add_argument('--use_double_quant', action='store_true', help='use_double_quant for 4-bit.')
group.add_argument('--compute_dtype', type=str, default='float16', help='compute dtype for 4-bit. Valid options: bfloat16, float16, float32.')
group.add_argument('--quant_type', type=str, default='nf4', help='quant_type for 4-bit. Valid options: nf4, fp4.')

# ExLlamaV3
group = parser.add_argument_group('ExLlamaV3')
group.add_argument('--gpu-split', type=str, help='Comma-separated list of VRAM (in GB) to use per GPU device for model layers. Example: 20,7,7.')
group.add_argument('--enable-tp', '--enable_tp', action='store_true', help='Enable Tensor Parallelism (TP) to split the model across GPUs.')
group.add_argument('--tp-backend', type=str, default='native', help='The backend for tensor parallelism. Valid options: native, nccl. Default: native.')
group.add_argument('--cfg-cache', action='store_true', help='Create an additional cache for CFG negative prompts. Necessary to use CFG with that loader.')

# Gradio
group = parser.add_argument_group('Gradio')
group.add_argument('--listen', action='store_true', help='Make the web UI reachable from your local network.')
group.add_argument('--listen-port', type=int, help='The listening port that the server will use.')
group.add_argument('--listen-host', type=str, help='The hostname that the server will use.')
group.add_argument('--share', action='store_true', help='Create a public URL. This is useful for running the web UI on Google Colab or similar.')
group.add_argument('--auto-launch', action='store_true', default=False, help='Open the web UI in the default browser upon launch.')
group.add_argument('--gradio-auth', type=str, help='Set Gradio authentication password in the format "username:password". Multiple credentials can also be supplied with "u1:p1,u2:p2,u3:p3".', default=None)
group.add_argument('--gradio-auth-path', type=str, help='Set the Gradio authentication file path. The file should contain one or more user:password pairs in the same format as above.', default=None)
group.add_argument('--ssl-keyfile', type=str, help='The path to the SSL certificate key file.', default=None)
group.add_argument('--ssl-certfile', type=str, help='The path to the SSL certificate cert file.', default=None)
group.add_argument('--subpath', type=str, help='Customize the subpath for gradio, use with reverse proxy')
group.add_argument('--old-colors', action='store_true', help='Use the legacy Gradio colors, before the December/2024 update.')
group.add_argument('--portable', action='store_true', help='Hide features not available in portable mode like training.')

# API
group = parser.add_argument_group('API')
group.add_argument('--api', action='store_true', help='Enable the API extension.')
group.add_argument('--public-api', action='store_true', help='Create a public URL for the API using Cloudflare.')
group.add_argument('--public-api-id', type=str, help='Tunnel ID for named Cloudflare Tunnel. Use together with public-api option.', default=None)
group.add_argument('--api-port', type=int, default=5000, help='The listening port for the API.')
group.add_argument('--api-key', type=str, default='', help='API authentication key.')
group.add_argument('--admin-key', type=str, default='', help='API authentication key for admin tasks like loading and unloading models. If not set, will be the same as --api-key.')
group.add_argument('--api-enable-ipv6', action='store_true', help='Enable IPv6 for the API')
group.add_argument('--api-disable-ipv4', action='store_true', help='Disable IPv4 for the API')
group.add_argument('--nowebui', action='store_true', help='Do not launch the Gradio UI. Useful for launching the API in standalone mode.')

# API generation defaults
_d = default_preset_values
group = parser.add_argument_group('API generation defaults')
group.add_argument('--temperature', type=float, default=_d['temperature'], metavar='N', help='Temperature')
group.add_argument('--dynatemp-low', type=float, default=_d['dynatemp_low'], metavar='N', help='Dynamic temperature low')
group.add_argument('--dynatemp-high', type=float, default=_d['dynatemp_high'], metavar='N', help='Dynamic temperature high')
group.add_argument('--dynatemp-exponent', type=float, default=_d['dynatemp_exponent'], metavar='N', help='Dynamic temperature exponent')
group.add_argument('--smoothing-factor', type=float, default=_d['smoothing_factor'], metavar='N', help='Smoothing factor')
group.add_argument('--smoothing-curve', type=float, default=_d['smoothing_curve'], metavar='N', help='Smoothing curve')
group.add_argument('--top-p', type=float, default=_d['top_p'], metavar='N', help='Top P')
group.add_argument('--top-k', type=int, default=_d['top_k'], metavar='N', help='Top K')
group.add_argument('--min-p', type=float, default=_d['min_p'], metavar='N', help='Min P')
group.add_argument('--top-n-sigma', type=float, default=_d['top_n_sigma'], metavar='N', help='Top N Sigma')
group.add_argument('--typical-p', type=float, default=_d['typical_p'], metavar='N', help='Typical P')
group.add_argument('--xtc-threshold', type=float, default=_d['xtc_threshold'], metavar='N', help='XTC threshold')
group.add_argument('--xtc-probability', type=float, default=_d['xtc_probability'], metavar='N', help='XTC probability')
group.add_argument('--epsilon-cutoff', type=float, default=_d['epsilon_cutoff'], metavar='N', help='Epsilon cutoff')
group.add_argument('--eta-cutoff', type=float, default=_d['eta_cutoff'], metavar='N', help='Eta cutoff')
group.add_argument('--tfs', type=float, default=_d['tfs'], metavar='N', help='TFS')
group.add_argument('--top-a', type=float, default=_d['top_a'], metavar='N', help='Top A')
group.add_argument('--adaptive-target', type=float, default=_d['adaptive_target'], metavar='N', help='Adaptive target')
group.add_argument('--adaptive-decay', type=float, default=_d['adaptive_decay'], metavar='N', help='Adaptive decay')
group.add_argument('--dry-multiplier', type=float, default=_d['dry_multiplier'], metavar='N', help='DRY multiplier')
group.add_argument('--dry-allowed-length', type=int, default=_d['dry_allowed_length'], metavar='N', help='DRY allowed length')
group.add_argument('--dry-base', type=float, default=_d['dry_base'], metavar='N', help='DRY base')
group.add_argument('--repetition-penalty', type=float, default=_d['repetition_penalty'], metavar='N', help='Repetition penalty')
group.add_argument('--frequency-penalty', type=float, default=_d['frequency_penalty'], metavar='N', help='Frequency penalty')
group.add_argument('--presence-penalty', type=float, default=_d['presence_penalty'], metavar='N', help='Presence penalty')
group.add_argument('--encoder-repetition-penalty', type=float, default=_d['encoder_repetition_penalty'], metavar='N', help='Encoder repetition penalty')
group.add_argument('--no-repeat-ngram-size', type=int, default=_d['no_repeat_ngram_size'], metavar='N', help='No repeat ngram size')
group.add_argument('--repetition-penalty-range', type=int, default=_d['repetition_penalty_range'], metavar='N', help='Repetition penalty range')
group.add_argument('--penalty-alpha', type=float, default=_d['penalty_alpha'], metavar='N', help='Penalty alpha')
group.add_argument('--guidance-scale', type=float, default=_d['guidance_scale'], metavar='N', help='Guidance scale')
group.add_argument('--mirostat-mode', type=int, default=_d['mirostat_mode'], metavar='N', help='Mirostat mode')
group.add_argument('--mirostat-tau', type=float, default=_d['mirostat_tau'], metavar='N', help='Mirostat tau')
group.add_argument('--mirostat-eta', type=float, default=_d['mirostat_eta'], metavar='N', help='Mirostat eta')
group.add_argument('--do-sample', action=argparse.BooleanOptionalAction, default=_d['do_sample'], help='Do sample')
group.add_argument('--dynamic-temperature', action=argparse.BooleanOptionalAction, default=_d['dynamic_temperature'], help='Dynamic temperature')
group.add_argument('--temperature-last', action=argparse.BooleanOptionalAction, default=_d['temperature_last'], help='Temperature last')
group.add_argument('--sampler-priority', type=str, default=_d['sampler_priority'], metavar='N', help='Sampler priority')
group.add_argument('--dry-sequence-breakers', type=str, default=_d['dry_sequence_breakers'], metavar='N', help='DRY sequence breakers')
group.add_argument('--enable-thinking', action=argparse.BooleanOptionalAction, default=True, help='Enable thinking')
group.add_argument('--reasoning-effort', type=str, default='medium', metavar='N', help='Reasoning effort')
group.add_argument('--chat-template-file', type=str, default=None, help='Path to a chat template file (.jinja, .jinja2, or .yaml) to use as the default instruction template for API requests. Overrides the model\'s built-in template.')

# Handle CMD_FLAGS.txt
cmd_flags_path = user_data_dir / "CMD_FLAGS.txt"
if cmd_flags_path.exists():
    with cmd_flags_path.open('r', encoding='utf-8') as f:
        cmd_flags = ' '.join(
            line.strip().rstrip('\\').strip()
            for line in f
            if line.strip().rstrip('\\').strip() and not line.strip().startswith('#')
        )

    if cmd_flags:
        # Command-line takes precedence over CMD_FLAGS.txt
        sys.argv = [sys.argv[0]] + shlex.split(cmd_flags) + sys.argv[1:]


args = parser.parse_args()
user_data_dir = Path(args.user_data_dir)  # Update from parsed args (may differ from pre-parse)
original_args = copy.deepcopy(args)
args_defaults = parser.parse_args([])

# Create a mapping of all argument aliases to their canonical names
alias_to_dest = {}
for action in parser._actions:
    for opt in action.option_strings:
        alias_to_dest[opt.lstrip('-').replace('-', '_')] = action.dest

provided_arguments = []
for arg in sys.argv[1:]:
    arg = arg.lstrip('-').replace('-', '_')
    if arg in alias_to_dest:
        provided_arguments.append(alias_to_dest[arg])
    elif hasattr(args, arg):
        provided_arguments.append(arg)

# Default generation parameters
neutral_samplers = default_preset()

# UI defaults
settings = {
    'show_controls': True,
    'start_with': '',
    'mode': 'instruct',
    'chat_style': 'cai-chat',
    'chat-instruct_command': 'Continue the chat dialogue below. Write a single reply for the character "<|character|>". Reply directly, without starting the reply with the character name.\n\n<|prompt|>',
    'enable_web_search': False,
    'web_search_pages': 3,
    'selected_tools': [],
    'prompt-notebook': '',
    'preset': 'Top-P' if (user_data_dir / 'presets/Top-P.yaml').exists() else None,
    'max_new_tokens': 512,
    'max_new_tokens_min': 1,
    'max_new_tokens_max': 4096,
    'prompt_lookup_num_tokens': 0,
    'max_tokens_second': 0,
    'auto_max_new_tokens': True,
    'ban_eos_token': False,
    'add_bos_token': True,
    'enable_thinking': True,
    'reasoning_effort': 'medium',
    'skip_special_tokens': True,
    'stream': True,
    'static_cache': False,
    'truncation_length': 8192,
    'seed': -1,
    'custom_stopping_strings': '',
    'custom_token_bans': '',
    'negative_prompt': '',
    'dark_theme': True,
    'show_two_notebook_columns': False,
    'paste_to_attachment': False,
    'include_past_attachments': True,

    # Generation parameters - Curve shape
    'temperature': neutral_samplers['temperature'],
    'dynatemp_low': neutral_samplers['dynatemp_low'],
    'dynatemp_high': neutral_samplers['dynatemp_high'],
    'dynatemp_exponent': neutral_samplers['dynatemp_exponent'],
    'smoothing_factor': neutral_samplers['smoothing_factor'],
    'smoothing_curve': neutral_samplers['smoothing_curve'],

    # Generation parameters - Curve cutoff
    'top_p': 0.95,
    'top_k': neutral_samplers['top_k'],
    'min_p': neutral_samplers['min_p'],
    'top_n_sigma': neutral_samplers['top_n_sigma'],
    'typical_p': neutral_samplers['typical_p'],
    'xtc_threshold': neutral_samplers['xtc_threshold'],
    'xtc_probability': neutral_samplers['xtc_probability'],
    'epsilon_cutoff': neutral_samplers['epsilon_cutoff'],
    'eta_cutoff': neutral_samplers['eta_cutoff'],
    'tfs': neutral_samplers['tfs'],
    'top_a': neutral_samplers['top_a'],
    'adaptive_target': neutral_samplers['adaptive_target'],
    'adaptive_decay': neutral_samplers['adaptive_decay'],

    # Generation parameters - Repetition suppression
    'dry_multiplier': neutral_samplers['dry_multiplier'],
    'dry_allowed_length': neutral_samplers['dry_allowed_length'],
    'dry_base': neutral_samplers['dry_base'],
    'repetition_penalty': neutral_samplers['repetition_penalty'],
    'frequency_penalty': neutral_samplers['frequency_penalty'],
    'presence_penalty': neutral_samplers['presence_penalty'],
    'encoder_repetition_penalty': neutral_samplers['encoder_repetition_penalty'],
    'no_repeat_ngram_size': neutral_samplers['no_repeat_ngram_size'],
    'repetition_penalty_range': neutral_samplers['repetition_penalty_range'],

    # Generation parameters - Alternative sampling methods
    'penalty_alpha': neutral_samplers['penalty_alpha'],
    'guidance_scale': neutral_samplers['guidance_scale'],
    'mirostat_mode': neutral_samplers['mirostat_mode'],
    'mirostat_tau': neutral_samplers['mirostat_tau'],
    'mirostat_eta': neutral_samplers['mirostat_eta'],

    # Generation parameters - Other options
    'do_sample': neutral_samplers['do_sample'],
    'dynamic_temperature': neutral_samplers['dynamic_temperature'],
    'temperature_last': neutral_samplers['temperature_last'],
    'sampler_priority': neutral_samplers['sampler_priority'],
    'dry_sequence_breakers': neutral_samplers['dry_sequence_breakers'],
    'grammar_string': '',

    # Character settings
    'character': 'Assistant',
    'user': 'Default',
    'name1': 'You',
    'name2': 'AI',
    'user_bio': '',
    'context': 'The following is a conversation with an AI Large Language Model. The AI has been trained to answer questions, provide recommendations, and help with decision making. The AI follows user requests. The AI thinks outside the box.',
    'greeting': 'How can I help you today?',
    'custom_system_message': '',
    'instruction_template_str': "{%- set ns = namespace(found=false) -%}\n{%- for message in messages -%}\n    {%- if message['role'] == 'system' -%}\n        {%- set ns.found = true -%}\n    {%- endif -%}\n{%- endfor -%}\n{%- if not ns.found -%}\n    {{- '' + 'Below is an instruction that describes a task. Write a response that appropriately completes the request.' + '\\n\\n' -}}\n{%- endif %}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' -%}\n        {{- '' + message['content'] + '\\n\\n' -}}\n    {%- else -%}\n        {%- if message['role'] == 'user' -%}\n            {{-'### Instruction:\\n' + message['content'] + '\\n\\n'-}}\n        {%- else -%}\n            {{-'### Response:\\n' + message['content'] + '\\n\\n' -}}\n        {%- endif -%}\n    {%- endif -%}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n    {{-'### Response:\\n'-}}\n{%- endif -%}",
    'chat_template_str': "{%- for message in messages %}\n    {%- if message['role'] == 'system' -%}\n        {%- if message['content'] -%}\n            {{- message['content'] + '\\n\\n' -}}\n        {%- endif -%}\n        {%- if user_bio -%}\n            {{- user_bio + '\\n\\n' -}}\n        {%- endif -%}\n    {%- elif message['role'] == 'tool' -%}\n        {{- '[Tool result: ' + message['content'] + ']\\n' -}}\n    {%- elif message['role'] == 'user' -%}\n        {{- name1 + ': ' + message['content'] + '\\n'-}}\n    {%- elif message['tool_calls'] is defined and message['tool_calls'] -%}\n        {%- for tc in message['tool_calls'] -%}\n            {{- '[Calling: ' + tc['function']['name'] + '(' + tc['function']['arguments'] + ')]\\n' -}}\n        {%- endfor -%}\n    {%- else -%}\n        {{- name2 + ': ' + message['content'] + '\\n' -}}\n    {%- endif -%}\n{%- endfor -%}\n{%- if add_generation_prompt %}\n    {{- name2 + ':' -}}\n{%- endif %}",

    # Extensions
    'default_extensions': [],

    # Image generation settings
    'image_prompt': '',
    'image_neg_prompt': '',
    'image_width': 1024,
    'image_height': 1024,
    'image_aspect_ratio': '1:1 Square',
    'image_steps': 9,
    'image_cfg_scale': 0.0,
    'image_seed': -1,
    'image_batch_size': 1,
    'image_batch_count': 1,
    'image_llm_variations': False,
    'image_llm_variations_prompt': 'Write a variation of the image generation prompt above. Consider the intent of the user with that prompt and write something that will likely please them, with added details. Output only the new prompt. Do not add any explanations, prefixes, or additional text.',
    'image_model_menu': 'None',
    'image_dtype': 'bfloat16',
    'image_attn_backend': 'flash_attention_2',
    'image_cpu_offload': False,
    'image_compile': False,
    'image_quant': 'none',
}

default_settings = copy.deepcopy(settings)


def do_cmd_flags_warnings():
    # Validate --chat-template-file
    if args.chat_template_file and not Path(args.chat_template_file).is_file():
        logger.error(f"--chat-template-file: file not found: {args.chat_template_file}")
        sys.exit(1)

    # Security warnings
    if args.trust_remote_code:
        logger.warning(
            "The `--trust-remote-code` flag is enabled.\n"
            "This allows models to execute arbitrary code on your machine.\n\n"
            "1. Only use with models from sources you fully trust.\n"
            "2. Set an access password with `--gradio-auth`."
        )

    if 'COLAB_GPU' not in os.environ and not args.nowebui:
        if args.share:
            logger.warning("The gradio \"share link\" feature uses a proprietary executable to create a reverse tunnel. Use it with care.")
        if any((args.listen, args.share)) and not any((args.gradio_auth, args.gradio_auth_path)):
            logger.warning("You are potentially exposing the web UI to the entire internet without any access password.\nYou can create one with the \"--gradio-auth\" flag like this:\n\n--gradio-auth username:password\n\nMake sure to replace username:password with your own.")
    if args.multi_user:
        logger.warning(
            'Multi-user mode is enabled. Known limitations:'
            '\n- The Stop button stops generation for all users, not just you.'
            '\n- Chat history is not saved and will be lost on page refresh.'
            '\n- Only one user can generate at a time unless using a parallel-capable backend (e.g. llama.cpp with --parallel N for N > 1, or ExLlamaV3).'
            '\n\nThis mode works best for small trusted teams.'
            '\n\nDo not expose publicly. Grayed-out actions can easily be bypassed client-side.\n'
        )


def apply_image_model_cli_overrides():
    """Apply command-line overrides for image model settings."""
    if args.image_model is not None:
        settings['image_model_menu'] = args.image_model
    if args.image_dtype is not None:
        settings['image_dtype'] = args.image_dtype
    if args.image_attn_backend is not None:
        settings['image_attn_backend'] = args.image_attn_backend
    if args.image_cpu_offload:
        settings['image_cpu_offload'] = True
    if args.image_compile:
        settings['image_compile'] = True
    if args.image_quant is not None:
        settings['image_quant'] = args.image_quant


def fix_loader_name(name):
    if not name:
        return name

    name = name.lower()
    if name in ['llama.cpp', 'llamacpp', 'llama-cpp', 'llama cpp']:
        return 'llama.cpp'
    elif name in ['transformers', 'huggingface', 'hf', 'hugging_face', 'hugging face']:
        return 'Transformers'
    elif name in ['exllamav3-hf', 'exllamav3_hf', 'exllama-v3-hf', 'exllama_v3_hf', 'exllama-v3_hf', 'exllama3-hf', 'exllama3_hf', 'exllama-3-hf', 'exllama_3_hf', 'exllama-3_hf']:
        return 'ExLlamav3_HF'
    elif name in ['exllamav3']:
        return 'ExLlamav3'
    elif name in ['tensorrt', 'tensorrtllm', 'tensorrt_llm', 'tensorrt-llm', 'tensort', 'tensortllm']:
        return 'TensorRT-LLM'


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

# Activate the API extension
if args.api or args.public_api:
    add_extension('openai', last=True)

# Load model-specific settings
p = Path(f'{args.model_dir}/config.yaml')
if p.exists():
    model_config = yaml.safe_load(open(p, 'r').read())
else:
    model_config = {}
del p


# Load custom model-specific settings
user_config = load_user_config()

model_config = OrderedDict(model_config)
user_config = OrderedDict(user_config)
