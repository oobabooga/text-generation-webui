import argparse

model = None
tokenizer = None
model_name = "None"
lora_name = "None"
soft_prompt_tensor = None
soft_prompt = False
is_RWKV = False

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
    'name1': 'Person 1',
    'name2': 'Person 2',
    'context': 'This is a conversation between two people.',
    'stop_at_newline': False,
    'chat_prompt_size': 2048,
    'chat_prompt_size_min': 0,
    'chat_prompt_size_max': 2048,
    'chat_generation_attempts': 1,
    'chat_generation_attempts_min': 1,
    'chat_generation_attempts_max': 5,
    'name1_pygmalion': 'You',
    'name2_pygmalion': 'Kawaii',
    'context_pygmalion': "Kawaii's persona: Kawaii is a cheerful person who loves to make others smile. She is an optimist who loves to spread happiness and positivity wherever she goes.\n<START>",
    'stop_at_newline_pygmalion': False,
    'default_extensions': [],
    'chat_default_extensions': ["gallery"],
    'presets': {
        'default': 'NovelAI-Sphinx Moth',
        'pygmalion-*': 'Pygmalion',
        'RWKV-*': 'Naive',
    },
    'prompts': {
        'default': 'Common sense questions and answers\n\nQuestion: \nFactual answer:',
        '^(gpt4chan|gpt-4chan|4chan)': '-----\n--- 865467536\nInput text\n--- 865467537\n',
        '(rosey|chip|joi)_.*_instruct.*': 'User: \n',
        'oasst-*': '<|prompter|>Write a story about future of AI development<|endoftext|><|assistant|>'
    },
    'lora_prompts': {
        'default': 'Common sense questions and answers\n\nQuestion: \nFactual answer:',
        'alpaca-lora-7b': "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n### Instruction:\nWrite a poem about the transformers Python library. \nMention the word \"large language models\" in that poem.\n### Response:\n"
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

parser = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog,max_help_position=54))
parser.add_argument('--model', type=str, help='Name of the model to load by default.')
parser.add_argument('--lora', type=str, help='Name of the LoRA to apply to the model by default.')
parser.add_argument('--notebook', action='store_true', help='Launch the web UI in notebook mode, where the output is written to the same text box as the input.')
parser.add_argument('--chat', action='store_true', help='Launch the web UI in chat mode.')
parser.add_argument('--cai-chat', action='store_true', help='Launch the web UI in chat mode with a style similar to Character.AI\'s. If the file img_bot.png or img_bot.jpg exists in the same folder as server.py, this image will be used as the bot\'s profile picture. Similarly, img_me.png or img_me.jpg will be used as your profile picture.')
parser.add_argument('--cpu', action='store_true', help='Use the CPU to generate text.')
parser.add_argument('--load-in-8bit', action='store_true', help='Load the model with 8-bit precision.')
parser.add_argument('--load-in-4bit', action='store_true', help='DEPRECATED: use --gptq-bits 4 instead.')
parser.add_argument('--gptq-bits', type=int, default=0, help='Load a pre-quantized model with specified precision. 2, 3, 4 and 8bit are supported. Currently only works with LLaMA and OPT.')
parser.add_argument('--gptq-model-type', type=str, help='Model type of pre-quantized model. Currently only LLaMa and OPT are supported.')
parser.add_argument('--bf16', action='store_true', help='Load the model with bfloat16 precision. Requires NVIDIA Ampere GPU.')
parser.add_argument('--auto-devices', action='store_true', help='Automatically split the model across the available GPU(s) and CPU.')
parser.add_argument('--disk', action='store_true', help='If the model is too large for your GPU(s) and CPU combined, send the remaining layers to the disk.')
parser.add_argument('--disk-cache-dir', type=str, default="cache", help='Directory to save the disk cache to. Defaults to "cache".')
parser.add_argument('--gpu-memory', type=int, nargs="+", help='Maxmimum GPU memory in GiB to be allocated per GPU. Example: --gpu-memory 10 for a single GPU, --gpu-memory 10 5 for two GPUs.')
parser.add_argument('--cpu-memory', type=int, help='Maximum CPU memory in GiB to allocate for offloaded weights. Must be an integer number. Defaults to 99.')
parser.add_argument('--flexgen', action='store_true', help='Enable the use of FlexGen offloading.')
parser.add_argument('--percent', type=int, nargs="+", default=[0, 100, 100, 0, 100, 0], help='FlexGen: allocation percentages. Must be 6 numbers separated by spaces (default: 0, 100, 100, 0, 100, 0).')
parser.add_argument("--compress-weight", action="store_true", help="FlexGen: activate weight compression.")
parser.add_argument("--pin-weight", type=str2bool, nargs="?", const=True, default=True, help="FlexGen: whether to pin weights (setting this to False reduces CPU memory by 20%%).")
parser.add_argument('--deepspeed', action='store_true', help='Enable the use of DeepSpeed ZeRO-3 for inference via the Transformers integration.')
parser.add_argument('--nvme-offload-dir', type=str, help='DeepSpeed: Directory to use for ZeRO-3 NVME offloading.')
parser.add_argument('--local_rank', type=int, default=0, help='DeepSpeed: Optional argument for distributed setups.')
parser.add_argument('--rwkv-strategy', type=str, default=None, help='RWKV: The strategy to use while loading the model. Examples: "cpu fp32", "cuda fp16", "cuda fp16i8".')
parser.add_argument('--rwkv-cuda-on', action='store_true', help='RWKV: Compile the CUDA kernel for better performance.')
parser.add_argument('--no-stream', action='store_true', help='Don\'t stream the text output in real time.')
parser.add_argument('--settings', type=str, help='Load the default interface settings from this json file. See settings-template.json for an example. If you create a file called settings.json, this file will be loaded by default without the need to use the --settings flag.')
parser.add_argument('--extensions', type=str, nargs="+", help='The list of extensions to load. If you want to load more than one extension, write the names separated by spaces.')
parser.add_argument('--listen', action='store_true', help='Make the web UI reachable from your local network.')
parser.add_argument('--listen-port', type=int, help='The listening port that the server will use.')
parser.add_argument('--share', action='store_true', help='Create a public URL. This is useful for running the web UI on Google Colab or similar.')
parser.add_argument('--auto-launch', action='store_true', default=False, help='Open the web UI in the default browser upon launch.')
parser.add_argument('--verbose', action='store_true', help='Print the prompts to the terminal.')
args = parser.parse_args()

# Provisional, this will be deleted later
if args.load_in_4bit:
    print("Warning: --load-in-4bit is deprecated and will be removed. Use --gptq-bits 4 instead.\n")
    args.gptq_bits = 4
