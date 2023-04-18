# Copied from https://github.com/johnsmith0031/alpaca_lora_4bit

import sys
from pathlib import Path

from alpaca_lora_4bit.autograd_4bit import (Autograd4bitQuantLinear,
                             load_llama_model_4bit_low_ram,
                             find_layers, make_quant_for_4bit_autograd)
from alpaca_lora_4bit.monkeypatch.peft_tuners_lora_monkey_patch import replace_peft_model_with_int4_lora_model
from alpaca_lora_4bit.models import Linear4bitLt

import time
import accelerate
from colorama import init, Fore, Back, Style
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub.utils._validators import HFValidationError

from modules import shared
from modules.GPTQ_loader_utils import find_quantized_model_file

replace_peft_model_with_int4_lora_model()


def load_model_4bit_gptq(model_name):
    config_path = str(Path(f'{shared.args.model_dir}/{model_name}'))
    model_path = str(find_quantized_model_file(model_name))

    model, tokenizer = load_model_4bit_low_ram(config_path, model_path, groupsize=shared.args.groupsize, is_v1_model=shared.args.is_v1_model)

    # check if model is llama
    if 'llama' in str(type(model)).lower():
        print(Style.BRIGHT + Fore.CYAN + "Model is Llama. Applying Llama specific patches ...")
        try:
            tokenizer.eos_token_id = 2
            tokenizer.bos_token_id = 1
            tokenizer.pad_token_id = 0
        except:
            pass

    return model, tokenizer


def load_model_4bit_low_ram(config_path, model_path, groupsize=-1, device_map="auto", is_v1_model=False):

    print(Style.BRIGHT + Fore.CYAN + "Loading Model ...")
    t0 = time.time()

    with accelerate.init_empty_weights():
        config = AutoConfig.from_pretrained(config_path)
        model = AutoModelForCausalLM.from_config(config)
        model = model.eval()
        layers = find_layers(model)
        for name in ['lm_head']:
            if name in layers:
                del layers[name]
        make_quant_for_4bit_autograd(model, layers, groupsize=groupsize, is_v1_model=is_v1_model)
    model = accelerate.load_checkpoint_and_dispatch(
        model=model,
        checkpoint=model_path,
        device_map=device_map
    )

    for n, m in model.named_modules():
        if isinstance(m, Autograd4bitQuantLinear):
            if m.is_v1_model:
                m.zeros = m.zeros.half()
            m.scales = m.scales.half()
            m.bias = m.bias.half()

    try:
        tokenizer = AutoTokenizer.from_pretrained(config_path)
    except HFValidationError as e:
        tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.truncation_side = 'left'

    print(Style.BRIGHT + Fore.GREEN + f"Loaded the model in {(time.time()-t0):.2f} seconds.")

    return model, tokenizer
