# Copied from https://github.com/johnsmith0031/alpaca_lora_4bit

from pathlib import Path

import alpaca_lora_4bit.autograd_4bit as autograd_4bit
from alpaca_lora_4bit.amp_wrapper import AMPWrapper
from alpaca_lora_4bit.autograd_4bit import (
    Autograd4bitQuantLinear,
    load_llama_model_4bit_low_ram
)
from alpaca_lora_4bit.models import Linear4bitLt
from alpaca_lora_4bit.monkeypatch.peft_tuners_lora_monkey_patch import (
    replace_peft_model_with_int4_lora_model
)

from modules import shared
from modules.GPTQ_loader import find_quantized_model_file

replace_peft_model_with_int4_lora_model()


def load_model_llama(model_name):
    config_path = str(Path(f'{shared.args.model_dir}/{model_name}'))
    model_path = str(find_quantized_model_file(model_name))
    model, tokenizer = load_llama_model_4bit_low_ram(config_path, model_path, groupsize=shared.args.groupsize, is_v1_model=False)
    for _, m in model.named_modules():
        if isinstance(m, Autograd4bitQuantLinear) or isinstance(m, Linear4bitLt):
            if m.is_v1_model:
                m.zeros = m.zeros.half()
            m.scales = m.scales.half()
            m.bias = m.bias.half()

    autograd_4bit.auto_switch = True

    model.half()
    wrapper = AMPWrapper(model)
    wrapper.apply_generate()

    return model, tokenizer
