#!/usr/bin/env python3

import requests

HOST = '0.0.0.0:5000'


def generate(prompt, tokens=200):
    request = {'prompt': prompt, 'max_new_tokens': tokens}
    response = requests.post(f'http://{HOST}/api/v1/generate', json=request)

    if response.status_code == 200:
        return response.json()['results'][0]['text']


def model_api(request):
    response = requests.post(f'http://{HOST}/api/v1/model', json=request)
    return response.json()


# print some common settings
def print_basic_model_info(response):
    basic_settings = ['truncation_length', 'instruction_template']
    print("Model: ", response['result']['model_name'])
    print("Lora(s): ", response['result']['lora_names'])
    for setting in basic_settings:
        print(setting, "=", response['result']['shared.settings'][setting])


# model info
def model_info():
    response = model_api({'action': 'info'})
    print_basic_model_info(response)


# simple loader
def model_load(model_name):
    return model_api({'action': 'load', 'model_name': model_name})


# complex loader
def complex_model_load(model):

    def guess_groupsize(model_name):
        if '1024g' in model_name:
            return 1024
        elif '128g' in model_name:
            return 128
        elif '32g' in model_name:
            return 32
        else:
            return -1

    req = {
        'action': 'load',
        'model_name': model,
        'args': {
            'loader': 'AutoGPTQ',

            'bf16': False,
            'load_in_8bit': False,
            'groupsize': 0,
            'wbits': 0,

            # llama.cpp
            'threads': 0,
            'n_batch': 512,
            'no_mmap': False,
            'mlock': False,
            'cache_capacity': None,
            'n_gpu_layers': 0,
            'n_ctx': 2048,

            # RWKV
            'rwkv_strategy': None,
            'rwkv_cuda_on': False,

            # b&b 4-bit
            # 'load_in_4bit': False,
            # 'compute_dtype': 'float16',
            # 'quant_type': 'nf4',
            # 'use_double_quant': False,

            # "cpu": false,
            # "auto_devices": false,
            # "gpu_memory": null,
            # "cpu_memory": null,
            # "disk": false,
            # "disk_cache_dir": "cache",
        },
    }

    model = model.lower()

    if '4bit' in model or 'gptq' in model or 'int4' in model:
        req['args']['wbits'] = 4
        req['args']['groupsize'] = guess_groupsize(model)
    elif '3bit' in model:
        req['args']['wbits'] = 3
        req['args']['groupsize'] = guess_groupsize(model)
    else:
        req['args']['gptq_for_llama'] = False

    if '8bit' in model:
        req['args']['load_in_8bit'] = True
    elif '-hf' in model or 'fp16' in model:
        if '7b' in model:
            req['args']['bf16'] = True  # for 24GB
        elif '13b' in model:
            req['args']['load_in_8bit'] = True  # for 24GB
    elif 'gguf' in model:
        # req['args']['threads'] = 16
        if '7b' in model:
            req['args']['n_gpu_layers'] = 100
        elif '13b' in model:
            req['args']['n_gpu_layers'] = 100
        elif '30b' in model or '33b' in model:
            req['args']['n_gpu_layers'] = 59  # 24GB
        elif '65b' in model:
            req['args']['n_gpu_layers'] = 42  # 24GB
    elif 'rwkv' in model:
        req['args']['rwkv_cuda_on'] = True
        if '14b' in model:
            req['args']['rwkv_strategy'] = 'cuda f16i8'  # 24GB
        else:
            req['args']['rwkv_strategy'] = 'cuda f16'  # 24GB

    return model_api(req)


if __name__ == '__main__':
    for model in model_api({'action': 'list'})['result']:
        try:
            resp = complex_model_load(model)

            if 'error' in resp:
                print(f"❌ {model} FAIL Error: {resp['error']['message']}")
                continue
            else:
                print_basic_model_info(resp)

            ans = generate("0,1,1,2,3,5,8,13,", tokens=2)

            if '21' in ans:
                print(f"✅ {model} PASS ({ans})")
            else:
                print(f"❌ {model} FAIL ({ans})")

        except Exception as e:
            print(f"❌ {model} FAIL Exception: {repr(e)}")


# 0,1,1,2,3,5,8,13, is the fibonacci sequence, the next number is 21.
# Some results below.
""" $ ./model-api-example.py
Model:  4bit_gpt4-x-alpaca-13b-native-4bit-128g-cuda
Lora(s):  []
truncation_length = 2048
instruction_template = Alpaca
✅ 4bit_gpt4-x-alpaca-13b-native-4bit-128g-cuda PASS (21)
Model:  4bit_WizardLM-13B-Uncensored-4bit-128g
Lora(s):  []
truncation_length = 2048
instruction_template = WizardLM
✅ 4bit_WizardLM-13B-Uncensored-4bit-128g PASS (21)
Model:  Aeala_VicUnlocked-alpaca-30b-4bit
Lora(s):  []
truncation_length = 2048
instruction_template = Alpaca
✅ Aeala_VicUnlocked-alpaca-30b-4bit PASS (21)
Model:  alpaca-30b-4bit
Lora(s):  []
truncation_length = 2048
instruction_template = Alpaca
✅ alpaca-30b-4bit PASS (21)
"""
