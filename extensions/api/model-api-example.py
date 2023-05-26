#!/usr/bin/env python3

import requests

HOST = '0.0.0.0:5000'

def model_api(request):
    response = requests.post(f'http://{HOST}/api/v1/model', json=request)

    if response.status_code == 200:
        return response.json()['result']


def generate(prompt, tokens = 200):
    request = { 'prompt': prompt, 'max_new_tokens': tokens }
    response = requests.post(f'http://{HOST}/api/v1/generate', json=request)

    if response.status_code == 200:
        return response.json()['results'][0]['text']


def guess_groupsize(model):
    if '1024g' in model:
        return 1024
    elif '128g' in model:
        return 128
    elif '32g' in model:
        return 32
    else:
        return -1

def auto_model_loader_req(model):
    req = {
        'action': 'load',
        'model_name': model,
        'args': {
            'bf16': False,
            'load_in_8bit': False,
            'groupsize': None,
            'wbits': 0,
            'trust_remote_code': False,

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
            #'load_in_4bit': False,
            #'compute_dtype': 'float16',
            #'quant_type': 'nf4',
            #'use_double_quant': False,

            #"cpu": false,
            #"auto_devices": false,
            #"gpu_memory": null,
            #"cpu_memory": null,
            #"disk": false,
            #"disk_cache_dir": "cache",

        },
    }

    model = model.lower()

    if '4bit' in model or 'gptq' in model or 'int4' in model:
        req['args']['wbits'] = 4
        req['args']['groupsize'] = guess_groupsize(model)
    elif '3bit' in model:
        req['args']['wbits'] = 3
        req['args']['groupsize'] = guess_groupsize(model)
    elif '8bit' in model:
        req['args']['load_in_8bit'] = True
    elif '-hf' in model:
        if '7b' in model:
            req['args']['bf16'] = True # for 24GB
        elif '13b' in model:
            req['args']['load_in_8bit'] = True # for 24GB
    elif 'ggml' in model:
        #req['args']['threads'] = 16
        if '7b' in model:
            req['args']['n_gpu_layers'] = 100
        elif '13b' in model:
            req['args']['n_gpu_layers'] = 100
        elif '30b' in model or '33b' in model:
            req['args']['n_gpu_layers'] = 59 # 24GB
        elif '65b' in model:
            req['args']['n_gpu_layers'] = 42 # 24GB
    elif 'rwkv' in model:
        req['args']['rwkv_cuda_on'] = True
        if '14b' in model:
            req['args']['rwkv_strategy'] = 'cuda f16i8' # 24GB
        else:
            req['args']['rwkv_strategy'] = 'cuda f16' # 24GB


    if 'mpt-7b' in model or 'chatglm' in model:
        req['args']['trust_remote_code'] = True

    return req


if __name__ == '__main__':
    for model in model_api({'action': 'list'}):
        req = auto_model_loader_req(model)

        try:
            resp = model_api(req)

            ans = generate("0,1,1,2,3,5,8,13,", tokens=2)

            if '21' in ans:
                print (f"✅ {model} PASS ({ans})")
            else:
                print(model, req, resp)
                print (f"❌ {model} FAIL ({ans})")

        except Exception as e:
            print (f"❌ {model} FAIL Exception: {repr(e)}")
            

# 0,1,1,2,3,5,8,13, is the fibonacci sequence, the next number is 21.
# My results below after removing models that didn't pass.
# Some folders were renamed so the auto-loader worked, Ex: added -hf or -128g, etc.
""" $ ./model-api-example.py 
✅ 4bit_gpt4-x-alpaca-13b-native-4bit-128g-cuda PASS (21)
✅ 4bit_WizardLM-13B-Uncensored-4bit-128g PASS (21)
✅ alpaca-30b-4bit PASS (21)
✅ alpaca-native-4bit PASS (21)
✅ ausboss_llama-13b-supercot-4bit-128g PASS (21)
✅ digitous_Alpacino30b-4bit PASS (21)
✅ eachadea_vicuna-13b-1.1-HF PASS (21)
✅ gpt4-x-alpaca-13b-native-4bit-128g PASS (21)
✅ koala-13B-HF PASS (21)
✅ koala-13B-HF_safetensors PASS (21)
✅ llama-7b PASS (21)
✅ llama-7b-4bit-1g PASS (21)
✅ llama-7b-hf PASS (21)
✅ llama-13b-4bit-128g PASS (21)
✅ llama-13b-hf PASS (21)
✅ llama-30b-4bit-1g PASS (21)
✅ llama-30b-4bit-128g PASS (21)
✅ llama-30b-hf-3bit-32g PASS (21)
✅ llama-30b-hf-3bit-128g PASS (21)
✅ MetaIX_GPT4-X-Alpaca-30B-4bit PASS (21)
✅ MetaIX_Guanaco-33B-4bit PASS (21)
✅ MetaIX_OpenAssistant-Llama-30b-4bit PASS (21)
✅ Monero_Guanaco-13b-Merged-4bit-ts-128g PASS (21)
✅ nomic-ai_gpt4all-13b-snoozy-hf PASS (21)
✅ TheBloke_GPT4All-13B-snoozy-GPTQ-128g PASS (21)
✅ TheBloke_guanaco-33B-GPTQ PASS (21)
✅ TheBloke_koala-13B-GPTQ-4bit-128g PASS (21)
✅ TheBloke_Manticore-13B-GPTQ-128g PASS (21)
✅ TheBloke_OpenAssistant-SFT-7-Llama-30B-GPTQ-1024g PASS (21)
✅ TheBloke_stable-vicuna-13B-GPTQ-128g PASS (21)
✅ TheBloke_vicuna-13B-1.1-GPTQ-4bit-128g PASS (21)
✅ TheBloke_vicuna-13B-1.1-HF PASS (21)
✅ TheBloke_wizard-mega-13B-GPTQ-128g PASS (21)
✅ TheBloke_wizard-vicuna-13B-GPTQ-128g PASS (21)
✅ TheBloke_wizard-vicuna-13B-HF PASS (21)
✅ TheBloke_Wizard-Vicuna-13B-Uncensored-GPTQ-128g PASS (21)
✅ tsumeone_llama-30b-supercot-3bit-128g-cuda PASS (21)
✅ tsumeone_llama-30b-supercot-4bit-cuda PASS (21)
✅ vicuna-13b-GPTQ-4bit-128g PASS (21)
✅ vicuna-13b-hf PASS (21)
"""