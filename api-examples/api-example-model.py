#!/usr/bin/env python3

import requests

HOST = '0.0.0.0:5000'

def model_api(request):
    response = requests.post(f'http://{HOST}/api/v1/model', json=request)

    return response.json()


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
            'autogptq': True,
            # autogptq or gptq-for-llama should be True, not both. gptq-for-llama is still slightly more compatible, but the default is autogptq.
            'gptq-for-llama': False, 

            'bf16': False,
            'load_in_8bit': False,
            'groupsize': 0,
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
    else:
        req['args']['autogptq'] = False
        req['args']['gptq-for-llama'] = False

    if '8bit' in model:
        req['args']['load_in_8bit'] = True
    elif '-hf' in model or 'fp16' in model:
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
    for model in model_api({'action': 'list'})['result']:
        req = auto_model_loader_req(model)

        try:
            resp = model_api(req)

            if 'error' in resp:
                print (f"❌ {model} FAIL Error: {resp['error']['message']}")
                continue

            ans = generate("0,1,1,2,3,5,8,13,", tokens=2)

            if '21' in ans:
                print (f"✅ {model} PASS ({ans})")
            else:
                print(model, req, resp)
                print (f"❌ {model} FAIL ({ans})")

        except Exception as e:
            print (f"❌ {model} FAIL Exception: {repr(e)}")
            

# 0,1,1,2,3,5,8,13, is the fibonacci sequence, the next number is 21.
# Some results below.
# Some folders were renamed so the auto-loader worked, Ex: added -hf or -128g, etc.
""" $ ./model-api-example.py 
✅ eachadea_vicuna-13b-1.1-HF PASS (21)
✅ gpt4-x-alpaca-13b-native-4bit-128g PASS (21)
✅ koala-13B-HF PASS (21)
✅ llama-7b-4bit-1g PASS (21)
❌ llama-7b-hf FAIL Error: FileNotFoundError(2, 'No such file or directory')
"""