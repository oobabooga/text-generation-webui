import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

import modules.shared as shared

transformers.logging.set_verbosity_error()

local_rank = None

if shared.args.flexgen:
    from flexgen.flex_opt import (CompressionConfig, ExecutionEnv, OptLM,
                                  Policy, str2bool)

if shared.args.deepspeed:
    import deepspeed
    from transformers.deepspeed import (HfDeepSpeedConfig,
                                        is_deepspeed_zero3_enabled)

    from modules.deepspeed_parameters import generate_ds_config

    # Distributed setup
    local_rank = shared.args.local_rank if shared.args.local_rank is not None else int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()
    ds_config = generate_ds_config(shared.args.bf16, 1 * world_size, shared.args.nvme_offload_dir)
    dschf = HfDeepSpeedConfig(ds_config) # Keep this object alive for the Transformers integration


def load_model(model_name):
    print(f"Loading {model_name}...")
    t0 = time.time()

    shared.is_RWKV = model_name.lower().startswith('rwkv-')

    # Default settings
    if not any([shared.args.cpu, shared.args.load_in_8bit, shared.args.gptq_bits, shared.args.auto_devices, shared.args.disk, shared.args.gpu_memory is not None, shared.args.cpu_memory is not None, shared.args.deepspeed, shared.args.flexgen, shared.is_RWKV]):
        if any(size in shared.model_name.lower() for size in ('13b', '20b', '30b')):
            model = AutoModelForCausalLM.from_pretrained(Path(f"models/{shared.model_name}"), device_map='auto', load_in_8bit=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(Path(f"models/{shared.model_name}"), low_cpu_mem_usage=True, torch_dtype=torch.bfloat16 if shared.args.bf16 else torch.float16).cuda()

    # FlexGen
    elif shared.args.flexgen:
        # Initialize environment
        env = ExecutionEnv.create(shared.args.disk_cache_dir)

        # Offloading policy
        policy = Policy(1, 1,
                        shared.args.percent[0], shared.args.percent[1],
                        shared.args.percent[2], shared.args.percent[3],
                        shared.args.percent[4], shared.args.percent[5],
                        overlap=True, sep_layer=True, pin_weight=shared.args.pin_weight,
                        cpu_cache_compute=False, attn_sparsity=1.0,
                        compress_weight=shared.args.compress_weight,
                        comp_weight_config=CompressionConfig(
                            num_bits=4, group_size=64,
                            group_dim=0, symmetric=False),
                        compress_cache=False,
                        comp_cache_config=CompressionConfig(
                            num_bits=4, group_size=64,
                            group_dim=2, symmetric=False))

        model = OptLM(f"facebook/{shared.model_name}", env, "models", policy)

    # DeepSpeed ZeRO-3
    elif shared.args.deepspeed:
        model = AutoModelForCausalLM.from_pretrained(Path(f"models/{shared.model_name}"), torch_dtype=torch.bfloat16 if shared.args.bf16 else torch.float16)
        model = deepspeed.initialize(model=model, config_params=ds_config, model_parameters=None, optimizer=None, lr_scheduler=None)[0]
        model.module.eval() # Inference
        print(f"DeepSpeed ZeRO-3 is enabled: {is_deepspeed_zero3_enabled()}")

    # RMKV model (not on HuggingFace)
    elif shared.is_RWKV:
        from modules.RWKV import RWKVModel, RWKVTokenizer

        model = RWKVModel.from_pretrained(Path(f'models/{model_name}'), dtype="fp32" if shared.args.cpu else "bf16" if shared.args.bf16 else "fp16", device="cpu" if shared.args.cpu else "cuda")
        tokenizer = RWKVTokenizer.from_pretrained(Path('models'))

        return model, tokenizer

    # Quantized model
    elif shared.args.gptq_bits > 0:
        from modules.GPTQ_loader import load_quantized

        model = load_quantized(model_name)

    # Custom
    else:
        command = "AutoModelForCausalLM.from_pretrained"
        params = ["low_cpu_mem_usage=True"]
        if not shared.args.cpu and not torch.cuda.is_available():
            print("Warning: no GPU has been detected.\nFalling back to CPU mode.\n")
            shared.args.cpu = True

        if shared.args.cpu:
            params.append("low_cpu_mem_usage=True")
            params.append("torch_dtype=torch.float32")
        else:
            params.append("device_map='auto'")
            params.append("load_in_8bit=True" if shared.args.load_in_8bit else "torch_dtype=torch.bfloat16" if shared.args.bf16 else "torch_dtype=torch.float16")

            if shared.args.gpu_memory:
                memory_map = shared.args.gpu_memory
                max_memory = f"max_memory={{0: '{memory_map[0]}GiB'"
                for i in range(1, len(memory_map)):
                    max_memory += (f", {i}: '{memory_map[i]}GiB'")
                max_memory += (f", 'cpu': '{shared.args.cpu_memory or '99'}GiB'}}")
                params.append(max_memory)
            elif not shared.args.load_in_8bit:
                total_mem = (torch.cuda.get_device_properties(0).total_memory/(1024*1024))
                suggestion = round((total_mem-1000)/1000)*1000
                if total_mem-suggestion < 800:
                    suggestion -= 1000
                suggestion = int(round(suggestion/1000))
                print(f"\033[1;32;1mAuto-assiging --gpu-memory {suggestion} for your GPU to try to prevent out-of-memory errors.\nYou can manually set other values.\033[0;37;0m")
                params.append(f"max_memory={{0: '{suggestion}GiB', 'cpu': '{shared.args.cpu_memory or '99'}GiB'}}")
            if shared.args.disk:
                params.append(f"offload_folder='{shared.args.disk_cache_dir}'")

        command = f"{command}(Path(f'models/{shared.model_name}'), {', '.join(set(params))})"
        model = eval(command)

    # Loading the tokenizer
    if shared.model_name.lower().startswith(('gpt4chan', 'gpt-4chan', '4chan')) and Path("models/gpt-j-6B/").exists():
        tokenizer = AutoTokenizer.from_pretrained(Path("models/gpt-j-6B/"))
    else:
        tokenizer = AutoTokenizer.from_pretrained(Path(f"models/{shared.model_name}/"))
    tokenizer.truncation_side = 'left'

    print(f"Loaded the model in {(time.time()-t0):.2f} seconds.")
    return model, tokenizer

def load_soft_prompt(name):
    if name == 'None':
        shared.soft_prompt = False
        shared.soft_prompt_tensor = None
    else:
        with zipfile.ZipFile(Path(f'softprompts/{name}.zip')) as zf:
            zf.extract('tensor.npy')
            zf.extract('meta.json')
            j = json.loads(open('meta.json', 'r').read())
            print(f"\nLoading the softprompt \"{name}\".")
            for field in j:
                if field != 'name':
                    if type(j[field]) is list:
                        print(f"{field}: {', '.join(j[field])}")
                    else:
                        print(f"{field}: {j[field]}")
            print()
            tensor = np.load('tensor.npy')
            Path('tensor.npy').unlink()
            Path('meta.json').unlink()
        tensor = torch.Tensor(tensor).to(device=shared.model.device, dtype=shared.model.dtype)
        tensor = torch.reshape(tensor, (1, tensor.shape[0], tensor.shape[1]))

        shared.soft_prompt = True
        shared.soft_prompt_tensor = tensor

    return name
