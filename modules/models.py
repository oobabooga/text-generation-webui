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
    from flexgen.flex_opt import CompressionConfig, ExecutionEnv, OptLM, Policy

if shared.args.deepspeed:
    import deepspeed
    from transformers.deepspeed import HfDeepSpeedConfig, is_deepspeed_zero3_enabled

    from modules.deepspeed_parameters import generate_ds_config

    # Distributed setup
    local_rank = (
        shared.args.local_rank
        if shared.args.local_rank is not None
        else int(os.getenv("LOCAL_RANK", "0"))
    )
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()
    ds_config = generate_ds_config(
        shared.args.bf16, 1 * world_size, shared.args.nvme_offload_dir
    )
    dschf = HfDeepSpeedConfig(
        ds_config
    )  # Keep this object alive for the Transformers integration


def load_RMVK_tokenizer():
    from modules.RWKV import RWKVTokenizer

    tokenizer = RWKVTokenizer.from_pretrained(Path("models"))

    return tokenizer


def load_hf_tokenizer(model_name):
    is_gpt_4chan = shared.model_name.lower().startswith(
        ("gpt4chan", "gpt-4chan", "4chan")
    )
    is_gpt_j_6B_available = Path("models/gpt-j-6B/").exists()

    if is_gpt_4chan and is_gpt_j_6B_available:
        tokenizer_path = Path("models/gpt-j-6B/")
    else:
        tokenizer_path = Path(f"models/{model_name}/")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if not (is_gpt_4chan and is_gpt_j_6B_available):
        tokenizer.truncation_side = "left"

    return tokenizer


def load_tokenizer(model_name):
    if shared.is_RWKV:
        return load_RMVK_tokenizer()
    else:
        return load_hf_tokenizer(model_name)


def load_deepspeed_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        Path(f"models/{model_name}"),
        torch_dtype=torch.bfloat16 if shared.args.bf16 else torch.float16,
    )
    model = deepspeed.initialize(
        model=model,
        config_params=ds_config,
        model_parameters=None,
        optimizer=None,
        lr_scheduler=None,
    )[0]
    model.module.eval()  # Inference
    print(f"DeepSpeed ZeRO-3 is enabled: {is_deepspeed_zero3_enabled()}")

    return model


def load_flexgen_model(model_name):
    # Initialize environment
    env = ExecutionEnv.create(shared.args.disk_cache_dir)

    # Offloading policy
    policy = Policy(
        1,
        1,
        shared.args.percent[0],
        shared.args.percent[1],
        shared.args.percent[2],
        shared.args.percent[3],
        shared.args.percent[4],
        shared.args.percent[5],
        overlap=True,
        sep_layer=True,
        pin_weight=shared.args.pin_weight,
        cpu_cache_compute=False,
        attn_sparsity=1.0,
        compress_weight=shared.args.compress_weight,
        comp_weight_config=CompressionConfig(
            num_bits=4, group_size=64, group_dim=0, symmetric=False
        ),
        compress_cache=False,
        comp_cache_config=CompressionConfig(
            num_bits=4, group_size=64, group_dim=2, symmetric=False
        ),
    )

    model = OptLM(f"facebook/{model_name}", env, "models", policy)

    return model


def load_RWKV_model(model_name):
    from modules.RWKV import RWKVModel

    model = RWKVModel.from_pretrained(
        Path(f"models/{model_name}"),
        dtype="fp32" if shared.args.cpu else "bf16" if shared.args.bf16 else "fp16",
        device="cpu" if shared.args.cpu else "cuda",
    )

    return model


def load_default_model(model_name):
    model_path = Path(f"models/{model_name}")
    is_large_model = any(size in model_name.lower() for size in ("13b", "20b", "30b"))

    if is_large_model:
        params = {
            "device_map": "auto",
            "load_in_8bit": True,
        }
    else:
        params = {
            "low_cpu_mem_usage": True,
            "torch_dtype": torch.bfloat16 if shared.args.bf16 else torch.float16,
        }

    model = AutoModelForCausalLM.from_pretrained(model_path, **params)
    if not is_large_model:
        model = model.cuda()

    return model


def check_for_GPU():
    if not shared.args.cpu and not torch.cuda.is_available():
        print(
            "Warning: torch.cuda.is_available() returned False.\nThis means that no GPU has been detected.\nFalling back to CPU mode.\n"
        )
    shared.args.cpu = True


def suggested_GPU_memory():
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    suggestion = round((total_mem - 1000) / 1000) * 1000
    if total_mem - suggestion < 800:
        suggestion -= 1000
    suggestion = int(round(suggestion / 1000))
    print(
        f"\033[1;32;1mAuto-assiging --gpu-memory {suggestion} for your GPU to try to prevent out-of-memory errors.\nYou can manually set other values.\033[0;37;0m"
    )

    return suggestion


def load_custom_model(model_name):
    params = {"low_cpu_mem_usage": True}

    check_for_GPU()

    if shared.args.cpu:
        params.update(
            {
                "low_cpu_mem_usage": True,
                "torch_dtype": torch.float32,
            }
        )
    else:
        params.update(
            {
                "device_map": "auto",
            }
        )

        if shared.args.load_in_8bit:
            params.update(
                {
                    "load_in_8bit": True,
                }
            )
        elif shared.args.bf16:
            params.update(
                {
                    "torch_dtype": torch.bfloat16,
                }
            )
        else:
            params.update(
                {
                    "torch_dtype": torch.float16,
                }
            )

        if shared.args.gpu_memory:
            memory_map = shared.args.gpu_memory
            max_memory = {
                gpu_device: f"{memory}GiB"
                for gpu_device, memory in enumerate(memory_map)
            }
            max_memory.update({"cpu": f"{shared.args.cpu_memory or '99'}GiB"})

            params.update({"max_memory": max_memory})

        elif not shared.args.load_in_8bit:
            suggested = suggested_GPU_memory()
            params.update(
                {
                    "max_memory": {
                        0: f"{suggested}GiB",
                        "cpu": f"{shared.args.cpu_memory or '99'}GiB",
                    }
                }
            )

        if shared.args.disk:
            params.update({"offload_folder": shared.args.disk_cache_dir})

    model = AutoModelForCausalLM.from_pretrained(
        Path(f"models/{model_name}"), **params
    )

    return model


def load_model(model_name):
    print(f"Loading {model_name}...")
    t0 = time.time()

    shared.is_RWKV = model_name.lower().startswith("rwkv-")

    defaul_options_selected = not any(
        [
            shared.args.cpu,
            shared.args.load_in_8bit,
            shared.args.gptq_bits,
            shared.args.auto_devices,
            shared.args.disk,
            shared.args.gpu_memory is not None,
            shared.args.cpu_memory is not None,
            shared.args.deepspeed,
            shared.args.flexgen,
            shared.is_RWKV,
        ]
    )

    # RMKV model (not on HuggingFace)
    if shared.is_RWKV:
        model = load_RWKV_model(model_name)

    # Default settings
    elif defaul_options_selected:
        model = load_default_model(model_name)

    # FlexGen
    elif shared.args.flexgen:
        model = load_flexgen_model(model_name)

    # DeepSpeed ZeRO-3
    elif shared.args.deepspeed:
        model = load_deepspeed_model(model_name)

    # Quantized model
    elif shared.args.gptq_bits > 0:
        from modules.GPTQ_loader import load_quantized

        model = load_quantized(model_name)

    # Custom
    else:
        model = load_custom_model(model_name)

    tokenizer = load_tokenizer(model_name)

    print(f"Loaded the model in {(time.time()-t0):.2f} seconds.")
    return model, tokenizer


def load_soft_prompt(name):
    if name == "None":
        shared.soft_prompt = False
        shared.soft_prompt_tensor = None
    else:
        with zipfile.ZipFile(Path(f"softprompts/{name}.zip")) as zf:
            zf.extract("tensor.npy")
            zf.extract("meta.json")
            j = json.loads(open("meta.json", "r").read())
            print(f'\nLoading the softprompt "{name}".')
            for field in j:
                if field != "name":
                    if type(j[field]) is list:
                        print(f"{field}: {', '.join(j[field])}")
                    else:
                        print(f"{field}: {j[field]}")
            print()
            tensor = np.load("tensor.npy")
            Path("tensor.npy").unlink()
            Path("meta.json").unlink()
        tensor = torch.Tensor(tensor).to(
            device=shared.model.device, dtype=shared.model.dtype
        )
        tensor = torch.reshape(tensor, (1, tensor.shape[0], tensor.shape[1]))

        shared.soft_prompt = True
        shared.soft_prompt_tensor = tensor

    return name
