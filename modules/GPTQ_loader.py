import inspect
import re
import sys
from pathlib import Path

import accelerate
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM

import modules.shared as shared

sys.path.insert(0, str(Path("repositories/GPTQ-for-LLaMa")))
import llama_inference_offload
from modelutils import find_layers
from quant import make_quant


def _load_quant(model, checkpoint, wbits, groupsize=-1, faster_kernel=False, exclude_layers=['lm_head'], kernel_switch_threshold=128):

    def noop(*args, **kwargs):
        pass

    config = AutoConfig.from_pretrained(model)
    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = AutoModelForCausalLM.from_config(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in exclude_layers:
        if name in layers:
            del layers[name]

    gptq_args = inspect.getfullargspec(make_quant).args

    make_quant_kwargs = {
        'module': model,
        'names': layers,
        'bits': wbits,
    }
    if 'groupsize' in gptq_args:
        make_quant_kwargs['groupsize'] = groupsize
    if 'faster' in gptq_args:
        make_quant_kwargs['faster'] = faster_kernel
    if 'kernel_switch_threshold' in gptq_args:
        make_quant_kwargs['kernel_switch_threshold'] = kernel_switch_threshold

    make_quant(**make_quant_kwargs)

    del layers

    print('Loading model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint), strict=False)
    else:
        model.load_state_dict(torch.load(checkpoint), strict=False)
    model.seqlen = 2048
    print('Done.')

    return model


def load_quantized(model_name):
    if not shared.args.model_type:
        # Try to determine model type from model name
        name = model_name.lower()
        if any((k in name for k in ['llama', 'alpaca', 'vicuna'])):
            model_type = 'llama'
        elif any((k in name for k in ['opt-', 'galactica'])):
            model_type = 'opt'
        elif any((k in name for k in ['gpt-j', 'pygmalion-6b'])):
            model_type = 'gptj'
        else:
            print("Can't determine model type from model name. Please specify it manually using --model_type "
                  "argument")
            exit()
    else:
        model_type = shared.args.model_type.lower()

    if shared.args.pre_layer and model_type == 'llama':
        load_quant = llama_inference_offload.load_quant
    elif model_type in ('llama', 'opt', 'gptj'):
        if shared.args.pre_layer:
            print("Warning: ignoring --pre_layer because it only works for llama model type.")
        load_quant = _load_quant
    else:
        print("Unknown pre-quantized model type specified. Only 'llama', 'opt' and 'gptj' are supported")
        exit()

    # Now we are going to try to locate the quantized model file.
    path_to_model = Path(f'{shared.args.model_dir}/{model_name}')
    found_pts = list(path_to_model.glob("*.pt"))
    found_safetensors = list(path_to_model.glob("*.safetensors"))
    pt_path = None

    if len(found_pts) == 1:
        pt_path = found_pts[0]
    elif len(found_safetensors) == 1:
        pt_path = found_safetensors[0]
    else:
        if path_to_model.name.lower().startswith('llama-7b'):
            pt_model = f'llama-7b-{shared.args.wbits}bit'
        elif path_to_model.name.lower().startswith('llama-13b'):
            pt_model = f'llama-13b-{shared.args.wbits}bit'
        elif path_to_model.name.lower().startswith('llama-30b'):
            pt_model = f'llama-30b-{shared.args.wbits}bit'
        elif path_to_model.name.lower().startswith('llama-65b'):
            pt_model = f'llama-65b-{shared.args.wbits}bit'
        else:
            pt_model = f'{model_name}-{shared.args.wbits}bit'

        # Try to find the .safetensors or .pt both in the model dir and in the subfolder
        for path in [Path(p + ext) for ext in ['.safetensors', '.pt'] for p in [f"{shared.args.model_dir}/{pt_model}", f"{path_to_model}/{pt_model}"]]:
            if path.exists():
                print(f"Found {path}")
                pt_path = path
                break

    if not pt_path:
        print("Could not find the quantized model in .pt or .safetensors format, exiting...")
        exit()

    # qwopqwop200's offload
    if model_type == 'llama' and shared.args.pre_layer:
        model = load_quant(str(path_to_model), str(pt_path), shared.args.wbits, shared.args.groupsize, shared.args.pre_layer)
    else:
        threshold = False if model_type == 'gptj' else 128
        model = load_quant(str(path_to_model), str(pt_path), shared.args.wbits, shared.args.groupsize, kernel_switch_threshold=threshold)

        # accelerate offload (doesn't work properly)
        if shared.args.gpu_memory:
            memory_map = list(map(lambda x: x.strip(), shared.args.gpu_memory))
            max_cpu_memory = shared.args.cpu_memory.strip() if shared.args.cpu_memory is not None else '99GiB'
            max_memory = {}
            for i in range(len(memory_map)):
                max_memory[i] = f'{memory_map[i]}GiB' if not re.match('.*ib$', memory_map[i].lower()) else memory_map[i]
            max_memory['cpu'] = max_cpu_memory

            device_map = accelerate.infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=["LlamaDecoderLayer"])
            print("Using the following device map for the 4-bit model:", device_map)
            # https://huggingface.co/docs/accelerate/package_reference/big_modeling#accelerate.dispatch_model
            model = accelerate.dispatch_model(model, device_map=device_map, offload_buffers=True)

        # No offload
        elif not shared.args.cpu:
            model = model.to(torch.device('cuda:0'))

    return model
