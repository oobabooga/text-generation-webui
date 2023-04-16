from pathlib import Path

import gptq_triton
import modules.shared as shared
import torch


def load_quantized(model_name):
    # Find the model type
    if not shared.args.model_type:
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

    # Check if the model type is supported
    if shared.args.pre_layer:
        print("Error: --pre_layer is not supported for gptq_triton.")
        exit()
    elif model_type not in ('llama',):
        print("Error: gptq_triton only supports llama models.")
        exit()
    
    # Load the model
    model = gptq_triton.load_quant(Path(shared.args.model_dir) / model_name)
    model = model.to(torch.device('cuda'))

    return model