from pathlib import Path

import torch
from peft import PeftModel

import modules.shared as shared
from modules.models import reload_model


def add_lora_to_model(lora_name):

    # If a LoRA had been previously loaded, or if we want
    # to unload a LoRA, reload the model
    if shared.lora_name not in ['None', ''] or lora_name in ['None', '']:
        reload_model()
    shared.lora_name = lora_name

    if lora_name not in ['None', '']:
        print(f"Adding the LoRA {lora_name} to the model...")
        params = {}
        if not shared.args.cpu:
            params['dtype'] = shared.model.dtype
            if hasattr(shared.model, "hf_device_map"):
                params['device_map'] = {"base_model.model." + k: v for k, v in shared.model.hf_device_map.items()}
            elif shared.args.load_in_8bit:
                params['device_map'] = {'': 0}

        shared.model = PeftModel.from_pretrained(shared.model, Path(f"{shared.args.lora_dir}/{lora_name}"), **params)
        if not shared.args.load_in_8bit and not shared.args.cpu:
            shared.model.half()
            if not hasattr(shared.model, "hf_device_map"):
                if torch.has_mps:
                    device = torch.device('mps')
                    shared.model = shared.model.to(device)
                else:
                    shared.model = shared.model.cuda()
