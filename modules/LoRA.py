import logging
from pathlib import Path

import torch
from peft import PeftModel

import modules.shared as shared


def add_lora_to_model(lora_names):
    prior_set = set(shared.lora_names)
    added_set = set(lora_names) - prior_set
    removed_set = prior_set - set(lora_names)
    shared.lora_names = list(lora_names)

    # If no LoRA needs to be added or removed, exit
    if len(added_set) == 0 and len(removed_set) == 0:
        return

    # Add a LoRA when another LoRA is already present
    if len(removed_set) == 0 and len(prior_set) > 0:
        logging.info(f"Adding the LoRA(s) named {added_set} to the model...")
        for lora in added_set:
            shared.model.load_adapter(Path(f"{shared.args.lora_dir}/{lora}"), lora)

        return

    # If any LoRA needs to be removed, start over
    if len(removed_set) > 0:
        shared.model.disable_adapter()
        shared.model = shared.model.base_model.model

    if len(lora_names) > 0:
        logging.info("Applying the following LoRAs to {}: {}".format(shared.model_name, ', '.join(lora_names)))
        params = {}
        if not shared.args.cpu:
            params['dtype'] = shared.model.dtype
            if hasattr(shared.model, "hf_device_map"):
                params['device_map'] = {"base_model.model." + k: v for k, v in shared.model.hf_device_map.items()}
            elif shared.args.load_in_8bit:
                params['device_map'] = {'': 0}

        shared.model = PeftModel.from_pretrained(shared.model, Path(f"{shared.args.lora_dir}/{lora_names[0]}"), **params)

        for lora in lora_names[1:]:
            shared.model.load_adapter(Path(f"{shared.args.lora_dir}/{lora}"), lora)

        if not shared.args.load_in_8bit and not shared.args.cpu:
            shared.model.half()
            if not hasattr(shared.model, "hf_device_map"):
                if torch.has_mps:
                    device = torch.device('mps')
                    shared.model = shared.model.to(device)
                else:
                    shared.model = shared.model.cuda()
