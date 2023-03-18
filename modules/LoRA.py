from pathlib import Path

import modules.shared as shared
from modules.models import load_model


def add_lora_to_model(lora_name):

    from peft import PeftModel

    # Is there a more efficient way of returning to the base model?
    if lora_name == "None":
        print("Reloading the model to remove the LoRA...")
        shared.model, shared.tokenizer = load_model(shared.model_name)
    else:
        # Why doesn't this work in 16-bit mode?
        print(f"Adding the LoRA {lora_name} to the model...")

        params = {}
        #params['device_map'] = {'': 0}
        #params['dtype'] = shared.model.dtype
        shared.model = PeftModel.from_pretrained(shared.model, Path(f"loras/{lora_name}"), **params)
