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
        print(f"Adding the LoRA {lora_name} to the model...")
        
        params = {}
        if shared.args.load_in_8bit:
            params['device_map'] = {'': 0}
        else:
            params['device_map'] = 'auto' 
            params['dtype'] = shared.model.dtype
            
        shared.model = PeftModel.from_pretrained(shared.model, Path(f"loras/{lora_name}"), **params)
        if not shared.args.load_in_8bit:
            shared.model.half()
