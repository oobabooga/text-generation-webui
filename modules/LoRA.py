from pathlib import Path

import torch
from peft import PeftModel

import modules.shared as shared
from modules.logging_colors import logger
from modules.models import reload_model


def add_lora_to_model(lora_names):
    if shared.model_name == 'None' or shared.model is None:
        logger.error("No model is loaded, please load a base model first before loading Lora.")
        return False
    else:
        if 'GPTQForCausalLM' in shared.model.__class__.__name__ or shared.args.loader == 'AutoGPTQ':
            return add_lora_autogptq(lora_names)
        elif shared.model.__class__.__name__ in ['ExllamaModel', 'ExllamaHF'] or shared.args.loader == 'ExLlama':
            return add_lora_exllama(lora_names)
        elif 'LlamaCppModel' in shared.model.__class__.__name__ or shared.args.loader == 'llama.cpp':
            return add_lora_llamacpp(lora_names)
        else:
            return add_lora_transformers(lora_names)


def get_lora_path(lora_name):
    p = Path(lora_name)
    if p.exists():
        lora_name = p.parts[-1]

    return Path(f"{shared.args.lora_dir}/{lora_name}")


def add_lora_llamacpp(lora_names):
    if len(lora_names) == 0:
        shared.args.lora_path = ''
        reload_model()
        shared.lora_names = []
        logger.info("Successfully removed LoRAs")
        return False
    else:
        if len(lora_names) > 1:
            logger.warning('Llama.cpp can only work with 1 LoRA at the moment. Only the first one in the list will be loaded.')
        lora_path = get_lora_path(lora_names[0])
        logger.info("Applying the following LoRAs to {}: {}".format(shared.model_name, ', '.join([lora_names[0]])))
        if lora_path.is_file():
            shared.args.lora_path = f"{shared.args.lora_dir}/{lora_names[0]}"
        elif lora_path.is_dir():
            lora_path = lora_path / "ggml-adapter-model.bin"
            if lora_path.is_file():
                shared.args.lora_path = f"{shared.args.lora_dir}/{lora_names[0]}/ggml-adapter-model.bin"
            else:
                logger.error(f"Could not find the LoRA model. Make sure ggml-adapter-model.bin is in {shared.args.lora_dir}/{lora_names[0]} folder.")
                return False
        else:
            logger.error("Wrong path!")
            return False
        flag = True
        try:
            reload_model()
        except:
            flag = False
            logger.error("Llama apply LoRA error")
        shared.lora_names = [lora_names[0]]
        return flag



def add_lora_exllama(lora_names):

    try:
        from exllama.lora import ExLlamaLora
    except:
        try:
            from repositories.exllama.lora import ExLlamaLora
        except:
            logger.error("Could not find the file repositories/exllama/lora.py. Make sure that exllama is cloned inside repositories/ and is up to date.")
            return False

    if len(lora_names) == 0:
        if shared.model.__class__.__name__ == 'ExllamaModel':
            shared.model.generator.lora = None
        else:
            shared.model.lora = None

        shared.lora_names = []
        logger.info("Successfully removed LoRAs")
        return False
    else:
        if len(lora_names) > 1:
            logger.warning('ExLlama can only work with 1 LoRA at the moment. Only the first one in the list will be loaded.')

        lora_path = get_lora_path(lora_names[0])
        lora_config_path = lora_path / "adapter_config.json"
        lora_adapter_path = lora_path / "adapter_model.bin"

        logger.info("Applying the following LoRAs to {}: {}".format(shared.model_name, ', '.join([lora_names[0]])))
        if shared.model.__class__.__name__ == 'ExllamaModel':
            lora = ExLlamaLora(shared.model.model, str(lora_config_path), str(lora_adapter_path))
            shared.model.generator.lora = lora
        else:
            lora = ExLlamaLora(shared.model.ex_model, str(lora_config_path), str(lora_adapter_path))
            shared.model.lora = lora

        shared.lora_names = [lora_names[0]]
        return True


# Adapted from https://github.com/Ph0rk0z/text-generation-webui-testing
def add_lora_autogptq(lora_names):

    try:
        from auto_gptq import get_gptq_peft_model
        from auto_gptq.utils.peft_utils import GPTQLoraConfig
    except:
        logger.error("This version of AutoGPTQ does not support LoRA. You need to install from source or wait for a new release.")
        return False

    if len(lora_names) == 0:
        reload_model()

        shared.lora_names = []
        logger.info("Successfully removed LoRAs")
        return False
    else:
        if len(lora_names) > 1:
            logger.warning('AutoGPTQ can only work with 1 LoRA at the moment. Only the first one in the list will be loaded.')
        if not shared.args.no_inject_fused_attention:
            logger.warning('Fused Atttention + AutoGPTQ may break Lora loading. Disable it.')

        peft_config = GPTQLoraConfig(
            inference_mode=True,
        )

        lora_path = get_lora_path(lora_names[0])
        logger.info("Applying the following LoRAs to {}: {}".format(shared.model_name, ', '.join([lora_names[0]])))
        shared.model = get_gptq_peft_model(shared.model, peft_config, lora_path)
        shared.lora_names = [lora_names[0]]
        return True


def add_lora_transformers(lora_names):
    prior_set = set(shared.lora_names)
    added_set = set(lora_names) - prior_set
    removed_set = prior_set - set(lora_names)

    # If no LoRA needs to be added or removed, exit
    if len(added_set) == 0 and len(removed_set) == 0:
        return True

    # Add a LoRA when another LoRA is already present
    if len(removed_set) == 0 and len(prior_set) > 0:
        logger.info(f"Adding the LoRA(s) named {added_set} to the model...")
        for lora in added_set:
            shared.model.load_adapter(get_lora_path(lora), lora)

        return True

    # If any LoRA needs to be removed, start over
    if len(removed_set) > 0:
        # shared.model may no longer be PeftModel
        if hasattr(shared.model, 'disable_adapter'):
            shared.model.disable_adapter()
            shared.model = shared.model.base_model.model

    if len(lora_names) > 0:
        params = {}
        if not shared.args.cpu:
            if shared.args.load_in_4bit or shared.args.load_in_8bit:
                params['peft_type'] = shared.model.dtype
            else:
                params['dtype'] = shared.model.dtype
                if hasattr(shared.model, "hf_device_map"):
                    params['device_map'] = {"base_model.model." + k: v for k, v in shared.model.hf_device_map.items()}

        logger.info("Applying the following LoRAs to {}: {}".format(shared.model_name, ', '.join(lora_names)))
        shared.model = PeftModel.from_pretrained(shared.model, get_lora_path(lora_names[0]), adapter_name=lora_names[0], **params)
        for lora in lora_names[1:]:
            shared.model.load_adapter(get_lora_path(lora), lora)

        shared.lora_names = lora_names

        if not shared.args.load_in_8bit and not shared.args.cpu:
            shared.model.half()
            if not hasattr(shared.model, "hf_device_map"):
                if torch.backends.mps.is_available():
                    device = torch.device('mps')
                    shared.model = shared.model.to(device)
                else:
                    shared.model = shared.model.cuda()
