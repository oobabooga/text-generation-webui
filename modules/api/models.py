from modules import loaders, shared
from modules.LoRA import add_lora_to_model
from modules.models import load_model, unload_model
from modules.models_settings import get_model_metadata, update_model_parameters
from modules.utils import get_available_loras, get_available_models


def get_current_model_info():
    return {
        'model_name': shared.model_name,
        'lora_names': shared.lora_names,
        'loader': shared.args.loader
    }


def list_models():
    return {'model_names': get_available_models()}


def list_models_openai_format():
    """Returns model list in OpenAI API format"""
    if shared.model_name and shared.model_name != 'None':
        data = [model_info_dict(shared.model_name)]
    else:
        data = []

    return {
        "object": "list",
        "data": data
    }


def model_info_dict(model_name: str) -> dict:
    return {
        "id": model_name,
        "object": "model",
        "created": 0,
        "owned_by": "user"
    }


def _load_model(data):
    model_name = data["model_name"]
    args = data.get("args")

    unload_model()
    model_settings = get_model_metadata(model_name)

    # Update shared.args with custom model loading settings
    # Security: only allow keys that correspond to model loading
    # parameters exposed in the UI. Never allow security-sensitive
    # flags like trust_remote_code or extra_flags to be set via the API.
    blocked_keys = {'extra_flags'}
    allowed_keys = set(loaders.list_model_elements()) - blocked_keys

    # Reset all loader args to their startup values before applying new ones,
    # so settings from a previous API load don't leak into this one.
    # Include blocked keys in the reset (safe: restores startup value, not API-controlled).
    for k in allowed_keys | blocked_keys:
        if hasattr(shared.args, k) and hasattr(shared.original_args, k):
            setattr(shared.args, k, getattr(shared.original_args, k))

    update_model_parameters(model_settings)

    if args:
        for k in args:
            if k in allowed_keys and hasattr(shared.args, k):
                setattr(shared.args, k, args[k])

    shared.model, shared.tokenizer = load_model(model_name)


def list_loras():
    return {'lora_names': get_available_loras()[1:]}


def load_loras(lora_names):
    add_lora_to_model(lora_names)


def unload_all_loras():
    add_lora_to_model([])
