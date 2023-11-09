from modules import shared
from modules.models import load_model, unload_model
from modules.models_settings import get_model_metadata, update_model_parameters
from modules.utils import get_available_models


def get_current_model_info():
    return {
        'model_name': shared.model_name,
        'lora_names': shared.lora_names
    }


def list_models():
    result = {
        "object": "list",
        "data": []
    }

    for model in get_dummy_models() + get_available_models()[1:]:
        result["data"].append(model_info_dict(model))

    return result


def model_info_dict(model_name: str) -> dict:
    return {
        "id": model_name,
        "object": "model",
        "created": 0,
        "owned_by": "user"
    }


def get_dummy_models() -> list:
    return [  # these are expected by so much, so include some here as a dummy
        'gpt-3.5-turbo',
        'text-embedding-ada-002',
    ]


def _load_model(data):
    model_name = data["model_name"]
    args = data["args"]
    settings = data["settings"]

    unload_model()
    model_settings = get_model_metadata(model_name)
    update_model_parameters(model_settings)

    # Update shared.args with custom model loading settings
    if args:
        for k in args:
            if hasattr(shared.args, k):
                setattr(shared.args, k, args[k])

    shared.model, shared.tokenizer = load_model(model_name)

    # Update shared.settings with custom generation defaults
    if settings:
        for k in settings:
            if k in shared.settings:
                shared.settings[k] = settings[k]
