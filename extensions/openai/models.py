from modules import shared
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
