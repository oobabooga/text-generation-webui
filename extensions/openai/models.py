from extensions.openai.embeddings import get_embeddings_model_name
from extensions.openai.errors import OpenAIError
from modules import shared
from modules.models import load_model as _load_model
from modules.models import unload_model
from modules.models_settings import get_model_metadata, update_model_parameters
from modules.utils import get_available_models


def get_current_model_list() -> list:
    return [shared.model_name]  # The real chat/completions model, maybe "None"


def get_pseudo_model_list() -> list:
    return [  # these are expected by so much, so include some here as a dummy
        'gpt-3.5-turbo',
        'text-embedding-ada-002',
    ]


def load_model(model_name: str) -> dict:
    resp = {
        "id": model_name,
        "object": "engine",
        "owner": "self",
        "ready": True,
    }
    if model_name not in get_pseudo_model_list() + [get_embeddings_model_name()] + get_current_model_list():  # Real model only
        # No args. Maybe it works anyways!
        # TODO: hack some heuristics into args for better results

        shared.model_name = model_name
        unload_model()

        model_settings = get_model_metadata(shared.model_name)
        shared.settings.update({k: v for k, v in model_settings.items() if k in shared.settings})
        update_model_parameters(model_settings, initial=True)

        if shared.settings['mode'] != 'instruct':
            shared.settings['instruction_template'] = None

        shared.model, shared.tokenizer = _load_model(shared.model_name)

        if not shared.model:  # load failed.
            shared.model_name = "None"
            raise OpenAIError(f"Model load failed for: {shared.model_name}")

    return resp


def list_models(is_legacy: bool = False) -> dict:
    # TODO: Lora's?
    all_model_list = get_current_model_list() + [get_embeddings_model_name()] + get_pseudo_model_list() + get_available_models()

    models = {}

    if is_legacy:
        models = [{"id": id, "object": "engine", "owner": "user", "ready": True} for id in all_model_list]
        if not shared.model:
            models[0]['ready'] = False
    else:
        models = [{"id": id, "object": "model", "owned_by": "user", "permission": []} for id in all_model_list]

    resp = {
        "object": "list",
        "data": models,
    }

    return resp


def model_info(model_name: str) -> dict:
    return {
        "id": model_name,
        "object": "model",
        "owned_by": "user",
        "permission": []
    }
