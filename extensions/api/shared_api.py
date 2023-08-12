from modules import shared
from modules.text_generation import (
    get_encoded_length,
    stop_everything_event
)
from modules.utils import get_available_models
from modules.models import load_model, unload_model
from modules.models_settings import (
    get_model_settings_from_yamls,
    update_model_parameters
)
from modules.LoRA import add_lora_to_model
from modules.logging_colors import logger


# any method defined in this file can NOT be async since they are intended to be used by both, the HTTP and the websocket server


def get_model_info():
    return {
        'model_name': shared.model_name,
        'lora_names': shared.lora_names,
        # dump
        'shared.settings': shared.settings,
        'shared.args': vars(shared.args),
    }


def ensureModelLoaded(context):
    if shared.model_name is not None and shared.model_name != 'None':
        return True
    else:
        logger.warning('api request not handled, no model is loaded')
        context['responseHandler'](context, {
            'event': 'warning',
            'message': 'no model loaded'
        })
        return False


def _handle_stop_streaming_request(context, message): #2nd argument is there for compatibility reasons...
    stop_everything_event()
    context['responseHandler'](context, { 'event': 'stop-stream', 'results': 'success' })


def _handle_token_count_request(connectionContext, message):
    if not ensureModelLoaded(connectionContext):
        return

    connectionContext['responseHandler'](connectionContext, {
        'event': 'token-count',
        'count': get_encoded_length(message['prompt']),
        'prompt': message['prompt']
    })


def _handle_model_request(connectionContext, message):
    # Actions: info, load, list, unload
    action = message.get('action', '')

    if action == 'load':
        model_name = message['model_name']
        args = message.get('args', {})


        logger.info('model loading args: %s', args)

        for k in args:
            setattr(shared.args, k, args[k])

        shared.model_name = model_name
        unload_model()

        model_settings = get_model_settings_from_yamls(shared.model_name)
        shared.settings.update(model_settings)
        update_model_parameters(model_settings, initial=True)

        if shared.settings['mode'] != 'instruct':
            shared.settings['instruction_template'] = None

        try:
            shared.model, shared.tokenizer = load_model(shared.model_name)
            if shared.args.lora:
                add_lora_to_model(shared.args.lora)  # list

        except Exception as e:
            connectionContext['responseHandler'](connectionContext, { 'event': 'error', 'message': repr(e) })
            raise e

        shared.args.model = shared.model_name

        connectionContext['responseHandler'](connectionContext, { 'event': 'model-loaded', 'result': get_model_info() })

    elif action == 'unload':
        unload_model()
        shared.args.model = None
        connectionContext['responseHandler'](connectionContext, { 'event': 'model-unloaded', 'result': get_model_info() })

    elif action == 'list':
        connectionContext['responseHandler'](connectionContext, {'event': 'model-list', 'result': get_available_models() })

    elif action == 'info':
        connectionContext['responseHandler'](connectionContext, {'event': 'model-info', 'result': get_model_info() })

    else:
        # by default return the same as the GET interface
        connectionContext['responseHandler'](connectionContext, { 'result': shared.model_name })