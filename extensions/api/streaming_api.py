import asyncio
import json
from threading import Thread

from extensions.api.shared_api import (
    ensureModelLoaded,
    _handle_token_count_request,
    _handle_model_request,
    _handle_stop_streaming_request
)

from extensions.api.util import (
    build_parameters,
    try_start_cloudflared,
    with_api_lock
)
from modules import shared
from modules.chat import generate_chat_reply
from modules.text_generation import generate_reply
from websockets.server import serve
from modules.logging_colors import logger

def WebsocketResponseHandler(context, message):
    return asyncio.create_task(context['websocket'].send(json.dumps(message)))

@with_api_lock
async def _handle_stream_message(context, message):
    if not ensureModelLoaded(context):
        return

    prompt = message['prompt']
    generate_params = build_parameters(message)
    stopping_strings = generate_params.pop('stopping_strings')
    generate_params['stream'] = True

    generator = generate_reply(
        prompt, generate_params, stopping_strings=stopping_strings, is_chat=False)

    # As we stream, only send the new bytes.
    skip_index = 0
    message_num = 0

    for a in generator:
        to_send = a[skip_index:]
        if to_send is None or chr(0xfffd) in to_send:  # partial unicode character, don't send it yet.
            continue

        await context['responseHandler'](context, {
            'event': 'text_stream',
            'message_num': message_num,
            'text': to_send
        })

        await asyncio.sleep(0)
        skip_index += len(to_send)
        message_num += 1

    await context['responseHandler'](context, {
        'event': 'stream_end',
        'message_num': message_num
    })


@with_api_lock
async def _handle_chat_stream_message(connectionContext, message):
    if not ensureModelLoaded(connectionContext):
        return

    user_input = message['user_input']
    generate_params = build_parameters(message, chat=True)
    generate_params['stream'] = True
    regenerate = message.get('regenerate', False)
    _continue = message.get('_continue', False)

    generator = generate_chat_reply(
        user_input, generate_params, regenerate=regenerate, _continue=_continue, loading_message=False)

    message_num = 0
    for a in generator:
        await connectionContext['responseHandler'](connectionContext, {
            'event': 'text_stream',
            'message_num': message_num,
            'history': a
        })

        await asyncio.sleep(0)
        message_num += 1

    await connectionContext['responseHandler'](connectionContext, {
        'event': 'stream_end',
        'message_num': message_num
    })


WEBSOCKET_PATH_HANDLER_DICT = {
    '/api/v1/stream': {
        'handler': _handle_stream_message,
        'async': True
    },
    '/api/v1/chat-stream': {
        'handler': _handle_chat_stream_message,
        'async': True
    },
    '/api/v1/token-count': {
        'handler': _handle_token_count_request,
        'async': False
    },
    '/api/v1/model': {
        'handler': _handle_model_request,
        'async': False
    },
    '/api/v1/stop-stream': {
        'handler': _handle_stop_streaming_request,
        'async': False
    }
}

async def _handle_connection(websocket, path):

    if WEBSOCKET_PATH_HANDLER_DICT.get(path):
        connectionContext = {
            'responseHandler': WebsocketResponseHandler,
            'websocket': websocket
        }

        async for message in websocket:
            try:
                if WEBSOCKET_PATH_HANDLER_DICT[path]['async']:
                    await WEBSOCKET_PATH_HANDLER_DICT[path]['handler'](connectionContext, json.loads(message))
                else:
                    WEBSOCKET_PATH_HANDLER_DICT[path]['handler'](connectionContext, json.loads(message))
            except ValueError: # catch JSON parsing errors
                logger.warning("API request not handled, malformed JSON")
                connectionContext['responseHandler'](connectionContext, { 'event': 'error', 'message': 'malformed JSON data received'})
        return

    else:
        logger.warning('Streaming api: unknown path: %s', path)
        return


async def _run(host: str, port: int):
    async with serve(_handle_connection, host, port, ping_interval=None):
        await asyncio.Future()  # run forever


def _run_server(port: int, share: bool = False, tunnel_id=str):
    address = '0.0.0.0' if shared.args.listen else '127.0.0.1'

    def on_start(public_url: str):
        public_url = public_url.replace('https://', 'wss://')
        for path in WEBSOCKET_PATH_HANDLER_DICT:
            logger.info("Starting websocket API server at public url %s:%s%s", public_url, port, path)

    if share:
        try:
            try_start_cloudflared(port, tunnel_id, max_attempts=3, on_start=on_start)
        except Exception as e:
            print(e)
    else:
        for path in WEBSOCKET_PATH_HANDLER_DICT:
            logger.info("Starting websocket API server at ws://%s:%s%s", address, port, path)

    asyncio.run(_run(host=address, port=port))


def start_server(port: int, share: bool = False, tunnel_id=str):
    Thread(target=_run_server, args=[port, share, tunnel_id], daemon=True).start()
