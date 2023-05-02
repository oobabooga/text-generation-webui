import json
import asyncio
from websockets.server import serve
from threading import Thread

from modules import shared
from modules.text_generation import generate_reply

from extensions.api.util import build_parameters, try_start_cloudflared

PATH = '/api/v1/stream'


async def _handle_connection(websocket, path):

    if path != PATH:
        print(f'Streaming api: unknown path: {path}')
        return

    async for message in websocket:
        message = json.loads(message)

        prompt = message['prompt']
        generate_params = build_parameters(message)
        stopping_strings = generate_params.pop('stopping_strings')

        generator = generate_reply(
            prompt, generate_params, stopping_strings=stopping_strings)

        # As we stream, only send the new bytes.
        skip_index = len(prompt) if not shared.is_chat() else 0
        message_num = 0

        for a in generator:
            to_send = ''
            if isinstance(a, str):
                to_send = a[skip_index:]
            else:
                to_send = a[0][skip_index:]

            await websocket.send(json.dumps({
                'event': 'text_stream',
                'message_num': message_num,
                'text': to_send
            }))

            await asyncio.sleep(0)

            skip_index += len(to_send)
            message_num += 1

        await websocket.send(json.dumps({
            'event': 'stream_end',
            'message_num': message_num
        }))


async def _run(host: str, port: int):
    async with serve(_handle_connection, host, port, ping_interval=None):
        await asyncio.Future()  # run forever


def _run_server(port: int, share: bool = False):
    address = '0.0.0.0' if shared.args.listen else '127.0.0.1'

    def on_start(public_url: str):
        public_url = public_url.replace('https://', 'wss://')
        print(f'Starting streaming server at public url {public_url}{PATH}')

    if share:
        try:
            try_start_cloudflared(port, max_attempts=3, on_start=on_start)
        except Exception as e:
            print(e)
    else:
        print(f'Starting streaming server at ws://{address}:{port}{PATH}')

    asyncio.run(_run(host=address, port=port))


def start_server(port: int, share: bool = False):
    Thread(target=_run_server, args=[port, share], daemon=True).start()
