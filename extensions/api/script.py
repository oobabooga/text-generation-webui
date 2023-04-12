import json
from threading import Thread
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import asyncio
import websockets
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from modules import shared
from modules.text_generation import encode, generate_reply
import re

params = {
    'port': 5000,
    'portws': 4999,
}

def get_generate_params(data):
    return {
        'max_new_tokens': int(data.get('max_new_tokens', 200)),
        'do_sample': bool(data.get('do_sample', True)),
        'temperature': float(data.get('temperature', 0.8)),
        'top_p': float(data.get('top_p', 0.98)),
        'typical_p': float(data.get('typical', 1)),
        'repetition_penalty': float(data.get('rep_pen', 1.2)),
        'encoder_repetition_penalty': 1,
        'top_k': int(data.get('top_k', 0)),
        'min_length': int(data.get('min_length', 0)),
        'no_repeat_ngram_size': int(data.get('no_repeat_ngram_size', 0)),
        'num_beams': int(data.get('num_beams', 1)),
        'penalty_alpha': float(data.get('penalty_alpha', 0)),
        'length_penalty': float(data.get('length_penalty', 1)),
        'early_stopping': bool(data.get('early_stopping', False)),
        'seed': int(data.get('seed', -1)),
        'add_bos_token': int(data.get('add_bos_token', True)),
        'custom_stopping_strings': data.get('custom_stopping_strings', []),
        'truncation_length': int(data.get('truncation_length', 2048)),
        'ban_eos_token': bool(data.get('ban_eos_token', False)),
    }

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
            prompt = body['prompt']
            prompt_lines = [k.strip() for k in prompt.split('\n')]

            while len(prompt_lines) >= 0 and len(encode('\n'.join(prompt_lines))) > body.get('max_context_length', 2048):
                prompt_lines.pop(0)

            prompt = '\n'.join(prompt_lines)
            generate_params = get_generate_params(body)

            generator = generate_reply(
                prompt,
                generate_params,
            )

            answer = ''
            for a in generator:
                if isinstance(a, str):
                    answer = a
                else:
                    answer = a[0]

            response = json.dumps({
                'results': [{
                    'text': answer[len(prompt):]
                }]
            })
            self.wfile.write(response.encode('utf-8'))
        elif self.path == '/api/v1/token-count':
            # Not compatible with KoboldAI api
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            tokens = encode(body['prompt'])[0]
            response = json.dumps({
                'results': [{
                    'tokens': len(tokens)
                }]
            })
            self.wfile.write(response.encode('utf-8'))
        else:
            self.send_error(404)


async def handle_client(websocket, path):
    async for message in websocket:
        data = json.loads(message)
        prompt = data.get("prompt")
        generate_params = get_generate_params(data)
        if prompt is not None:
            try:
                full_reply = ""
                end_string_present = False
                for a in generate_reply(prompt, generate_params):
                    if isinstance(a, str):
                        answer = a
                    else:
                        answer = a[0]
                    full_reply += answer
                    for stopping_string in data.get("custom_stopping_strings", []):
                        # 4 because token most length <= 4
                        pattern = re.escape(stopping_string) + r"\s?.{0,4}$"
                        if re.search(pattern, full_reply):
                            end_string_present = True
                            print("generate_reply() did't stop, break")
                    if end_string_present:
                        break
                    await websocket.send(json.dumps({"text": answer}))
                await websocket.send(json.dumps({"generation_complete": True}))
                if not end_string_present:
                    print("generate_reply() auto stopped")
                print("websocket.close() 1")
                await websocket.close()
            except websockets.exceptions.ConnectionClosedError as e:
                print(f"WS connection closed while generating reply: {str(e)}")
        else:
            try:
                await websocket.send(json.dumps({"error": "Invalid input"}))
                print("websocket.close() 2")
                await websocket.close()
            except websockets.exceptions.ConnectionClosedError as e:
                print(f"WS connection closed while reporting error: {str(e)}")

async def websocket_server(loop):
    server = await websockets.serve(handle_client, '0.0.0.0' if shared.args.listen else '127.0.0.1', params['portws'], loop=loop)
    print("\nWebSocket server listening on port", params['portws'])
    await server.wait_closed()

def run_websocket_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(websocket_server(loop))
    finally:
        loop.close()

def run_http_server():
    server_addr = ('0.0.0.0' if shared.args.listen else '127.0.0.1', params['port'])
    httpd = ThreadingHTTPServer(server_addr, Handler)
    if shared.args.share:
        try:
            from flask_cloudflared import _run_cloudflared
            public_url = _run_cloudflared(params['port'], params['port'] + 1)
            print(f'Starting KoboldAI compatible api at {public_url}/api')
        except ImportError:
            print('You should install flask_cloudflared manually')
    else:
        print(f'Starting KoboldAI compatible api at http://{server_addr[0]}:{server_addr[1]}/api')
    httpd.serve_forever()

def run_dual_stack_server():
    Thread(target=run_websocket_server, daemon=True).start()
    run_http_server()

def setup():
    Thread(target=run_dual_stack_server, daemon=True).start()
    
