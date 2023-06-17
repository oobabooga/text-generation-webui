import json
import torch

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from extensions.api.util import build_parameters, try_start_cloudflared
from modules import shared
from modules.chat import generate_chat_reply
from modules.LoRA import add_lora_to_model
from modules.models import load_model, unload_model
from modules.models_settings import (get_model_settings_from_yamls,
                                     update_model_parameters)
from modules.text_generation import (encode, decode, generate_reply,
                                     stop_everything_event)
from modules.utils import get_available_models


def get_model_info():
    return {
        'model_name': shared.model_name,
        'lora_names': shared.lora_names,
        # dump
        'shared.settings': shared.settings,
        'shared.args': vars(shared.args),
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/v1/model':
            self.send_response(200)
            self.end_headers()
            response = json.dumps({
                'result': shared.model_name
            })

            self.wfile.write(response.encode('utf-8'))
        else:
            self.send_error(404)

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = json.loads(self.rfile.read(content_length).decode('utf-8'))

        if self.path == '/api/v1/generate':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()

            prompt = body['prompt']
            generate_params = build_parameters(body)
            stopping_strings = generate_params.pop('stopping_strings')
            generate_params['stream'] = False

            generator = generate_reply(
                prompt, generate_params, stopping_strings=stopping_strings, is_chat=False)

            answer = ''
            for a in generator:
                answer = a

            response = json.dumps({
                'results': [{
                    'text': answer
                }]
            })

            self.wfile.write(response.encode('utf-8'))

        elif self.path == '/api/v1/chat':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()

            user_input = body['user_input']
            history = body['history']
            regenerate = body.get('regenerate', False)
            _continue = body.get('_continue', False)

            generate_params = build_parameters(body, chat=True)
            generate_params['stream'] = False

            generator = generate_chat_reply(
                user_input, history, generate_params, regenerate=regenerate, _continue=_continue, loading_message=False)

            answer = history
            for a in generator:
                answer = a

            response = json.dumps({
                'results': [{
                    'history': answer
                }]
            })

            self.wfile.write(response.encode('utf-8'))

        elif self.path == '/api/v1/stop-stream':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()

            stop_everything_event()

            response = json.dumps({
                'results': 'success'
            })

            self.wfile.write(response.encode('utf-8'))

        elif self.path == '/api/v1/model':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()

            # by default return the same as the GET interface
            result = shared.model_name

            # Actions: info, load, list, unload
            action = body.get('action', '')

            if action == 'load':
                model_name = body['model_name']
                args = body.get('args', {})
                print('args', args)
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
                    response = json.dumps({'error': {'message': repr(e)}})

                    self.wfile.write(response.encode('utf-8'))
                    raise e

                shared.args.model = shared.model_name

                result = get_model_info()

            elif action == 'unload':
                unload_model()
                shared.model_name = None
                shared.args.model = None
                result = get_model_info()

            elif action == 'list':
                result = get_available_models()

            elif action == 'info':
                result = get_model_info()

            response = json.dumps({
                'result': result,
            })

            self.wfile.write(response.encode('utf-8'))

        elif self.path == '/api/v1/token-count':
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

        elif self.path == '/api/v1/encode':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()

            prompt = body['prompt']
            add_special_tokens = body.get('add_special_tokens', True)
            add_bos_token = body.get('add_bos_token', True)
            truncation_length = body.get('truncation_length')

            tokens = encode(prompt, add_special_tokens, add_bos_token, truncation_length)[0]
            # tensor to list
            token_ids = [t.item() for t in tokens]

            # Using encode directly for each token would cause spaces to be lost
            # and non-english characters could be encoded as multiple tokens
            token_texts = []
            tokens = torch.tensor([])
            new_ids = []
            length = 0
            for t in token_ids:
                new_tokens = torch.tensor([t])
                tokens = torch.cat((tokens, new_tokens))
                new_ids.extend([t])
                decoded_text = decode(tokens, not add_special_tokens)
                # Take the new part from the end of decoded_text as the text of the new_tokens
                new_text_length = len(decoded_text) - length
                new_text = decoded_text[-new_text_length:]

                # chr(0xfffd) is a partial unicode character
                if chr(0xfffd) in new_text:
                    continue

                length = len(decoded_text)
                token_texts.append({
                    'text': new_text,
                    'ids': new_ids
                })
                new_ids = []

            if len(new_ids) > 0:
                token_texts.append({
                    'text': new_text,
                    'ids': new_ids
                })

            response = json.dumps({
                'token_ids': token_ids,
                'token_texts': token_texts,
                'tokens_count': len(token_ids)
            })

            self.wfile.write(response.encode('utf-8'))

        elif self.path == '/api/v1/decode':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()

            token_ids = body['token_ids']
            skip_special_tokens = body.get('skip_special_tokens', True)

            # Using encode directly for each token would cause spaces to be lost
            # and non-english characters could be encoded as multiple tokens
            token_texts = []
            tokens = torch.tensor([])
            new_ids = []
            length = 0
            for t in token_ids:
                new_tokens = torch.tensor([t])
                tokens = torch.cat((tokens, new_tokens))
                new_ids.extend([t])
                decoded_text = decode(tokens, skip_special_tokens)
                # Take the new part from the end of decoded_text as the text of the new_tokens
                new_text_length = len(decoded_text) - length
                new_text = decoded_text[-new_text_length:]

                # chr(0xfffd) is a partial unicode character
                if chr(0xfffd) in new_text:
                    continue

                length = len(decoded_text)
                token_texts.append({
                    'text': new_text,
                    'ids': new_ids
                })
                new_ids = []

            if len(new_ids) > 0:
                token_texts.append({
                    'text': new_text,
                    'ids': new_ids
                })

            response = json.dumps({
                'decoded_text': decoded_text,
                'token_texts': token_texts
            })

            self.wfile.write(response.encode('utf-8'))

        else:
            self.send_error(404)


def _run_server(port: int, share: bool = False):
    address = '0.0.0.0' if shared.args.listen else '127.0.0.1'

    server = ThreadingHTTPServer((address, port), Handler)

    def on_start(public_url: str):
        print(f'Starting non-streaming server at public url {public_url}/api')

    if share:
        try:
            try_start_cloudflared(port, max_attempts=3, on_start=on_start)
        except Exception:
            pass
    else:
        print(
            f'Starting API at http://{address}:{port}/api')

    server.serve_forever()


def start_server(port: int, share: bool = False):
    Thread(target=_run_server, args=[port, share], daemon=True).start()
