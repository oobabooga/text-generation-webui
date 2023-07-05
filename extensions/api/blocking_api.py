import json
import yaml
import base64
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from extensions.api.util import build_parameters, try_start_cloudflared
from modules import shared
from modules.chat import generate_chat_reply
from modules.LoRA import add_lora_to_model
from modules.models import load_model, unload_model
from modules.models_settings import (get_model_settings_from_yamls,
                                     update_model_parameters)
from modules.text_generation import (encode, generate_reply,
                                     stop_everything_event)
from modules.utils import (get_available_models,
                           get_available_presets,
                           get_available_prompts,
                           get_available_characters,
                           get_available_instruction_templates)


def get_model_info():
    return {
        'model_name': shared.model_name,
        'lora_names': shared.lora_names,
        # dump
        'shared.settings': shared.settings,
        'shared.args': vars(shared.args),
    }


class Handler(BaseHTTPRequestHandler):
    def simple_json_results(self, resp):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()

        response = json.dumps({
            'results': resp,
        })

        self.wfile.write(response.encode('utf-8'))

    def do_GET(self):
        if self.path == '/api/v1/model':
            self.send_response(200)
            self.end_headers()
            response = json.dumps({
                'result': shared.model_name
            })

            self.wfile.write(response.encode('utf-8'))

        elif self.path.startswith('/api/v1/prompts'):
            args = parse_qs(urlparse(self.path).query)
            name = args.get('name', None)
            if name:
                name = name[0]
                filepath = Path(f'prompts/{name}.txt')
                if filepath.exists():
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = f.read()
                        self.simple_json_results(data)
                else:
                    self.send_error(404, message="Prompt not found")
                    return
            else:
                self.simple_json_results(get_available_prompts())
            
        elif self.path.startswith('/api/v1/presets'):
            args = parse_qs(urlparse(self.path).query)
            name = args.get('name', None)
            if name:
                name = name[0]
                filepath = Path(f'presets/{name}.yaml')
                if filepath.exists():
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                        self.simple_json_results(data)
                else:
                    self.send_error(404, message="Preset not found")
                    return
            else:
                self.simple_json_results(get_available_presets())

        elif self.path.startswith('/api/v1/characters'):
            args = parse_qs(urlparse(self.path).query)
            name = args.get('name', None)
            #picture_response_format = args.get('picture_response_format', 'b64_json')
            if name:
                name = name[0]

                picture_data = None
                data = None

                for extension in ['png', 'jpg', 'jpeg']:
                    filepath = Path(f"characters/{name}.{extension}")
                    if filepath.exists():
                        with open(filepath, 'rb') as f:
                            file_contents = f.read()
                            encoded_bytes = base64.b64encode(file_contents)

                            # Turn raw base64 encoded bytes into ASCII
                            # TODO: support 'url' and 'data': url ? data_url?
                            img_data = encoded_bytes.decode('ascii')
                            img_filename = f"{name}.{extension}"
                            img_encoding = 'b64_json' # like SD, maybe also accept 'url'.. 'data_url'?
                            picture_data = {
                                'filename': img_filename,
                                'encoding': img_encoding, 
                                'data': img_data, # or 'data': url, and/or #'url': f"data:image/png;base64,{img_data}",
                            }
                            break

                for extension in ["yml", "yaml", "json"]:
                    filepath = Path(f'characters/{name}.{extension}')
                    if filepath.exists():
                        with open(filepath, 'r', encoding='utf-8') as f:
                            file_contents = f.read()
                            data = json.loads(file_contents) if extension == "json" else yaml.safe_load(file_contents)
                            break

                if not (picture_data or data):
                    self.send_error(404, message="Character not found")
                    return

                resp = {
                    'data': data,
                    'picture': picture_data,
                }
                self.simple_json_results(resp)

            else:
                self.simple_json_results(get_available_characters())

        elif self.path.startswith('/api/v1/instruction_templates'):
            args = parse_qs(urlparse(self.path).query)
            name = args.get('name', None)
            if name:
                name = name[0]
                filepath = Path(f'characters/instruction-following/{name}.yaml')
                if filepath.exists():
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                        self.simple_json_results(data)
                else:
                    self.send_error(404, message="Instruction Template not found")
                    return
            else:
                self.simple_json_results(get_available_instruction_templates())

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
            regenerate = body.get('regenerate', False)
            _continue = body.get('_continue', False)

            generate_params = build_parameters(body, chat=True)
            generate_params['stream'] = False

            generator = generate_chat_reply(
                user_input, generate_params, regenerate=regenerate, _continue=_continue, loading_message=False)

            answer = generate_params['history']
            for a in generator:
                answer = a

            response = json.dumps({
                'results': [{
                    'history': answer
                }]
            })

            self.wfile.write(response.encode('utf-8'))

        elif self.path == '/api/v1/stop-stream':
            stop_everything_event()
            self.simple_json_results('success')

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
            tokens = encode(body['prompt'])[0]
            self.simple_json_results([{'tokens': len(tokens)}])

        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', '*')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()


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
