import os
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from pathlib import Path

from modules import shared
from modules.text_generation import encode, generate_reply

params = {
    'port': 5000,
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
            prompt_lines = [k.strip() for k in prompt.split('\n')]
            max_context = body.get('max_context_length', 2048)
            while len(prompt_lines) >= 0 and len(encode('\n'.join(prompt_lines))) > max_context:
                prompt_lines.pop(0)

            prompt = '\n'.join(prompt_lines)
            defaults = get_defaults(body)
            generate_params = {
                'max_new_tokens': int(body.get('max_length', defaults['max_new_tokens'])),
                'do_sample': bool(body.get('do_sample', defaults['do_sample'])),
                'temperature': float(body.get('temperature', defaults['temperature'])),
                'top_p': float(body.get('top_p', defaults['top_p'])),
                'typical_p': float(body.get('typical', defaults['typical_p'])),
                'repetition_penalty': float(body.get('rep_pen', defaults['repetition_penalty'])),
                'encoder_repetition_penalty': defaults['encoder_repetition_penalty'],
                'top_k': int(body.get('top_k', defaults['top_k'])),
                'min_length': int(body.get('min_length', defaults['min_length'])),
                'no_repeat_ngram_size': int(body.get('no_repeat_ngram_size', defaults['no_repeat_ngram_size'])),
                'num_beams': int(body.get('num_beams', defaults['num_beams'])),
                'penalty_alpha': float(body.get('penalty_alpha', defaults['penalty_alpha'])),
                'length_penalty': float(body.get('length_penalty', defaults['length_penalty'])),
                'early_stopping': bool(body.get('early_stopping', defaults['early_stopping'])),
                'seed': int(body.get('seed', defaults['seed'])),
                'add_bos_token': int(body.get('add_bos_token', defaults['add_bos_token'])),
                'truncation_length': int(body.get('truncation_length', defaults['truncation_length'])),
                'ban_eos_token': bool(body.get('ban_eos_token', defaults['ban_eos_token'])),
                'skip_special_tokens': bool(body.get('skip_special_tokens', defaults['skip_special_tokens'])),
                'custom_stopping_strings': '',  # leave this blank
                'stopping_strings': body.get('stopping_strings', defaults['stopping_strings']),
            }
            stopping_strings = generate_params.pop('stopping_strings')
            generator = generate_reply(
                prompt, generate_params, stopping_strings=stopping_strings)
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


def run_server():
    server_addr = (
        '0.0.0.0' if shared.args.listen else '127.0.0.1', params['port'])
    server = ThreadingHTTPServer(server_addr, Handler)
    if shared.args.share:
        try:
            from flask_cloudflared import _run_cloudflared
            public_url = _run_cloudflared(params['port'], params['port'] + 1)
            print(f'Starting KoboldAI compatible api at {public_url}/api')
        except ImportError:
            print('You should install flask_cloudflared manually')
    else:
        print(
            f'Starting KoboldAI compatible api at http://{server_addr[0]}:{server_addr[1]}/api')
    server.serve_forever()


def setup():
    Thread(target=run_server, daemon=True).start()


def get_defaults(body):
    defaults = {
        'max_new_tokens':  int(200),
        'do_sample': bool(True),
        'temperature': float(0.5),
        'top_p': float(1),
        'typical_p': float(1),
        'repetition_penalty': float(1.1),
        'encoder_repetition_penalty': 1,
        'top_k': int(0),
        'min_length': int(0),
        'no_repeat_ngram_size': int(0),
        'num_beams': int(1),
        'penalty_alpha': float(0),
        'length_penalty': float(1),
        'early_stopping': bool(False),
        'seed': int(-1),
        'add_bos_token': int(True),
        'truncation_length': int(2048),
        'ban_eos_token': bool(False),
        'skip_special_tokens': bool(True),
        'custom_stopping_strings': '',  # leave this blank
        'stopping_strings':  [],
    }
    if body.get('preset'):
        preset_path =  str(Path.cwd())+"/presets/"+str(body.get('preset'))+'.txt'
        if os.path.exists(preset_path):
            print("[API]Loading preset: "+str(preset_path))
            with open(preset_path) as f:
                for line in f:
                    name, value = line.strip().split('=')
                    if value.isdigit():
                        value = int(value)
                    elif '.' in value:
                        value = float(value)
                    elif value.lower() == 'true' or value.lower() == 'false':
                        value = bool(value)
                    else:
                        value = str(value)
                    if value == '' or value is None:
                        value = defaults[name]
                    defaults[name] = value
        else:
            print("[API]Preset not found: "+str(preset_path))
    return defaults
