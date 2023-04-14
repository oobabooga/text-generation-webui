import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

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

            prompt = body["data"]["prompt"]
            prompt_lines = [k.strip() for k in prompt.split('\n')]

            max_context = body.get('max_context_length', 2048)

            while len(prompt_lines) >= 0 and len(encode('\n'.join(prompt_lines))) > max_context:
                prompt_lines.pop(0)

            prompt = '\n'.join(prompt_lines)
            generate_params = {
                'max_new_tokens': int(body["data"]["params"].get('max_length', 200)),
                'do_sample': bool(body["data"]["params"].get('do_sample', True)),
                'temperature': float(body["data"]["params"].get('temperature', 0.5)),
                'top_p': float(body["data"]["params"].get('top_p', 1)),
                'typical_p': float(body["data"]["params"].get('typical', 1)),
                'repetition_penalty': float(body["data"]["params"].get('rep_pen', 1.1)),
                'encoder_repetition_penalty': int(body["data"]["params"].get('encoder_repetition_penalty', 1)),
                'top_k': int(body["data"]["params"].get('top_k', 0)),
                'min_length': int(body["data"]["params"].get('min_length', 0)),
                'no_repeat_ngram_size': int(body["data"]["params"].get('no_repeat_ngram_size', 0)),
                'num_beams': int(body["data"]["params"].get('num_beams', 1)),
                'penalty_alpha': float(body["data"]["params"].get('penalty_alpha', 0)),
                'length_penalty': float(body["data"]["params"].get('length_penalty', 1)),
                'early_stopping': bool(body["data"]["params"].get('early_stopping', False)),
                'seed': int(body["data"]["params"].get('seed', -1)),
                'add_bos_token': int(body["data"]["params"].get('add_bos_token', True)),
                'custom_stopping_strings': body["data"]["params"].get('custom_stopping_strings', []),
                'truncation_length': int(body["data"]["params"].get('truncation_length', 2048)),
                'ban_eos_token': bool(body["data"]["params"].get('ban_eos_token', False)),
            }

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


def run_server():
    server_addr = ('0.0.0.0' if shared.args.listen else '127.0.0.1', params['port'])
    server = ThreadingHTTPServer(server_addr, Handler)
    if shared.args.share:
        try:
            from flask_cloudflared import _run_cloudflared
            public_url = _run_cloudflared(params['port'], params['port'] + 1)
            print(f'Starting API at {public_url}/api')
        except ImportError:
            print('You should install flask_cloudflared manually')
    else:
        print(f'Starting API at http://{server_addr[0]}:{server_addr[1]}/api')
    server.serve_forever()


def setup():
    Thread(target=run_server, daemon=True).start()
