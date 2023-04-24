import json
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock, Thread

from modules import shared
from modules.text_generation import encode, generate_reply

from extensions.api.util import build_parameters, try_start_cloudflared


class Handler(BaseHTTPRequestHandler):
    def __init__(self, lock, request, client_address, server):
        self.lock = lock
        super().__init__(request, client_address, server)
        
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

            generator = generate_reply(
                prompt, generate_params, stopping_strings=stopping_strings)
            
            def iterate_generator(generator):
                answer = ''
                for a in generator:
                    if isinstance(a, str):
                        answer = a
                    else:
                        answer = a[0]
                return answer
            
            if shared.args.thread_safe:
                with self.lock:
                    answer = iterate_generator(generator)
            else:
                answer = iterate_generator(generator)   

            response = json.dumps({
                'results': [{
                    'text': answer if shared.is_chat() else answer[len(prompt):]
                }]
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
        else:
            self.send_error(404)


def _run_server(port: int, share: bool=False):
    address = '0.0.0.0' if shared.args.listen else '127.0.0.1'

    lock = Lock()
    handler_threadsafe = partial(Handler,lock)
    server = ThreadingHTTPServer((address, port), handler_threadsafe)

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
