import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from modules import shared
from modules.text_generation import encode, generate_reply
from modules.chat import chatbot_wrapper, save_history
from modules.extensions import apply_extensions

from extensions.api.util import build_parameters, try_start_cloudflared


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
        elif self.path == '/api/v1/chat':
            if shared.is_chat():
                if 'chat' in body:
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    action = body['chat']
                    # TODO: Need to use parameters better - couldn't load all of them from shared (input_params gr.State not accessible). Params from request override shared.
                    generate_params = {**shared.settings,**build_parameters(body)}
                    mode = generate_params['mode']
                    answer = ['','']
                    if action == 'generate':
                        prompt = body['prompt']
                        # Run chatbot using shared state (loaded from UI) to update shared.history and return cleaned message history
                        generator = chatbot_wrapper(
                            prompt, generate_params)
                        for a in generator:
                            # Latest reply message in shared.history['visible']
                            answer = a[-1]
                        # Save shared.history state
                        save_history(mode)
                    elif action == 'regenerate':
                        # Based on chat.regenerate_wrapper
                        generator = chatbot_wrapper(
                            '', generate_params, regenerate=True)
                        for a in generator:
                            answer = a[-1]
                        save_history(mode)
                    elif action == 'continue':
                        # Based on chat.continue_wrapper
                        generator = chatbot_wrapper(
                            '', generate_params, _continue=True)
                        for a in generator:
                            answer = a[-1]
                        save_history(mode)
                    elif action == 'clear-history':
                        # Attempted to use clear_chat_log directly. Failed with type error.
                        # Based on chat.clear_chat_log([shared.gradio[k] for k in ['name1', 'name2', 'greeting', 'mode']])
                        shared.history['visible'] = []
                        shared.history['internal'] = []
                        greeting = generate_params['greeting']
                        if greeting != '':
                            shared.history['internal'] += [['<|BEGIN-VISIBLE-CHAT|>', greeting]]
                            shared.history['visible'] += [['', apply_extensions("output", greeting)]]
                        # Save cleared logs
                        save_history(mode)
                        print("Chat API: History cleared")
                        
                    response = json.dumps({
                        'results': [{
                            'text': answer[1],
                            'input': answer[0]
                        }]
                    })
                    self.wfile.write(response.encode('utf-8'))
                else:
                    self.send_error(405)
                    print("Error: Chat API requested but 'chat': <action> is not set in request")
            else:
                self.send_error(405)
                print('Error: Chat API requested while chat interface mode is not active. ')
        else:
            self.send_error(404)


def _run_server(port: int, share: bool=False):
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
