import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from extensions.api.util import build_parameters, try_start_cloudflared
from modules import shared
from modules.chat import generate_chat_reply
from modules.text_generation import generate_reply
from extensions.api.shared_api import (
    ensureModelLoaded,
    _handle_stop_streaming_request,
    _handle_model_request,
    _handle_token_count_request
)
from modules.logging_colors import logger


def _handle_generate_request_blocking(connectionContext, message):
    if not ensureModelLoaded(connectionContext):
        return

    prompt = message['prompt']
    generate_params = build_parameters(message)
    stopping_strings = generate_params.pop('stopping_strings')
    generate_params['stream'] = False

    generator = generate_reply(
        prompt, generate_params, stopping_strings=stopping_strings, is_chat=False)

    answer = ''
    for a in generator:
        answer = a

    connectionContext['responseHandler'](connectionContext, {'results': [{'text': answer}]})

def _handle_chat_request_blocking(connectionContext, message):
    if not ensureModelLoaded(connectionContext):
        return

    user_input = message['user_input']
    regenerate = message.get('regenerate', False)
    _continue = message.get('_continue', False)

    generate_params = build_parameters(message, chat=True)
    generate_params['stream'] = False

    generator = generate_chat_reply(
        user_input, generate_params, regenerate=regenerate, _continue=_continue, loading_message=False)

    answer = generate_params['history']
    for a in generator:
        answer = a

    connectionContext['responseHandler'](connectionContext, { 'results': [{'history': answer}]})

HTTP_PATH_HANDLER_DICT = {
    '/api/v1/generate': {
        'handler': _handle_generate_request_blocking
    },
    '/api/v1/chat': {
        'handler': _handle_chat_request_blocking
    },
    '/api/v1/token-count': {
        'handler': _handle_token_count_request
    },
    '/api/v1/model': {
        'handler': _handle_model_request
    },
    '/api/v1/stop-stream': {
        'handler': _handle_stop_streaming_request
    }
}

class Handler(BaseHTTPRequestHandler):

    def SendClientResponse(self, connectionContext, message):
        try:
            connectionContext['http-handler'].send_response(200)
            connectionContext['http-handler'].end_headers()
            connectionContext['http-handler'].wfile.write(json.dumps(message).encode('utf-8'))
            return True
        except BrokenPipeError:
            logger.warning("client closed connection before the server could send a response")
            return False


    def do_GET(self):
        connectionContext = {
            'responseHandler': self.SendClientResponse,
            'http-handler': self
        }

        if self.path == '/api/v1/model':
            HTTP_PATH_HANDLER_DICT['/api/v1/model']['handler'](connectionContext, {})
        else:
            self.send_error(404)

    def do_POST(self):

        connectionContext = {
            'responseHandler': self.SendClientResponse,
            'http-handler': self
        }

        try:
            content_length = int(self.headers['Content-Length'])
            body = json.loads(self.rfile.read(content_length).decode('utf-8'))
        except ValueError:  # catch JSON parsing errors
            logger.warning("API request not handled, malformed JSON")
            self.SendClientResponse(connectionContext, {'event': 'error', 'message': 'malformed JSON data received'})
            return


        if HTTP_PATH_HANDLER_DICT.get(self.path):
            HTTP_PATH_HANDLER_DICT[self.path]['handler'](connectionContext, body)

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


def _run_server(port: int, share: bool = False, tunnel_id=str):
    address = '0.0.0.0' if shared.args.listen else '127.0.0.1'

    server = ThreadingHTTPServer((address, port), Handler)

    def on_start(public_url: str):
        for path in HTTP_PATH_HANDLER_DICT:
            logger.info("Starting HTTP API server at public url %s%s", public_url, path)

    if share:
        try:
            try_start_cloudflared(port, tunnel_id, max_attempts=3, on_start=on_start)
        except Exception:
            pass
    else:
        for path in HTTP_PATH_HANDLER_DICT:
            logger.info("Starting HTTP API at http://%s:%s%s", address, port, path)

    server.serve_forever()


def start_server(port: int, share: bool = False, tunnel_id=str):
    Thread(target=_run_server, args=[port, share, tunnel_id], daemon=True).start()
