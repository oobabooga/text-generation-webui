import json
import os
import ssl
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import extensions.openai.completions as OAIcompletions
import extensions.openai.edits as OAIedits
import extensions.openai.embeddings as OAIembeddings
import extensions.openai.images as OAIimages
import extensions.openai.models as OAImodels
import extensions.openai.moderations as OAImoderations
from extensions.openai.defaults import clamp, default, get_default_req_params
from extensions.openai.errors import (
    InvalidRequestError,
    OpenAIError,
    ServiceUnavailableError
)
from extensions.openai.tokens import token_count, token_decode, token_encode
from extensions.openai.utils import debug_msg
from modules import shared

import cgi
import speech_recognition as sr
from pydub import AudioSegment

params = {
    # default params
    'port': 5001,
    'embedding_device': 'cpu',
    'embedding_model': 'all-mpnet-base-v2',
    
    # optional params
    'sd_webui_url': '',
    'debug': 0
}

class Handler(BaseHTTPRequestHandler):
    def send_access_control_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Credentials", "true")
        self.send_header(
            "Access-Control-Allow-Methods",
            "GET,HEAD,OPTIONS,POST,PUT"
        )
        self.send_header(
            "Access-Control-Allow-Headers",
            "Origin, Accept, X-Requested-With, Content-Type, "
            "Access-Control-Request-Method, Access-Control-Request-Headers, "
            "Authorization"
        )

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_access_control_headers()
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write("OK".encode('utf-8'))

    def start_sse(self):
        self.send_response(200)
        self.send_access_control_headers()
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        # self.send_header('Connection', 'keep-alive')
        self.end_headers()

    def send_sse(self, chunk: dict):
        response = 'data: ' + json.dumps(chunk) + '\r\n\r\n'
        debug_msg(response[:-4])
        self.wfile.write(response.encode('utf-8'))

    def end_sse(self):
        response = 'data: [DONE]\r\n\r\n'
        debug_msg(response[:-4])
        self.wfile.write(response.encode('utf-8'))

    def return_json(self, ret: dict, code: int = 200, no_debug=False):
        self.send_response(code)
        self.send_access_control_headers()
        self.send_header('Content-Type', 'application/json')

        response = json.dumps(ret)
        r_utf8 = response.encode('utf-8')

        self.send_header('Content-Length', str(len(r_utf8)))
        self.end_headers()

        self.wfile.write(r_utf8)
        if not no_debug:
            debug_msg(r_utf8)

    def openai_error(self, message, code=500, error_type='APIError', param='', internal_message=''):

        error_resp = {
            'error': {
                'message': message,
                'code': code,
                'type': error_type,
                'param': param,
            }
        }
        if internal_message:
            print(error_type, message)
            print(internal_message)
            # error_resp['internal_message'] = internal_message

        self.return_json(error_resp, code)

    def openai_error_handler(func):
        def wrapper(self):
            try:
                func(self)
            except InvalidRequestError as e:
                self.openai_error(e.message, e.code, e.__class__.__name__, e.param, internal_message=e.internal_message)
            except OpenAIError as e:
                self.openai_error(e.message, e.code, e.__class__.__name__, internal_message=e.internal_message)
            except Exception as e:
                self.openai_error(repr(e), 500, 'OpenAIError', internal_message=traceback.format_exc())

        return wrapper

    @openai_error_handler
    def do_GET(self):
        debug_msg(self.requestline)
        debug_msg(self.headers)

        if self.path.startswith('/v1/engines') or self.path.startswith('/v1/models'):
            is_legacy = 'engines' in self.path
            is_list = self.path.split('?')[0].split('#')[0] in ['/v1/engines', '/v1/models']
            if is_legacy and not is_list:
                model_name = self.path[self.path.find('/v1/engines/') + len('/v1/engines/'):]
                resp = OAImodels.load_model(model_name)
            elif is_list:
                resp = OAImodels.list_models(is_legacy)
            else:
                model_name = self.path[len('/v1/models/'):]
                resp = OAImodels.model_info(model_name)

            self.return_json(resp)

        elif '/billing/usage' in self.path:
            #  Ex. /v1/dashboard/billing/usage?start_date=2023-05-01&end_date=2023-05-31
            self.return_json({"total_usage": 0}, no_debug=True)

        else:
            self.send_error(404)

    @openai_error_handler
    def do_POST(self):

        if '/v1/audio/transcriptions' in self.path:
            r = sr.Recognizer()

            # Parse the form data
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={'REQUEST_METHOD': 'POST', 'CONTENT_TYPE': self.headers['Content-Type']}
            )
            
            audio_file = form['file'].file
            audio_data = AudioSegment.from_file(audio_file)
            
            # Convert AudioSegment to raw data
            raw_data = audio_data.raw_data
            
            # Create AudioData object
            audio_data = sr.AudioData(raw_data, audio_data.frame_rate, audio_data.sample_width)
            whipser_language = form.getvalue('language', None)
            whipser_model = form.getvalue('model', 'tiny')  # Use the model from the form data if it exists, otherwise default to tiny

            transcription = {"text": ""}
            
            try:
                transcription["text"] = r.recognize_whisper(audio_data, language=whipser_language, model=whipser_model)
            except sr.UnknownValueError:
                print("Whisper could not understand audio")
                transcription["text"] = "Whisper could not understand audio UnknownValueError"
            except sr.RequestError as e:
                print("Could not request results from Whisper", e)
                transcription["text"] = "Whisper could not understand audio RequestError"
            
            self.return_json(transcription, no_debug=True)
            return   
            
        debug_msg(self.requestline)
        debug_msg(self.headers)

        content_length = self.headers.get('Content-Length')
        transfer_encoding = self.headers.get('Transfer-Encoding')

        if content_length:
            body = json.loads(self.rfile.read(int(content_length)).decode('utf-8'))
        elif transfer_encoding == 'chunked':
            chunks = []
            while True:
                chunk_size = int(self.rfile.readline(), 16)  # Read the chunk size
                if chunk_size == 0:
                    break  # End of chunks
                chunks.append(self.rfile.read(chunk_size))
                self.rfile.readline()  # Consume the trailing newline after each chunk
            body = json.loads(b''.join(chunks).decode('utf-8'))
        else:
            self.send_response(400, "Bad Request: Either Content-Length or Transfer-Encoding header expected.")
            self.end_headers()
            return

        debug_msg(body)

        if '/completions' in self.path or '/generate' in self.path:

            if not shared.model:
                raise ServiceUnavailableError("No model loaded.")

            is_legacy = '/generate' in self.path
            is_streaming = body.get('stream', False)

            if is_streaming:
                self.start_sse()

                response = []
                if 'chat' in self.path:
                    response = OAIcompletions.stream_chat_completions(body, is_legacy=is_legacy)
                else:
                    response = OAIcompletions.stream_completions(body, is_legacy=is_legacy)

                for resp in response:
                    self.send_sse(resp)

                self.end_sse()

            else:
                response = ''
                if 'chat' in self.path:
                    response = OAIcompletions.chat_completions(body, is_legacy=is_legacy)
                else:
                    response = OAIcompletions.completions(body, is_legacy=is_legacy)

                self.return_json(response)

        elif '/edits' in self.path:
            # deprecated

            if not shared.model:
                raise ServiceUnavailableError("No model loaded.")

            req_params = get_default_req_params()

            instruction = body['instruction']
            input = body.get('input', '')
            temperature = clamp(default(body, 'temperature', req_params['temperature']), 0.001, 1.999)  # fixup absolute 0.0
            top_p = clamp(default(body, 'top_p', req_params['top_p']), 0.001, 1.0)

            response = OAIedits.edits(instruction, input, temperature, top_p)

            self.return_json(response)

        elif '/images/generations' in self.path:
            if not os.environ.get('SD_WEBUI_URL', params.get('sd_webui_url', '')):
                raise ServiceUnavailableError("Stable Diffusion not available. SD_WEBUI_URL not set.")

            prompt = body['prompt']
            size = default(body, 'size', '1024x1024')
            response_format = default(body, 'response_format', 'url')  # or b64_json
            n = default(body, 'n', 1)  # ignore the batch limits of max 10

            response = OAIimages.generations(prompt=prompt, size=size, response_format=response_format, n=n)

            self.return_json(response, no_debug=True)

        elif '/embeddings' in self.path:
            encoding_format = body.get('encoding_format', '')

            input = body.get('input', body.get('text', ''))
            if not input:
                raise InvalidRequestError("Missing required argument input", params='input')

            if type(input) is str:
                input = [input]

            response = OAIembeddings.embeddings(input, encoding_format)

            self.return_json(response, no_debug=True)

        elif '/moderations' in self.path:
            input = body['input']
            if not input:
                raise InvalidRequestError("Missing required argument input", params='input')

            response = OAImoderations.moderations(input)

            self.return_json(response, no_debug=True)

        elif self.path == '/api/v1/token-count':
            # NOT STANDARD. lifted from the api extension, but it's still very useful to calculate tokenized length client side.
            response = token_count(body['prompt'])

            self.return_json(response, no_debug=True)

        elif self.path == '/api/v1/token/encode':
            # NOT STANDARD. needed to support logit_bias, logprobs and token arrays for native models
            encoding_format = body.get('encoding_format', '')

            response = token_encode(body['input'], encoding_format)

            self.return_json(response, no_debug=True)

        elif self.path == '/api/v1/token/decode':
            # NOT STANDARD. needed to support logit_bias, logprobs and token arrays for native models
            encoding_format = body.get('encoding_format', '')

            response = token_decode(body['input'], encoding_format)

            self.return_json(response, no_debug=True)

        else:
            self.send_error(404)


def run_server():
    port = int(os.environ.get('OPENEDAI_PORT', params.get('port', 5001)))
    server_addr = ('0.0.0.0' if shared.args.listen else '127.0.0.1', port)
    server = ThreadingHTTPServer(server_addr, Handler)
    
    ssl_certfile=os.environ.get('OPENEDAI_CERT_PATH', shared.args.ssl_certfile)
    ssl_keyfile=os.environ.get('OPENEDAI_KEY_PATH', shared.args.ssl_keyfile)
    ssl_verify=True if (ssl_keyfile and ssl_certfile) else False
    if ssl_verify:        
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(ssl_certfile, ssl_keyfile)
        server.socket = context.wrap_socket(server.socket, server_side=True)
        
    if shared.args.share:
        try:
            from flask_cloudflared import _run_cloudflared
            public_url = _run_cloudflared(port, port + 1)
            print(f'OpenAI compatible API ready at: OPENAI_API_BASE={public_url}/v1')
        except ImportError:
            print('You should install flask_cloudflared manually')
    else:
        if ssl_verify:
            print(f'OpenAI compatible API ready at: OPENAI_API_BASE=https://{server_addr[0]}:{server_addr[1]}/v1')
        else:
            print(f'OpenAI compatible API ready at: OPENAI_API_BASE=http://{server_addr[0]}:{server_addr[1]}/v1')
    
    server.serve_forever()


def setup():
    Thread(target=run_server, daemon=True).start()
