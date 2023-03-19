from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from modules import shared
from modules.text_generation import generate_reply, encode
import json

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
            prompt_lines = [l.strip() for l in prompt.split('\n')]

            max_context = body.get('max_context_length', 2048)

            while len(prompt_lines) >= 0 and len(encode('\n'.join(prompt_lines))) > max_context:
                prompt_lines.pop(0)

            prompt = '\n'.join(prompt_lines)

            generator = generate_reply(
                question = prompt, 
                max_new_tokens = body.get('max_length', 200), 
                do_sample=True, 
                temperature=body.get('temperature', 0.5), 
                top_p=body.get('top_p', 1), 
                typical_p=body.get('typical', 1), 
                repetition_penalty=body.get('rep_pen', 1.1), 
                encoder_repetition_penalty=1, 
                top_k=body.get('top_k', 0), 
                min_length=0, 
                no_repeat_ngram_size=0, 
                num_beams=1, 
                penalty_alpha=0, 
                length_penalty=1,
                early_stopping=False,
            )

            answer = ''
            for a in generator:
                answer = a[0]

            response = json.dumps({
                'results': [{
                    'text': answer[len(prompt):]
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
            from flask_cloudflared import  _run_cloudflared
            public_url = _run_cloudflared(params['port'], params['port'] + 1)
            print(f'Starting KoboldAI compatible api at {public_url}/api')
        except ImportError:
            print('You should install flask_cloudflared manually')
    else:
        print(f'Starting KoboldAI compatible api at http://{server_addr[0]}:{server_addr[1]}/api')
    server.serve_forever()

def ui():
    Thread(target=run_server, daemon=True).start()