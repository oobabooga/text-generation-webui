import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import guidance
from modules import shared

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

        if self.path == '/guidance_api/v1/generate':
            # TODO: add request validation
            # For now disabled to avoid an extra dependency on validation libraries, like Pydantic

            prompt_template = body["prompt_template"]
            input_vars = body["input_vars"]
            kwargs = body["guidance_kwargs"]
            output_vars = body["output_vars"]

            guidance_program = guidance(prompt_template)
            program_result = guidance_program(
                **kwargs,
                stream=False,
                async_mode=False,
                caching=False,
                **input_vars,
                llm=shared.guidance_model,
            )
            output = {"__main__": str(program_result)}
            for output_var in output_vars:
                output[output_var] = program_result[output_var]
            
            response = json.dumps(output)
            self.wfile.write(response.encode('utf-8'))


def _run_server(port: int):
    address = '0.0.0.0' if shared.args.listen else '127.0.0.1'

    server = ThreadingHTTPServer((address, port), Handler)
    print(f'Starting API at http://{address}:{port}/api')

    server.serve_forever()


def start_server(port: int):
    if not shared.guidance_model:
        raise ValueError("Guidance model was not properly initialized. Cannot start guidance extension.")

    Thread(target=_run_server, args=[port], daemon=True).start()
