"""Loads model into the guidance library (https://github.com/microsoft/guidance).
It aims to reduce the entry barrier of using the guidance library with quantized models  

The easiest way to get started with this extension is by using the Python client wrapper:

https://github.com/ChuloAI/andromeda-chain

Example:

```
from andromeda_chain import AndromedaChain, AndromedaPrompt, AndromedaResponse
chain = AndromedaChain("http://0.0.0.0:9000/guidance_api/v1/generate")

prompt = AndromedaPrompt(
    name="hello",
    prompt_template="Howdy: {{gen 'expert_names' temperature=0 max_tokens=300}}",
    input_vars=[],
    output_vars=["expert_names"]
)

response: AndromedaResponse = chain.run_guidance_prompt(prompt)
```

"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import guidance
from modules import shared

# This extension depends on having the model already fully loaded, including LoRA

guidance_model = None

def load_guidance_model_singleton():
    global guidance_model

    if guidance_model:
        return guidance_model
    try:
        import guidance
    except ImportError:
        raise ImportError("Please run 'pip install guidance' before using the guidance extension.")
    
    if not shared.model or not shared.tokenizer:
        raise ValueError("Cannot use guidance extension without a pre-initialized model!")
    
    # For now only supports HF Transformers
    # As far as I know, this includes:
    #  - 8 and 4 bits quantizations loaded through bitsandbytes
    #  - GPTQ variants
    #  - Models with LoRAs

    guidance_model = guidance.llms.Transformers(
        model=shared.model,
        tokenizer=shared.tokenizer,
        device=shared.args.guidance_device
    )
    guidance.llm = guidance_model


class Handler(BaseHTTPRequestHandler):
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

            llm = load_guidance_model_singleton()
            guidance_program = guidance(prompt_template)
            program_result = guidance_program(
                **kwargs,
                stream=False,
                async_mode=False,
                caching=False,
                **input_vars,
                llm=llm,
            )
            output = {"__main__": str(program_result)}
            for output_var in output_vars:
                output[output_var] = program_result[output_var]
            

            response = json.dumps(output)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(response.encode('utf-8'))


def _run_server(port: int):
    address = '0.0.0.0' if shared.args.listen else '127.0.0.1'

    server = ThreadingHTTPServer((address, port), Handler)
    print(f'Starting Guidance API at http://{address}:{port}/guidance_api')

    server.serve_forever()


def start_server(port: int):
    Thread(target=_run_server, args=[port], daemon=True).start()
