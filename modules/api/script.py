import asyncio
import json
import logging
import os
import socket
import threading
import traceback
from collections import deque
from threading import Thread

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from sse_starlette import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool

import modules.api.completions as OAIcompletions
import modules.api.logits as OAIlogits
import modules.api.models as OAImodels
import modules.api.anthropic as Anthropic
from .tokens import token_count, token_decode, token_encode
from .errors import OpenAIError
from .utils import _start_cloudflared
from modules import shared
from modules.logging_colors import logger
from modules.models import unload_model
from modules.text_generation import stop_everything_event  # used by /v1/internal/stop-generation

from .typing import (
    AnthropicRequest,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatPromptResponse,
    CompletionRequest,
    CompletionResponse,
    DecodeRequest,
    DecodeResponse,
    EmbeddingsRequest,
    EmbeddingsResponse,
    EncodeRequest,
    EncodeResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    LoadLorasRequest,
    LoadModelRequest,
    LogitsRequest,
    LogitsResponse,
    LoraListResponse,
    ModelInfoResponse,
    ModelListResponse,
    TokenCountResponse,
    to_dict
)


async def _wait_for_disconnect(request: Request, stop_event: threading.Event):
    """Block until the client disconnects, then signal the stop_event."""
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            stop_event.set()
            return


def verify_api_key(authorization: str = Header(None)) -> None:
    expected_api_key = shared.args.api_key
    if expected_api_key and (authorization is None or authorization != f"Bearer {expected_api_key}"):
        raise HTTPException(status_code=401, detail="Unauthorized")


def verify_admin_key(authorization: str = Header(None)) -> None:
    expected_api_key = shared.args.admin_key
    if expected_api_key and (authorization is None or authorization != f"Bearer {expected_api_key}"):
        raise HTTPException(status_code=401, detail="Unauthorized")


def verify_anthropic_key(x_api_key: str = Header(None, alias="x-api-key")) -> None:
    expected_api_key = shared.args.api_key
    if expected_api_key and (x_api_key is None or x_api_key != expected_api_key):
        raise HTTPException(status_code=401, detail="Unauthorized")


class AnthropicError(Exception):
    def __init__(self, message: str, error_type: str = "invalid_request_error", status_code: int = 400):
        self.message = message
        self.error_type = error_type
        self.status_code = status_code


app = FastAPI()
check_key = [Depends(verify_api_key)]
check_admin_key = [Depends(verify_admin_key)]
check_anthropic_key = [Depends(verify_anthropic_key)]

# Configure CORS settings to allow all origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.exception_handler(OpenAIError)
async def openai_error_handler(request: Request, exc: OpenAIError):
    error_type = "server_error" if exc.code >= 500 else "invalid_request_error"
    return JSONResponse(
        status_code=exc.code,
        content={"error": {
            "message": exc.message,
            "type": error_type,
            "param": getattr(exc, 'param', None),
            "code": None
        }}
    )


@app.exception_handler(AnthropicError)
async def anthropic_error_handler(request: Request, exc: AnthropicError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"type": "error", "error": {"type": exc.error_type, "message": exc.message}}
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    if request.url.path.startswith("/v1/messages"):
        messages = "; ".join(
            f"{'.'.join(str(l) for l in e['loc'])}: {e['msg']}" for e in exc.errors()
        )
        return JSONResponse(
            status_code=400,
            content={"type": "error", "error": {"type": "invalid_request_error", "message": messages}}
        )

    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.middleware("http")
async def validate_host_header(request: Request, call_next):
    # Be strict about only approving access to localhost by default
    if not (shared.args.listen or shared.args.public_api):
        host = request.headers.get("host", "").split(":")[0]
        if host not in ["localhost", "127.0.0.1"]:
            return JSONResponse(
                status_code=400,
                content={"detail": "Invalid host header"}
            )

    return await call_next(request)


@app.options("/", dependencies=check_key)
async def options_route():
    return JSONResponse(content="OK")


@app.post('/v1/completions', response_model=CompletionResponse, dependencies=check_key)
async def openai_completions(request: Request, request_data: CompletionRequest):
    path = request.url.path
    is_legacy = "/generate" in path

    if request_data.stream:
        if (request_data.n or 1) > 1:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "n > 1 is not supported with streaming.", "type": "invalid_request_error", "param": "n", "code": None}}
            )

        stop_event = threading.Event()

        async def generator():
            response = OAIcompletions.stream_completions(to_dict(request_data), is_legacy=is_legacy, stop_event=stop_event)
            try:
                async for resp in iterate_in_threadpool(response):
                    disconnected = await request.is_disconnected()
                    if disconnected:
                        break

                    yield {"data": json.dumps(resp)}

                yield {"data": "[DONE]"}
            finally:
                stop_event.set()
                response.close()

        return EventSourceResponse(generator(), sep="\n")  # SSE streaming

    else:
        stop_event = threading.Event()
        monitor = asyncio.create_task(_wait_for_disconnect(request, stop_event))
        try:
            response = await asyncio.to_thread(
                OAIcompletions.completions,
                to_dict(request_data),
                is_legacy=is_legacy,
                stop_event=stop_event
            )
        finally:
            stop_event.set()
            monitor.cancel()

        return JSONResponse(response)


@app.post('/v1/chat/completions', response_model=ChatCompletionResponse, dependencies=check_key)
async def openai_chat_completions(request: Request, request_data: ChatCompletionRequest):
    path = request.url.path
    is_legacy = "/generate" in path

    if request_data.stream:
        stop_event = threading.Event()

        async def generator():
            response = OAIcompletions.stream_chat_completions(to_dict(request_data), is_legacy=is_legacy, stop_event=stop_event)
            try:
                async for resp in iterate_in_threadpool(response):
                    disconnected = await request.is_disconnected()
                    if disconnected:
                        break

                    yield {"data": json.dumps(resp)}

                yield {"data": "[DONE]"}
            finally:
                stop_event.set()
                response.close()

        return EventSourceResponse(generator(), sep="\n")  # SSE streaming

    else:
        stop_event = threading.Event()
        monitor = asyncio.create_task(_wait_for_disconnect(request, stop_event))
        try:
            response = await asyncio.to_thread(
                OAIcompletions.chat_completions,
                to_dict(request_data),
                is_legacy=is_legacy,
                stop_event=stop_event
            )
        finally:
            stop_event.set()
            monitor.cancel()

        return JSONResponse(response)


@app.post('/v1/messages', dependencies=check_anthropic_key)
async def anthropic_messages(request: Request, request_data: AnthropicRequest):
    body = to_dict(request_data)
    model = body.get('model') or shared.model_name or 'unknown'

    try:
        converted = Anthropic.convert_request(body)
    except Exception as e:
        raise AnthropicError(message=str(e))

    try:
        return await _anthropic_generate(request, request_data, converted, model)
    except OpenAIError as e:
        error_type = "invalid_request_error" if e.code < 500 else "api_error"
        if e.code == 503:
            error_type = "overloaded_error"
        raise AnthropicError(message=e.message, error_type=error_type, status_code=e.code)
    except Exception as e:
        raise AnthropicError(message=str(e) or "Internal server error", error_type="api_error", status_code=500)


async def _anthropic_generate(request, request_data, converted, model):
    if request_data.stream:
        stop_event = threading.Event()

        async def generator():
            converter = Anthropic.StreamConverter(model)
            response = OAIcompletions.stream_chat_completions(converted, is_legacy=False, stop_event=stop_event)
            try:
                async for resp in iterate_in_threadpool(response):
                    disconnected = await request.is_disconnected()
                    if disconnected:
                        break

                    for event in converter.process_chunk(resp):
                        yield event

                for event in converter.finish():
                    yield event
            except OpenAIError as e:
                error_type = "invalid_request_error" if e.code < 500 else "api_error"
                if e.code == 503:
                    error_type = "overloaded_error"
                yield {
                    "event": "error",
                    "data": json.dumps({"type": "error", "error": {"type": error_type, "message": e.message}})
                }
            finally:
                stop_event.set()
                response.close()

        return EventSourceResponse(generator(), sep="\n")

    else:
        stop_event = threading.Event()
        monitor = asyncio.create_task(_wait_for_disconnect(request, stop_event))
        try:
            openai_resp = await asyncio.to_thread(
                OAIcompletions.chat_completions,
                converted,
                is_legacy=False,
                stop_event=stop_event
            )
        finally:
            stop_event.set()
            monitor.cancel()

        return JSONResponse(Anthropic.build_response(openai_resp, model))


@app.get("/v1/models", dependencies=check_key)
@app.get("/v1/models/{model}", dependencies=check_key)
async def handle_models(request: Request):
    path = request.url.path
    is_list = request.url.path.split('?')[0].split('#')[0] == '/v1/models'

    if is_list:
        response = OAImodels.list_models_openai_format()
    else:
        model_name = path[len('/v1/models/'):]
        response = OAImodels.model_info_dict(model_name)

    return JSONResponse(response)


@app.get('/v1/billing/usage', dependencies=check_key)
def handle_billing_usage():
    '''
    Ex. /v1/dashboard/billing/usage?start_date=2023-05-01&end_date=2023-05-31
    '''
    return JSONResponse(content={"total_usage": 0})


@app.post('/v1/audio/transcriptions', dependencies=check_key)
async def handle_audio_transcription(request: Request):
    import speech_recognition as sr
    from pydub import AudioSegment

    r = sr.Recognizer()

    form = await request.form()
    audio_file = await form["file"].read()
    audio_data = AudioSegment.from_file(audio_file)

    # Convert AudioSegment to raw data
    raw_data = audio_data.raw_data

    # Create AudioData object
    audio_data = sr.AudioData(raw_data, audio_data.frame_rate, audio_data.sample_width)
    whisper_language = form.getvalue('language', None)
    whisper_model = form.getvalue('model', 'tiny')  # Use the model from the form data if it exists, otherwise default to tiny

    transcription = {"text": ""}

    try:
        transcription["text"] = r.recognize_whisper(audio_data, language=whisper_language, model=whisper_model)
    except sr.UnknownValueError:
        print("Whisper could not understand audio")
        transcription["text"] = "Whisper could not understand audio UnknownValueError"
    except sr.RequestError as e:
        print("Could not request results from Whisper", e)
        transcription["text"] = "Whisper could not understand audio RequestError"

    return JSONResponse(content=transcription)


@app.post('/v1/images/generations', response_model=ImageGenerationResponse, dependencies=check_key)
async def handle_image_generation(request_data: ImageGenerationRequest):
    import modules.api.images as OAIimages

    response = await asyncio.to_thread(OAIimages.generations, request_data)
    return JSONResponse(response)


@app.post("/v1/embeddings", response_model=EmbeddingsResponse, dependencies=check_key)
async def handle_embeddings(request: Request, request_data: EmbeddingsRequest):
    import modules.api.embeddings as OAIembeddings

    input = request_data.input
    if not input:
        raise HTTPException(status_code=400, detail="Missing required argument input")

    if type(input) is str:
        input = [input]

    response = OAIembeddings.embeddings(input, request_data.encoding_format)
    return JSONResponse(response)


@app.post("/v1/moderations", dependencies=check_key)
async def handle_moderations(request: Request):
    import modules.api.moderations as OAImoderations

    body = await request.json()
    input = body["input"]
    if not input:
        raise HTTPException(status_code=400, detail="Missing required argument input")

    response = OAImoderations.moderations(input)
    return JSONResponse(response)


@app.get("/v1/internal/health", dependencies=check_key)
async def handle_health_check():
    return JSONResponse(content={"status": "ok"})


@app.post("/v1/internal/encode", response_model=EncodeResponse, dependencies=check_key)
async def handle_token_encode(request_data: EncodeRequest):
    response = token_encode(request_data.text)
    return JSONResponse(response)


@app.post("/v1/internal/decode", response_model=DecodeResponse, dependencies=check_key)
async def handle_token_decode(request_data: DecodeRequest):
    response = token_decode(request_data.tokens)
    return JSONResponse(response)


@app.post("/v1/internal/token-count", response_model=TokenCountResponse, dependencies=check_key)
async def handle_token_count(request_data: EncodeRequest):
    response = token_count(request_data.text)
    return JSONResponse(response)


@app.post("/v1/internal/logits", response_model=LogitsResponse, dependencies=check_key)
async def handle_logits(request_data: LogitsRequest):
    '''
    Given a prompt, returns the top 50 most likely logits as a dict.
    The keys are the tokens, and the values are the probabilities.
    '''
    response = OAIlogits._get_next_logits(to_dict(request_data))
    return JSONResponse(response)


@app.post('/v1/internal/chat-prompt', response_model=ChatPromptResponse, dependencies=check_key)
async def handle_chat_prompt(request: Request, request_data: ChatCompletionRequest):
    path = request.url.path
    is_legacy = "/generate" in path
    generator = OAIcompletions.chat_completions_common(to_dict(request_data), is_legacy=is_legacy, prompt_only=True)
    response = deque(generator, maxlen=1).pop()
    return JSONResponse(response)


@app.post("/v1/internal/stop-generation", dependencies=check_key)
async def handle_stop_generation(request: Request):
    stop_everything_event()
    return JSONResponse(content="OK")


@app.get("/v1/internal/model/info", response_model=ModelInfoResponse, dependencies=check_key)
async def handle_model_info():
    payload = OAImodels.get_current_model_info()
    return JSONResponse(content=payload)


@app.get("/v1/internal/model/list", response_model=ModelListResponse, dependencies=check_admin_key)
async def handle_list_models():
    payload = OAImodels.list_models()
    return JSONResponse(content=payload)


@app.post("/v1/internal/model/load", dependencies=check_admin_key)
async def handle_load_model(request_data: LoadModelRequest):
    '''
    The "args" parameter can be used to modify loader flags before loading
    a model. Example:

    ```
    "args": {
      "load_in_4bit": true,
      "n_gpu_layers": 12
    }
    ```

    Loader args are reset to their startup defaults between loads, so
    settings from a previous load do not leak into the next one.

    The "instruction_template" parameter sets the default instruction
    template by name (from user_data/instruction-templates/). The
    "instruction_template_str" parameter sets it as a raw Jinja2 string
    and takes precedence over "instruction_template".
    '''

    try:
        OAImodels._load_model(to_dict(request_data))
        return JSONResponse(content="OK")
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to load the model.")


@app.post("/v1/internal/model/unload", dependencies=check_admin_key)
async def handle_unload_model():
    try:
        unload_model()
        return JSONResponse(content="OK")
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to unload the model.")


@app.get("/v1/internal/lora/list", response_model=LoraListResponse, dependencies=check_admin_key)
async def handle_list_loras():
    response = OAImodels.list_loras()
    return JSONResponse(content=response)


@app.post("/v1/internal/lora/load", dependencies=check_admin_key)
async def handle_load_loras(request_data: LoadLorasRequest):
    try:
        OAImodels.load_loras(request_data.lora_names)
        return JSONResponse(content="OK")
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail="Failed to apply the LoRA(s).")


@app.post("/v1/internal/lora/unload", dependencies=check_admin_key)
async def handle_unload_loras():
    OAImodels.unload_all_loras()
    return JSONResponse(content="OK")


def find_available_port(starting_port):
    """Try the starting port, then find an available one if it's taken."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('', starting_port))
            return starting_port
    except OSError:
        # Port is already in use, so find a new one
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))  # Bind to port 0 to get an available port
            new_port = s.getsockname()[1]
            logger.warning(f"Port {starting_port} is already in use. Using port {new_port} instead.")
            return new_port


def run_server():
    # Parse configuration
    port = int(os.environ.get('OPENEDAI_PORT', shared.args.api_port))
    port = find_available_port(port)
    ssl_certfile = os.environ.get('OPENEDAI_CERT_PATH', shared.args.ssl_certfile)
    ssl_keyfile = os.environ.get('OPENEDAI_KEY_PATH', shared.args.ssl_keyfile)

    # In the server configuration:
    server_addrs = []
    if shared.args.listen and shared.args.listen_host:
        server_addrs.append(shared.args.listen_host)
    else:
        if os.environ.get('OPENEDAI_ENABLE_IPV6', shared.args.api_enable_ipv6):
            server_addrs.append('::' if shared.args.listen else '::1')
        if not os.environ.get('OPENEDAI_DISABLE_IPV4', shared.args.api_disable_ipv4):
            server_addrs.append('0.0.0.0' if shared.args.listen else '127.0.0.1')

    if not server_addrs:
        raise Exception('you MUST enable IPv6 or IPv4 for the API to work')

    # Log server information
    if shared.args.public_api:
        _start_cloudflared(
            port,
            shared.args.public_api_id,
            max_attempts=3,
            on_start=lambda url: logger.info(f'OpenAI/Anthropic-compatible API URL:\n\n{url}/v1\n')
        )
    else:
        url_proto = 'https://' if (ssl_certfile and ssl_keyfile) else 'http://'
        urls = [f'{url_proto}[{addr}]:{port}/v1' if ':' in addr else f'{url_proto}{addr}:{port}/v1' for addr in server_addrs]
        if len(urls) > 1:
            logger.info('OpenAI/Anthropic-compatible API URLs:\n\n' + '\n'.join(urls) + '\n')
        else:
            logger.info('OpenAI/Anthropic-compatible API URL:\n\n' + '\n'.join(urls) + '\n')

    # Log API keys
    if shared.args.api_key:
        if not shared.args.admin_key:
            shared.args.admin_key = shared.args.api_key

        logger.info(f'OpenAI API key:\n\n{shared.args.api_key}\n')

    if shared.args.admin_key and shared.args.admin_key != shared.args.api_key:
        logger.info(f'OpenAI API admin key (for loading/unloading models):\n\n{shared.args.admin_key}\n')

    # Start server
    logging.getLogger("uvicorn.error").propagate = False
    uvicorn.run(app, host=server_addrs, port=port, ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile, access_log=False)


_server_started = False


def setup():
    global _server_started
    if _server_started:
        return

    _server_started = True
    if shared.args.nowebui:
        run_server()
    else:
        Thread(target=run_server, daemon=True).start()
