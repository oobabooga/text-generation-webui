import asyncio
import json
import logging
import os
import traceback
from collections import deque
from threading import Thread

import speech_recognition as sr
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from pydub import AudioSegment
from sse_starlette import EventSourceResponse

import extensions.openai.completions as OAIcompletions
import extensions.openai.embeddings as OAIembeddings
import extensions.openai.images as OAIimages
import extensions.openai.logits as OAIlogits
import extensions.openai.models as OAImodels
import extensions.openai.moderations as OAImoderations
from extensions.openai.errors import ServiceUnavailableError
from extensions.openai.tokens import token_count, token_decode, token_encode
from extensions.openai.utils import _start_cloudflared
from modules import shared
from modules.logging_colors import logger
from modules.models import unload_model
from modules.text_generation import stop_everything_event

from .typing import (
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

params = {
    'embedding_device': 'cpu',
    'embedding_model': 'sentence-transformers/all-mpnet-base-v2',
    'sd_webui_url': '',
    'debug': 0
}


streaming_semaphore = asyncio.Semaphore(1)


def verify_api_key(authorization: str = Header(None)) -> None:
    expected_api_key = shared.args.api_key
    if expected_api_key and (authorization is None or authorization != f"Bearer {expected_api_key}"):
        raise HTTPException(status_code=401, detail="Unauthorized")


def verify_admin_key(authorization: str = Header(None)) -> None:
    expected_api_key = shared.args.admin_key
    if expected_api_key and (authorization is None or authorization != f"Bearer {expected_api_key}"):
        raise HTTPException(status_code=401, detail="Unauthorized")


app = FastAPI()
check_key = [Depends(verify_api_key)]
check_admin_key = [Depends(verify_admin_key)]

# Configure CORS settings to allow all origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.options("/", dependencies=check_key)
async def options_route():
    return JSONResponse(content="OK")


@app.post('/v1/completions', response_model=CompletionResponse, dependencies=check_key)
async def openai_completions(request: Request, request_data: CompletionRequest):
    path = request.url.path
    is_legacy = "/generate" in path

    if request_data.stream:
        async def generator():
            async with streaming_semaphore:
                response = OAIcompletions.stream_completions(to_dict(request_data), is_legacy=is_legacy)
                for resp in response:
                    disconnected = await request.is_disconnected()
                    if disconnected:
                        break

                    yield {"data": json.dumps(resp)}

        return EventSourceResponse(generator())  # SSE streaming

    else:
        response = OAIcompletions.completions(to_dict(request_data), is_legacy=is_legacy)
        return JSONResponse(response)


@app.post('/v1/chat/completions', response_model=ChatCompletionResponse, dependencies=check_key)
async def openai_chat_completions(request: Request, request_data: ChatCompletionRequest):
    path = request.url.path
    is_legacy = "/generate" in path

    if request_data.stream:
        async def generator():
            async with streaming_semaphore:
                response = OAIcompletions.stream_chat_completions(to_dict(request_data), is_legacy=is_legacy)
                for resp in response:
                    disconnected = await request.is_disconnected()
                    if disconnected:
                        break

                    yield {"data": json.dumps(resp)}

        return EventSourceResponse(generator())  # SSE streaming

    else:
        response = OAIcompletions.chat_completions(to_dict(request_data), is_legacy=is_legacy)
        return JSONResponse(response)


@app.get("/v1/models", dependencies=check_key)
@app.get("/v1/models/{model}", dependencies=check_key)
async def handle_models(request: Request):
    path = request.url.path
    is_list = request.url.path.split('?')[0].split('#')[0] == '/v1/models'

    if is_list:
        response = OAImodels.list_dummy_models()
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


@app.post('/v1/images/generations', dependencies=check_key)
async def handle_image_generation(request: Request):

    if not os.environ.get('SD_WEBUI_URL', params.get('sd_webui_url', '')):
        raise ServiceUnavailableError("Stable Diffusion not available. SD_WEBUI_URL not set.")

    body = await request.json()
    prompt = body['prompt']
    size = body.get('size', '1024x1024')
    response_format = body.get('response_format', 'url')  # or b64_json
    n = body.get('n', 1)  # ignore the batch limits of max 10

    response = await OAIimages.generations(prompt=prompt, size=size, response_format=response_format, n=n)
    return JSONResponse(response)


@app.post("/v1/embeddings", response_model=EmbeddingsResponse, dependencies=check_key)
async def handle_embeddings(request: Request, request_data: EmbeddingsRequest):
    input = request_data.input
    if not input:
        raise HTTPException(status_code=400, detail="Missing required argument input")

    if type(input) is str:
        input = [input]

    response = OAIembeddings.embeddings(input, request_data.encoding_format)
    return JSONResponse(response)


@app.post("/v1/moderations", dependencies=check_key)
async def handle_moderations(request: Request):
    body = await request.json()
    input = body["input"]
    if not input:
        raise HTTPException(status_code=400, detail="Missing required argument input")

    response = OAImoderations.moderations(input)
    return JSONResponse(response)


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
    This endpoint is experimental and may change in the future.

    The "args" parameter can be used to modify flags like "--load-in-4bit"
    or "--n-gpu-layers" before loading a model. Example:

    ```
    "args": {
      "load_in_4bit": true,
      "n_gpu_layers": 12
    }
    ```

    Note that those settings will remain after loading the model. So you
    may need to change them back to load a second model.

    The "settings" parameter is also a dict but with keys for the
    shared.settings object. It can be used to modify the default instruction
    template like this:

    ```
    "settings": {
      "instruction_template": "Alpaca"
    }
    ```
    '''

    try:
        OAImodels._load_model(to_dict(request_data))
        return JSONResponse(content="OK")
    except:
        traceback.print_exc()
        return HTTPException(status_code=400, detail="Failed to load the model.")


@app.post("/v1/internal/model/unload", dependencies=check_admin_key)
async def handle_unload_model():
    unload_model()


@app.get("/v1/internal/lora/list", response_model=LoraListResponse, dependencies=check_admin_key)
async def handle_list_loras():
    response = OAImodels.list_loras()
    return JSONResponse(content=response)


@app.post("/v1/internal/lora/load", dependencies=check_admin_key)
async def handle_load_loras(request_data: LoadLorasRequest):
    try:
        OAImodels.load_loras(request_data.lora_names)
        return JSONResponse(content="OK")
    except:
        traceback.print_exc()
        return HTTPException(status_code=400, detail="Failed to apply the LoRA(s).")


@app.post("/v1/internal/lora/unload", dependencies=check_admin_key)
async def handle_unload_loras():
    OAImodels.unload_all_loras()
    return JSONResponse(content="OK")


def run_server():
    # Parse configuration
    port = int(os.environ.get('OPENEDAI_PORT', shared.args.api_port))
    ssl_certfile = os.environ.get('OPENEDAI_CERT_PATH', shared.args.ssl_certfile)
    ssl_keyfile = os.environ.get('OPENEDAI_KEY_PATH', shared.args.ssl_keyfile)

    # In the server configuration:
    server_addrs = []
    if os.environ.get('OPENEDAI_ENABLE_IPV6', shared.args.api_enable_ipv6):
        server_addrs.append('[::]' if shared.args.listen else '[::1]')
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
            on_start=lambda url: logger.info(f'OpenAI-compatible API URL:\n\n{url}\n')
        )
    else:
        url_proto = 'https://' if (ssl_certfile and ssl_keyfile) else 'http://'
        urls = [f'{url_proto}{addr}:{port}' for addr in server_addrs]
        if len(urls) > 1:
            logger.info('OpenAI-compatible API URLs:\n\n' + '\n'.join(urls) + '\n')
        else:
            logger.info('OpenAI-compatible API URL:\n\n' + '\n'.join(urls) + '\n')

    # Log API keys
    if shared.args.api_key:
        if not shared.args.admin_key:
            shared.args.admin_key = shared.args.api_key

        logger.info(f'OpenAI API key:\n\n{shared.args.api_key}\n')

    if shared.args.admin_key and shared.args.admin_key != shared.args.api_key:
        logger.info(f'OpenAI API admin key (for loading/unloading models):\n\n{shared.args.admin_key}\n')

    # Start server
    logging.getLogger("uvicorn.error").propagate = False
    uvicorn.run(app, host=server_addrs, port=port, ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile)


def setup():
    if shared.args.nowebui:
        run_server()
    else:
        Thread(target=run_server, daemon=True).start()
