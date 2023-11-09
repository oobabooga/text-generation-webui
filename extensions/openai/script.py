import asyncio
import json
import os
import traceback
from threading import Thread

import extensions.openai.completions as OAIcompletions
import extensions.openai.embeddings as OAIembeddings
import extensions.openai.images as OAIimages
import extensions.openai.models as OAImodels
import extensions.openai.moderations as OAImoderations
import speech_recognition as sr
import uvicorn
from extensions.openai.errors import ServiceUnavailableError
from extensions.openai.tokens import token_count, token_decode, token_encode
from extensions.openai.utils import _start_cloudflared
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from modules import shared
from modules.logging_colors import logger
from modules.text_generation import stop_everything_event
from pydub import AudioSegment
from sse_starlette import EventSourceResponse

from .typing import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    DecodeRequest,
    DecodeResponse,
    EncodeRequest,
    EncodeResponse,
    LoadModelRequest,
    ModelInfoResponse,
    TokenCountResponse,
    to_dict
)

params = {
    'embedding_device': 'cpu',
    'embedding_model': 'all-mpnet-base-v2',
    'sd_webui_url': '',
    'debug': 0
}


streaming_semaphore = asyncio.Semaphore(1)


def verify_api_key(authorization: str = Header(None)) -> None:
    expected_api_key = shared.args.api_key
    if expected_api_key and (authorization is None or authorization != f"Bearer {expected_api_key}"):
        raise HTTPException(status_code=401, detail="Unauthorized")


app = FastAPI(dependencies=[Depends(verify_api_key)])

# Configure CORS settings to allow all origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "HEAD", "OPTIONS", "POST", "PUT"],
    allow_headers=[
        "Origin",
        "Accept",
        "X-Requested-With",
        "Content-Type",
        "Access-Control-Request-Method",
        "Access-Control-Request-Headers",
        "Authorization",
    ],
)


@app.options("/")
async def options_route():
    return JSONResponse(content="OK")


@app.post('/v1/completions', response_model=CompletionResponse)
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


@app.post('/v1/chat/completions', response_model=ChatCompletionResponse)
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


@app.get("/v1/models")
@app.get("/v1/models/{model}")
async def handle_models(request: Request):
    path = request.url.path
    is_list = request.url.path.split('?')[0].split('#')[0] == '/v1/models'

    if is_list:
        response = OAImodels.list_models()
    else:
        model_name = path[len('/v1/models/'):]
        response = OAImodels.model_info_dict(model_name)

    return JSONResponse(response)


@app.get('/v1/billing/usage')
def handle_billing_usage():
    '''
    Ex. /v1/dashboard/billing/usage?start_date=2023-05-01&end_date=2023-05-31
    '''
    return JSONResponse(content={"total_usage": 0})


@app.post('/v1/audio/transcriptions')
async def handle_audio_transcription(request: Request):
    r = sr.Recognizer()

    form = await request.form()
    audio_file = await form["file"].read()
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

    return JSONResponse(content=transcription)


@app.post('/v1/images/generations')
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


@app.post("/v1/embeddings")
async def handle_embeddings(request: Request):
    body = await request.json()
    encoding_format = body.get("encoding_format", "")

    input = body.get('input', body.get('text', ''))
    if not input:
        raise HTTPException(status_code=400, detail="Missing required argument input")

    if type(input) is str:
        input = [input]

    response = OAIembeddings.embeddings(input, encoding_format)
    return JSONResponse(response)


@app.post("/v1/moderations")
async def handle_moderations(request: Request):
    body = await request.json()
    input = body["input"]
    if not input:
        raise HTTPException(status_code=400, detail="Missing required argument input")

    response = OAImoderations.moderations(input)
    return JSONResponse(response)


@app.post("/v1/internal/encode", response_model=EncodeResponse)
async def handle_token_encode(request_data: EncodeRequest):
    response = token_encode(request_data.text)
    return JSONResponse(response)


@app.post("/v1/internal/decode", response_model=DecodeResponse)
async def handle_token_decode(request_data: DecodeRequest):
    response = token_decode(request_data.tokens)
    return JSONResponse(response)


@app.post("/v1/internal/token-count", response_model=TokenCountResponse)
async def handle_token_count(request_data: EncodeRequest):
    response = token_count(request_data.text)
    return JSONResponse(response)


@app.post("/v1/internal/stop-generation")
async def handle_stop_generation(request: Request):
    stop_everything_event()
    return JSONResponse(content="OK")


@app.get("/v1/internal/model/info", response_model=ModelInfoResponse)
async def handle_model_info():
    payload = OAImodels.get_current_model_info()
    return JSONResponse(content=payload)


@app.post("/v1/internal/model/load")
async def handle_load_model(request_data: LoadModelRequest):
    '''
    This endpoint is experimental and may change in the future.

    The "args" parameter can be used to modify flags like "--load-in-4bit"
    or "--n-gpu-layers" before loading a model. Example:

    "args": {
      "load_in_4bit": true,
      "n_gpu_layers": 12
    }

    Note that those settings will remain after loading the model. So you
    may need to change them back to load a second model.

    The "settings" parameter is also a dict but with keys for the
    shared.settings object. It can be used to modify the default instruction
    template like this:

    "settings": {
      "instruction_template": "Alpaca"
    }
    '''

    try:
        OAImodels._load_model(to_dict(request_data))
        return JSONResponse(content="OK")
    except:
        traceback.print_exc()
        return HTTPException(status_code=400, detail="Failed to load the model.")


def run_server():
    server_addr = '0.0.0.0' if shared.args.listen else '127.0.0.1'
    port = int(os.environ.get('OPENEDAI_PORT', shared.args.api_port))

    ssl_certfile = os.environ.get('OPENEDAI_CERT_PATH', shared.args.ssl_certfile)
    ssl_keyfile = os.environ.get('OPENEDAI_KEY_PATH', shared.args.ssl_keyfile)

    if shared.args.public_api:
        def on_start(public_url: str):
            logger.info(f'OpenAI compatible API URL:\n\n{public_url}/v1\n')

        _start_cloudflared(port, shared.args.public_api_id, max_attempts=3, on_start=on_start)
    else:
        if ssl_keyfile and ssl_certfile:
            logger.info(f'OpenAI compatible API URL:\n\nhttps://{server_addr}:{port}/v1\n')
        else:
            logger.info(f'OpenAI compatible API URL:\n\nhttp://{server_addr}:{port}/v1\n')

    if shared.args.api_key:
        logger.info(f'OpenAI API key:\n\n{shared.args.api_key}\n')

    uvicorn.run(app, host=server_addr, port=port, ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile)


def setup():
    Thread(target=run_server, daemon=True).start()
