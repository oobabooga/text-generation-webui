import cgi
import json
import os
import ssl
import threading
import time
import traceback
from threading import Thread
from typing import Optional
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer

import speech_recognition as sr
import uvicorn
from extensions.openai.completions import (
    chat_completions,
    completions,
    stream_chat_completions,
    stream_completions
)
from extensions.openai.defaults import clamp, default, get_default_req_params
from extensions.openai.edits import edits
from extensions.openai.embeddings import embeddings
from extensions.openai.errors import (
    InvalidRequestError,
    OpenAIError,
    ServiceUnavailableError
)
from extensions.openai.images import generations
from extensions.openai.models import list_models, load_model, model_info
from extensions.openai.moderations import moderations
from extensions.openai.tokens import token_count, token_decode, token_encode
from extensions.openai.utils import debug_msg
from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from flask_cloudflared import _run_cloudflared
from modules import shared
from pydub import AudioSegment
from starlette.responses import StreamingResponse
from werkzeug.serving import make_server

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

params = {
    # default params
    'port': 5001,
    'embedding_device': 'cpu',
    'embedding_model': 'all-mpnet-base-v2',

    # optional params
    'sd_webui_url': '',
    'debug': 0
}


@app.on_event("startup")
async def startup_event():
    app.state.background_task = BackgroundTasks()


@app.post("/v1/audio/transcriptions")
async def handle_audio_transcription(request: Request):
    r = sr.Recognizer()

    form = await request.form()
    audio_file = await form["file"].read()
    audio_data = AudioSegment.from_file(audio_file)

    # Convert AudioSegment to raw data
    raw_data = audio_data.raw_data

    # Create AudioData object
    audio_data = sr.AudioData(raw_data, audio_data.frame_rate, audio_data.sample_width)
    whispher_language = form.get("language", None)
    whispher_model = form.get("model", "tiny")  # Use the model from the form data if it exists, otherwise default to tiny

    transcription = {"text": ""}

    try:
        transcription["text"] = r.recognize_whisper(audio_data, language=whispher_language, model=whispher_model)
    except sr.UnknownValueError:
        print("Whisper could not understand audio")
        transcription["text"] = "Whisper could not understand audio UnknownValueError"
    except sr.RequestError as e:
        print("Could not request results from Whisper", e)
        transcription["text"] = "Whisper could not understand audio RequestError"

    return JSONResponse(transcription, no_debug=True)


@app.route("/v1/engines/{engine}")
async def handle_engines(engine: str):
    is_legacy = engine != "models"
    resp = list_models(is_legacy)
    return JSONResponse(resp)


@app.route("/v1/models/{model_name}")
async def handle_models(model_name: str):
    resp = model_info(model_name)
    return JSONResponse(resp)


@app.route("/v1/billing/usage")
async def handle_billing_usage():
    return JSONResponse({"total_usage": 0})


@app.post("/v1/completions")
async def handle_completions(request: Request):
    body = await request.json()
    is_legacy = "/generate" in request.url.path
    is_streaming = body.get("stream", False)

    if is_streaming:
        async with request.stream() as stream:
            while True:
                chunk = await stream.read()
                if not chunk:
                    break
                yield chunk
    else:
        if "chat" in request.url.path:
            response = await chat_completions(body, is_legacy=is_legacy)
        else:
            response = await completions(body, is_legacy=is_legacy)

        return JSONResponse(response)


@app.post("/v1/edits")
async def handle_edits(request: Request):
    body = await request.json()
    instruction = body["instruction"]
    input = body.get("input", "")
    temperature = clamp(default(body, "temperature", get_default_req_params()["temperature"]), 0.001, 1.999)  # fixup absolute 0.0
    top_p = clamp(default(body, "top_p", get_default_req_params()["top_p"]), 0.001, 1.0)

    response = await edits(instruction, input, temperature, top_p)
    return JSONResponse(response)


@app.post("/v1/images/generations")
async def handle_image_generation(request: Request):
    body = await request.json()
    prompt = body["prompt"]
    size = default(body, "size", "1024x1024")
    response_format = default(body, "response_format", "url")  # or b64_json
    n = default(body, "n", 1)  # ignore the batch limits of max 10

    response = await generations(prompt=prompt, size=size, response_format=response_format, n=n)
    return JSONResponse(response, no_debug=True)


@app.post("/v1/embeddings")
async def handle_embeddings(request: Request):
    body = await request.json()
    encoding_format = body.get("encoding_format", "")

    input = body.get("input", body.get("text", ""))
    if not input:
        raise HTTPException(status_code=400, detail="Missing required argument input")

    if type(input) is str:
        input = [input]

    response = await embeddings(input, encoding_format)
    return JSONResponse(response, no_debug=True)


@app.post("/v1/moderations")
async def handle_moderations(request: Request):
    body = await request.json()
    input = body["input"]
    if not input:
        raise HTTPException(status_code=400, detail="Missing required argument input")

    response = await moderations(input)
    return JSONResponse(response, no_debug=True)


@app.post("/api/v1/token-count")
async def handle_token_count(request: Request):
    body = await request.json()
    response = token_count(body["prompt"])
    return JSONResponse(response, no_debug=True)


@app.post("/api/v1/token/encode")
async def handle_token_encode(request: Request):
    body = await request.json()
    encoding_format = body.get("encoding_format", "")
    response = token_encode(body["input"], encoding_format)
    return JSONResponse(response, no_debug=True)


@app.post("/api/v1/token/decode")
async def handle_token_decode(request: Request):
    body = await request.json()
    encoding_format = body.get("encoding_format", "")
    response = token_decode(body["input"], encoding_format)
    return JSONResponse(response, no_debug=True)


def run_server():
    port = int(os.environ.get('OPENEDAI_PORT', params.get('port', 5001)))
    ssl_certfile=os.environ.get('OPENEDAI_CERT_PATH', shared.args.ssl_certfile)
    ssl_keyfile=os.environ.get('OPENEDAI_KEY_PATH', shared.args.ssl_keyfile)
    ssl_verify=True if (ssl_keyfile and ssl_certfile) else False

    if shared.args.share:
        public_url=_run_cloudflared(port, port + 1)
        print(f'OpenAI compatible API ready at: OPENAI_API_BASE={public_url}/v1')
    else:
        if ssl_verify:
            print(f'OpenAI compatible API ready at: OPENAI_API_BASE=https://localhost:{port}/v1')
        else:
            print(f'OpenAI compatible API ready at: OPENAI_API_BASE=http://localhost:{port}/v1')

    uvicorn.run(app, host="0.0.0.0", port=port, ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile)


def setup():
    Thread(target=run_server, daemon=True).start()
