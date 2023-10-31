import json
import time
from typing import List

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sse_starlette import EventSourceResponse

app = FastAPI()

class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str | List[str]
    best_of: int | None = 1
    echo: bool | None = False
    frequency_penalty: float | None = 0
    logit_bias: dict | None = None
    logprobs: int | None = None
    max_tokens: int | None = 16
    n: int | None = 1
    presence_penalty: int | None = 0
    stop: str | List[str] | None = None
    stream: bool | None = False
    suffix: str | None = None
    temperature: float | None = 1
    top_p: float | None = 1
    user: str | None = None


class CompletionResponse(BaseModel):
    id: str
    choices: List[dict]
    created: int = int(time.time())
    model: str
    object: str = "text_completion"
    usage: dict


class ChatCompletionRequest(BaseModel):
    messages: List[dict]
    model: str | None = None
    frequency_penalty: float | None = 0
    function_call: str | dict | None = None
    functions: List[dict] | None = None
    logit_bias: dict | None = None
    max_tokens: int | None = float('inf')
    n: int | None = 1
    presence_penalty: int | None = 0
    stop: str | List[str] | None = None
    stream: bool | None = False
    temperature: float | None = 1
    top_p: float | None = 1
    user: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    choices: List[dict]
    created: int = int(time.time())
    model: str
    object: str = "chat.completion"
    usage: dict


class ChatCompletionChunkResponse(BaseModel):
    id: str
    choices: List[dict]
    created: int = int(time.time())
    model: str
    object: str = "chat.completion.chunk"


class ChatCompletionChunkChoiceObject(BaseModel):
    delta: dict
    finish_reason: str | None = None
    index: int = 0


class ModelResponse(BaseModel):
    object: str = "list"
    data: List[dict]


class ModelObject(BaseModel):
    id: str
    created: int = int(time.time())
    object: str = "model"
    owned_by: str = "Open Source"


def to_json(obj):
    return json.dumps(obj.model_dump(), indent=4)


def to_dict(obj):
    return obj.model_dump()


# Define an OPTIONS endpoint to handle pre-flight requests
@app.route("/v1/completions", methods=["OPTIONS", "POST"])
async def openai_options(request: Request):
    response = JSONResponse(content={}, headers={"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "POST, OPTIONS"})
    return response


# Define an endpoint to handle the OpenAI Completions request
@app.post("/v1/completions")
async def openai_completions(request_data: CompletionRequest):
    if request_data.stream:
        async def event_generator():
            for _ in range(3):
                yield {
                    "event": "openai_response",
                    "data": response
                }

        return EventSourceResponse(event_generator())
    else:
        return JSONResponse(content=response)


# Define an endpoint to handle the OpenAI Completions request
@app.post("/v1/chat/completions")
async def openai_chat_completions(request_data: ChatCompletionRequest):
    if request_data.stream:
        choice = ChatCompletionChunkChoiceObject(
            delta={"role": "assistant", "content": "la"}
        )

        response = ChatCompletionChunkResponse(
            id="test",
            choices=[to_dict(choice)],
            model=""
        )

        async def event_generator():
            for _ in range(3):
                yield {"data": to_json(response)}

            # response.finish_reason = "stop"
            choice.finish_reason = "stop"
            response.choices = [to_dict(choice)]
            yield {"data": to_json(response)}
            return

        return EventSourceResponse(event_generator())
    else:
        response = ChatCompletionResponse()
        return JSONResponse(content=response)


@app.get("/v1/models")
async def model_response():
    object = ModelObject(id="")
    response = ModelResponse(
        data = [to_dict(object)]
    )

    return JSONResponse(content=to_dict(response))


def start_api():
    import uvicorn

    # Run the FastAPI application with Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
