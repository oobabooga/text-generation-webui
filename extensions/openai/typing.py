import json
import time
from typing import List

from pydantic import BaseModel


class GenerationOptions(BaseModel):
    preset: str = "None"
    temperature: float = 1
    top_p: float = 1
    min_p: float = 0
    top_k: int = 0
    repetition_penalty: float = 1
    presence_penalty: float = 0
    frequency_penalty: float = 0
    repetition_penalty_range: int = 0
    typical_p: float = 1
    tfs: float = 1
    top_a: float = 0
    epsilon_cutoff: float = 0
    eta_cutoff: float = 0
    guidance_scale: float = 1
    negative_prompt: str = ''
    penalty_alpha: float = 0
    mirostat_mode: int = 0
    mirostat_tau: float = 5
    mirostat_eta: float = 0.1
    do_sample: bool = True
    seed: int = -1
    encoder_repetition_penalty: float = 1
    no_repeat_ngram_size: int = 0
    min_length: int = 0
    num_beams: int = 1
    length_penalty: float = 1
    early_stopping: bool = False
    truncation_length: int = 0
    max_tokens_second: int = 0
    custom_stopping_strings: str = ""
    custom_token_bans: str = ""
    auto_max_new_tokens: bool = False
    ban_eos_token: bool = False
    add_bos_token: bool = True
    skip_special_tokens: bool = True
    grammar_string: str = ""


class CompletionRequest(GenerationOptions):
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


class ChatCompletionRequest(GenerationOptions):
    messages: List[dict]
    model: str | None = None
    frequency_penalty: float | None = 0
    function_call: str | dict | None = None
    functions: List[dict] | None = None
    logit_bias: dict | None = None
    max_tokens: int | None = None
    n: int | None = 1
    presence_penalty: int | None = 0
    stop: str | List[str] | None = None
    stream: bool | None = False
    temperature: float | None = 1
    top_p: float | None = 1
    user: str | None = None

    mode: str = 'instruct'

    instruction_template: str | None = None
    name1_instruct: str | None = None
    name2_instruct: str | None = None
    context_instruct: str | None = None
    turn_template: str | None = None

    character: str | None = None
    name1: str | None = None
    name2: str | None = None
    context: str | None = None
    greeting: str | None = None

    chat_instruct_command: str | None = None

    continue_: bool = False


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
    return json.dumps(obj.__dict__, indent=4)


def to_dict(obj):
    return obj.__dict__
