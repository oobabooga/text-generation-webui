import json
import time
from typing import List

from pydantic import BaseModel, Field


class GenerationOptions(BaseModel):
    preset: str | None = Field(default=None, description="The name of a file under text-generation-webui/presets (without the .yaml extension). The sampling parameters that get overwritten by this option are the keys in the default_preset() function in modules/presets.py.")
    min_p: float = 0
    top_k: int = 0
    repetition_penalty: float = 1
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
    temperature_last: bool = False
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
    custom_token_bans: str = ""
    auto_max_new_tokens: bool = False
    ban_eos_token: bool = False
    add_bos_token: bool = True
    skip_special_tokens: bool = True
    grammar_string: str = ""


class CompletionRequestParams(BaseModel):
    model: str | None = Field(default=None, description="Unused parameter. To change the model, use the /v1/internal/model/load endpoint.")
    prompt: str | List[str]
    best_of: int | None = Field(default=1, description="Unused parameter.")
    echo: bool | None = False
    frequency_penalty: float | None = 0
    logit_bias: dict | None = None
    logprobs: int | None = None
    max_tokens: int | None = 16
    n: int | None = Field(default=1, description="Unused parameter.")
    presence_penalty: float | None = 0
    stop: str | List[str] | None = None
    stream: bool | None = False
    suffix: str | None = None
    temperature: float | None = 1
    top_p: float | None = 1
    user: str | None = Field(default=None, description="Unused parameter.")


class CompletionRequest(GenerationOptions, CompletionRequestParams):
    pass


class CompletionResponse(BaseModel):
    id: str
    choices: List[dict]
    created: int = int(time.time())
    model: str
    object: str = "text_completion"
    usage: dict


class ChatCompletionRequestParams(BaseModel):
    messages: List[dict]
    model: str | None = Field(default=None, description="Unused parameter. To change the model, use the /v1/internal/model/load endpoint.")
    frequency_penalty: float | None = 0
    function_call: str | dict | None = Field(default=None, description="Unused parameter.")
    functions: List[dict] | None = Field(default=None, description="Unused parameter.")
    logit_bias: dict | None = None
    max_tokens: int | None = None
    n: int | None = Field(default=1, description="Unused parameter.")
    presence_penalty: float | None = 0
    stop: str | List[str] | None = None
    stream: bool | None = False
    temperature: float | None = 1
    top_p: float | None = 1
    user: str | None = Field(default=None, description="Unused parameter.")

    mode: str = Field(default='instruct', description="Valid options: instruct, chat, chat-instruct.")

    instruction_template: str | None = Field(default=None, description="An instruction template defined under text-generation-webui/instruction-templates. If not set, the correct template will be guessed using the regex expressions in models/config.yaml.")
    turn_template: str | None = Field(default=None, description="Overwrites the value set by instruction_template.")
    name1_instruct: str | None = Field(default=None, description="Overwrites the value set by instruction_template.")
    name2_instruct: str | None = Field(default=None, description="Overwrites the value set by instruction_template.")
    context_instruct: str | None = Field(default=None, description="Overwrites the value set by instruction_template.")
    system_message: str | None = Field(default=None, description="Overwrites the value set by instruction_template.")

    character: str | None = Field(default=None, description="A character defined under text-generation-webui/characters. If not set, the default \"Assistant\" character will be used.")
    name1: str | None = Field(default=None, description="Your name (the user). By default, it's \"You\".")
    name2: str | None = Field(default=None, description="Overwrites the value set by character.")
    context: str | None = Field(default=None, description="Overwrites the value set by character.")
    greeting: str | None = Field(default=None, description="Overwrites the value set by character.")

    chat_instruct_command: str | None = None

    continue_: bool = Field(default=False, description="Makes the last bot message in the history be continued instead of starting a new message.")


class ChatCompletionRequest(GenerationOptions, ChatCompletionRequestParams):
    pass


class ChatCompletionResponse(BaseModel):
    id: str
    choices: List[dict]
    created: int = int(time.time())
    model: str
    object: str = "chat.completion"
    usage: dict


class EncodeRequest(BaseModel):
    text: str


class DecodeRequest(BaseModel):
    tokens: List[int]


class EncodeResponse(BaseModel):
    tokens: List[int]
    length: int


class DecodeResponse(BaseModel):
    text: str


class TokenCountResponse(BaseModel):
    length: int


class ModelInfoResponse(BaseModel):
    model_name: str
    lora_names: List[str]


class LoadModelRequest(BaseModel):
    model_name: str
    args: dict | None = None
    settings: dict | None = None


class EmbeddingsRequest(BaseModel):
    input: str | List[str]
    model: str | None = Field(default=None, description="Unused parameter. To change the model, set the OPENEDAI_EMBEDDING_MODEL and OPENEDAI_EMBEDDING_DEVICE environment variables before starting the server.")
    encoding_format: str = Field(default="float", description="Can be float or base64.")
    user: str | None = Field(default=None, description="Unused parameter.")


class EmbeddingsResponse(BaseModel):
    index: int
    embedding: List[float]
    object: str = "embedding"


def to_json(obj):
    return json.dumps(obj.__dict__, indent=4)


def to_dict(obj):
    return obj.__dict__
