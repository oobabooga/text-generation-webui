import json
import time
from typing import Dict, List

from pydantic import BaseModel, Field


class GenerationOptions(BaseModel):
    preset: str | None = Field(default=None, description="The name of a file under text-generation-webui/presets (without the .yaml extension). The sampling parameters that get overwritten by this option are the keys in the default_preset() function in modules/presets.py.")
    dynatemp_low: float = 1
    dynatemp_high: float = 1
    dynatemp_exponent: float = 1
    smoothing_factor: float = 0
    smoothing_curve: float = 1
    min_p: float = 0
    top_k: int = 0
    typical_p: float = 1
    xtc_threshold: float = 0.1
    xtc_probability: float = 0
    epsilon_cutoff: float = 0
    eta_cutoff: float = 0
    tfs: float = 1
    top_a: float = 0
    top_n_sigma: float = 0
    dry_multiplier: float = 0
    dry_allowed_length: int = 2
    dry_base: float = 1.75
    repetition_penalty: float = 1
    encoder_repetition_penalty: float = 1
    no_repeat_ngram_size: int = 0
    repetition_penalty_range: int = 1024
    penalty_alpha: float = 0
    guidance_scale: float = 1
    mirostat_mode: int = 0
    mirostat_tau: float = 5
    mirostat_eta: float = 0.1
    prompt_lookup_num_tokens: int = 0
    max_tokens_second: int = 0
    do_sample: bool = True
    dynamic_temperature: bool = False
    temperature_last: bool = False
    auto_max_new_tokens: bool = False
    ban_eos_token: bool = False
    add_bos_token: bool = True
    skip_special_tokens: bool = True
    static_cache: bool = False
    truncation_length: int = 0
    seed: int = -1
    sampler_priority: List[str] | str | None = Field(default=None, description="List of samplers where the first items will appear first in the stack. Example: [\"top_k\", \"temperature\", \"top_p\"].")
    custom_token_bans: str = ""
    negative_prompt: str = ''
    dry_sequence_breakers: str = '"\\n", ":", "\\"", "*"'
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

    instruction_template: str | None = Field(default=None, description="An instruction template defined under text-generation-webui/instruction-templates. If not set, the correct template will be automatically obtained from the model metadata.")
    instruction_template_str: str | None = Field(default=None, description="A Jinja2 instruction template. If set, will take precedence over everything else.")

    character: str | None = Field(default=None, description="A character defined under text-generation-webui/characters. If not set, the default \"Assistant\" character will be used.")
    bot_name: str | None = Field(default=None, description="Overwrites the value set by character field.", alias="name2")
    context: str | None = Field(default=None, description="Overwrites the value set by character field.")
    greeting: str | None = Field(default=None, description="Overwrites the value set by character field.")
    user_name: str | None = Field(default=None, description="Your name (the user). By default, it's \"You\".", alias="name1")
    user_bio: str | None = Field(default=None, description="The user description/personality.")
    chat_template_str: str | None = Field(default=None, description="Jinja2 template for chat.")

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


class ChatPromptResponse(BaseModel):
    prompt: str


class EmbeddingsRequest(BaseModel):
    input: str | List[str] | List[int] | List[List[int]]
    model: str | None = Field(default=None, description="Unused parameter. To change the model, set the OPENEDAI_EMBEDDING_MODEL and OPENEDAI_EMBEDDING_DEVICE environment variables before starting the server.")
    encoding_format: str = Field(default="float", description="Can be float or base64.")
    user: str | None = Field(default=None, description="Unused parameter.")


class EmbeddingsResponse(BaseModel):
    index: int
    embedding: List[float]
    object: str = "embedding"


class EncodeRequest(BaseModel):
    text: str


class EncodeResponse(BaseModel):
    tokens: List[int]
    length: int


class DecodeRequest(BaseModel):
    tokens: List[int]


class DecodeResponse(BaseModel):
    text: str


class TokenCountResponse(BaseModel):
    length: int


class LogitsRequestParams(BaseModel):
    prompt: str
    use_samplers: bool = False
    top_logits: int | None = 50
    frequency_penalty: float | None = 0
    max_tokens: int | None = 16
    presence_penalty: float | None = 0
    temperature: float | None = 1
    top_p: float | None = 1


class LogitsRequest(GenerationOptions, LogitsRequestParams):
    pass


class LogitsResponse(BaseModel):
    logits: Dict[str, float]


class ModelInfoResponse(BaseModel):
    model_name: str
    lora_names: List[str]


class ModelListResponse(BaseModel):
    model_names: List[str]


class LoadModelRequest(BaseModel):
    model_name: str
    args: dict | None = None
    settings: dict | None = None


class LoraListResponse(BaseModel):
    lora_names: List[str]


class LoadLorasRequest(BaseModel):
    lora_names: List[str]


def to_json(obj):
    return json.dumps(obj.__dict__, indent=4)


def to_dict(obj):
    return obj.__dict__
