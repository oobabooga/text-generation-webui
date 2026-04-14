import json
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator, validator

from modules import shared


class GenerationOptions(BaseModel):
    preset: str | None = Field(default=None, description="The name of a file under textgen/user_data/presets (without the .yaml extension). The sampling parameters that get overwritten by this option are the keys in the default_preset() function in modules/presets.py.")
    dynatemp_low: float = shared.args.dynatemp_low
    dynatemp_high: float = shared.args.dynatemp_high
    dynatemp_exponent: float = shared.args.dynatemp_exponent
    smoothing_factor: float = shared.args.smoothing_factor
    smoothing_curve: float = shared.args.smoothing_curve
    min_p: float = shared.args.min_p
    top_k: int = shared.args.top_k
    typical_p: float = shared.args.typical_p
    xtc_threshold: float = shared.args.xtc_threshold
    xtc_probability: float = shared.args.xtc_probability
    epsilon_cutoff: float = shared.args.epsilon_cutoff
    eta_cutoff: float = shared.args.eta_cutoff
    tfs: float = shared.args.tfs
    top_a: float = shared.args.top_a
    top_n_sigma: float = shared.args.top_n_sigma
    adaptive_target: float = shared.args.adaptive_target
    adaptive_decay: float = shared.args.adaptive_decay
    dry_multiplier: float = shared.args.dry_multiplier
    dry_allowed_length: int = shared.args.dry_allowed_length
    dry_base: float = shared.args.dry_base
    repetition_penalty: float = shared.args.repetition_penalty
    encoder_repetition_penalty: float = shared.args.encoder_repetition_penalty
    no_repeat_ngram_size: int = shared.args.no_repeat_ngram_size
    repetition_penalty_range: int = shared.args.repetition_penalty_range
    penalty_alpha: float = shared.args.penalty_alpha
    guidance_scale: float = shared.args.guidance_scale
    mirostat_mode: int = shared.args.mirostat_mode
    mirostat_tau: float = shared.args.mirostat_tau
    mirostat_eta: float = shared.args.mirostat_eta
    prompt_lookup_num_tokens: int = 0
    max_tokens_second: int = 0
    do_sample: bool = shared.args.do_sample
    dynamic_temperature: bool = shared.args.dynamic_temperature
    temperature_last: bool = shared.args.temperature_last
    auto_max_new_tokens: bool = False
    ban_eos_token: bool = False
    add_bos_token: bool = True
    enable_thinking: bool = shared.args.enable_thinking
    reasoning_effort: str = shared.args.reasoning_effort
    skip_special_tokens: bool = True
    static_cache: bool = False
    truncation_length: int = 0
    seed: int = -1
    sampler_priority: List[str] | str | None = Field(default=shared.args.sampler_priority, description="List of samplers where the first items will appear first in the stack. Example: [\"top_k\", \"temperature\", \"top_p\"].")
    custom_token_bans: str = ""
    negative_prompt: str = ''
    dry_sequence_breakers: str = shared.args.dry_sequence_breakers
    grammar_string: str = ""


class ToolDefinition(BaseModel):
    function: 'ToolFunction'
    type: str


class ToolFunction(BaseModel):
    model_config = ConfigDict(extra='allow')
    description: Optional[str] = None
    name: str
    parameters: Optional['ToolParameters'] = None


class ToolParameters(BaseModel):
    model_config = ConfigDict(extra='allow')
    properties: Optional[Dict[str, Any]] = None
    required: Optional[list[str]] = None
    type: str
    description: Optional[str] = None



class FunctionCall(BaseModel):
    name: str
    arguments: Optional[str] = None
    parameters: Optional[str] = None

    @validator('arguments', allow_reuse=True)
    def checkPropertyArgsOrParams(cls, v, values, **kwargs):
        if not v and not values.get('parameters'):
            raise ValueError("At least one of 'arguments' or 'parameters' must be provided as property in FunctionCall type")
        return v


class ToolCall(BaseModel):
    id: str
    index: int
    type: str
    function: FunctionCall


class StreamOptions(BaseModel):
    include_usage: bool | None = False


class CompletionRequestParams(BaseModel):
    model: str | None = Field(default=None, description="Unused parameter. To change the model, use the /v1/internal/model/load endpoint.")
    prompt: str | List[str] | None = Field(default=None, description="Text prompt for completion. Can also use 'messages' format for multimodal.")
    messages: List[dict] | None = Field(default=None, description="OpenAI messages format for multimodal support. Alternative to 'prompt'.")
    best_of: int | None = Field(default=1, description="Unused parameter.")
    echo: bool | None = False
    frequency_penalty: float | None = shared.args.frequency_penalty
    logit_bias: dict | None = None
    logprobs: int | None = None
    max_tokens: int | None = 512
    n: int | None = Field(default=1, description="Number of completions to generate. Only supported without streaming.")
    presence_penalty: float | None = shared.args.presence_penalty
    stop: str | List[str] | None = None
    stream: bool | None = False
    stream_options: StreamOptions | None = None
    suffix: str | None = None
    temperature: float | None = shared.args.temperature
    top_p: float | None = shared.args.top_p
    user: str | None = Field(default=None, description="Unused parameter.")

    @model_validator(mode='after')
    def validate_prompt_or_messages(self):
        if self.prompt is None and self.messages is None:
            raise ValueError("Either 'prompt' or 'messages' must be provided")
        return self


class CompletionRequest(GenerationOptions, CompletionRequestParams):
    pass


class CompletionResponse(BaseModel):
    id: str
    choices: List[dict]
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    object: str = "text_completion"
    usage: dict


class ChatCompletionRequestParams(BaseModel):
    messages: List[dict] = Field(..., min_length=1)
    model: str | None = Field(default=None, description="Unused parameter. To change the model, use the /v1/internal/model/load endpoint.")
    frequency_penalty: float | None = shared.args.frequency_penalty
    function_call: str | dict | None = Field(default=None, description="Unused parameter.")
    functions: List[dict] | None = Field(default=None, description="Unused parameter.")
    tools: List[dict] | None = Field(default=None, description="Tools signatures passed via MCP.")
    tool_choice: str | dict | None = Field(default=None, description="Controls tool use: 'auto', 'none', 'required', or {\"type\": \"function\", \"function\": {\"name\": \"...\"}}.")
    logit_bias: dict | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    n: int | None = Field(default=1, description="Unused parameter.")
    presence_penalty: float | None = shared.args.presence_penalty
    stop: str | List[str] | None = None
    stream: bool | None = False
    stream_options: StreamOptions | None = None
    temperature: float | None = shared.args.temperature
    top_p: float | None = shared.args.top_p
    user: str | None = Field(default=None, description="Unused parameter.")

    @model_validator(mode='after')
    def resolve_max_tokens(self):
        if self.max_tokens is None and self.max_completion_tokens is not None:
            self.max_tokens = self.max_completion_tokens
        return self

    mode: str = Field(default='instruct', description="Valid options: instruct, chat, chat-instruct.")

    instruction_template: str | None = Field(default=None, description="An instruction template defined under textgen/user_data/instruction-templates. If not set, the correct template will be automatically obtained from the model metadata.")
    instruction_template_str: str | None = Field(default=None, description="A Jinja2 instruction template. If set, will take precedence over everything else.")

    character: str | None = Field(default=None, description="A character defined under textgen/user_data/characters. If not set, the default \"Assistant\" character will be used.")
    bot_name: str | None = Field(default=None, description="Overwrites the value set by character field.", alias="name2")
    context: str | None = Field(default=None, description="Overwrites the value set by character field.")
    greeting: str | None = Field(default=None, description="Overwrites the value set by character field.")
    user_name: str | None = Field(default=None, description="Your name (the user). By default, it's \"You\".", alias="name1")
    user_bio: str | None = Field(default=None, description="The user description/personality.")
    chat_template_str: str | None = Field(default=None, description="Jinja2 template for chat.")

    chat_instruct_command: str | None = "Continue the chat dialogue below. Write a single reply for the character \"<|character|>\".\n\n<|prompt|>"

    continue_: bool = Field(default=False, description="Makes the last bot message in the history be continued instead of starting a new message.")


class ChatCompletionRequest(GenerationOptions, ChatCompletionRequestParams):
    pass


class ChatCompletionResponse(BaseModel):
    id: str
    choices: List[dict]
    created: int = Field(default_factory=lambda: int(time.time()))
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
    frequency_penalty: float | None = shared.args.frequency_penalty
    max_tokens: int | None = 512
    presence_penalty: float | None = shared.args.presence_penalty
    temperature: float | None = shared.args.temperature
    top_p: float | None = shared.args.top_p


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
    instruction_template: str | None = Field(default=None, description="An instruction template defined under textgen/user_data/instruction-templates. Sets the default template for all subsequent API requests.")
    instruction_template_str: str | None = Field(default=None, description="A Jinja2 instruction template string. If set, takes precedence over instruction_template.")


class LoraListResponse(BaseModel):
    lora_names: List[str]


class LoadLorasRequest(BaseModel):
    lora_names: List[str]


class AnthropicRequestParams(BaseModel):
    model: str | None = None
    messages: List[dict] = Field(..., min_length=1)
    max_tokens: int
    system: str | list | None = None
    temperature: float | None = shared.args.temperature
    top_p: float | None = shared.args.top_p
    stop_sequences: list[str] | None = None
    stream: bool = False
    tools: list[dict] | None = None
    tool_choice: dict | None = None
    thinking: dict | None = None
    metadata: dict | None = None


class AnthropicRequest(GenerationOptions, AnthropicRequestParams):
    pass


class ImageGenerationRequest(BaseModel):
    """Image-specific parameters for generation."""
    prompt: str
    negative_prompt: str = ""
    size: str = Field(default="1024x1024", description="'WIDTHxHEIGHT'")
    steps: int = Field(default=9, ge=1)
    cfg_scale: float = Field(default=0.0, ge=0.0)
    image_seed: int = Field(default=-1, description="-1 for random")
    batch_size: int | None = Field(default=None, ge=1, description="Parallel batch size (VRAM heavy)")
    n: int = Field(default=1, ge=1, description="Alias for batch_size (OpenAI compatibility)")
    batch_count: int = Field(default=1, ge=1, description="Sequential batch count")

    # OpenAI compatibility (unused)
    model: str | None = None
    response_format: str = "b64_json"
    user: str | None = None

    @model_validator(mode='after')
    def resolve_batch_size(self):
        if self.batch_size is None:
            self.batch_size = self.n
        return self

    def get_width_height(self) -> tuple[int, int]:
        try:
            parts = self.size.lower().split('x')
            return int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            return 1024, 1024


class ImageGenerationResponse(BaseModel):
    created: int = Field(default_factory=lambda: int(time.time()))
    data: List[dict]


def to_json(obj):
    return json.dumps(obj.__dict__, indent=4)


def to_dict(obj):
    return obj.__dict__
