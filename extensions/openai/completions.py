import base64
import copy
import re
import time
from collections import deque
from io import BytesIO

import requests
import tiktoken
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import LogitsProcessor, LogitsProcessorList

from extensions.openai.errors import InvalidRequestError
from extensions.openai.utils import debug_msg
from modules import shared
from modules.chat import (
    generate_chat_prompt,
    generate_chat_reply,
    load_character_memoized,
    load_instruction_template_memoized
)
from modules.presets import load_preset_memoized
from modules.text_generation import (
    decode,
    encode,
    generate_reply,
    get_reply_from_output_ids
)


class LogitsBiasProcessor(LogitsProcessor):
    def __init__(self, logit_bias={}):
        self.logit_bias = logit_bias
        if self.logit_bias:
            self.keys = list([int(key) for key in self.logit_bias.keys()])
            values = [self.logit_bias[str(key)] for key in self.keys]
            self.values = torch.tensor(values, dtype=torch.float, device=shared.model.device)
            debug_msg(f"{self})")

    def __call__(self, input_ids: torch.LongTensor, logits: torch.FloatTensor) -> torch.FloatTensor:
        if self.logit_bias:
            debug_msg(logits[0, self.keys], " + ", self.values)
            logits[0, self.keys] += self.values
            debug_msg(" --> ", logits[0, self.keys])
            debug_msg(" max/min ", float(torch.max(logits[0])), float(torch.min(logits[0])))

        return logits

    def __repr__(self):
        return f"<{self.__class__.__name__}(logit_bias={self.logit_bias})>"


class LogprobProcessor(LogitsProcessor):
    def __init__(self, logprobs=None):
        self.logprobs = logprobs
        self.token_alternatives = {}

    def __call__(self, input_ids: torch.LongTensor, logits: torch.FloatTensor) -> torch.FloatTensor:
        if self.logprobs is not None:  # 0-5
            log_e_probabilities = F.log_softmax(logits, dim=1)
            top_values, top_indices = torch.topk(log_e_probabilities, k=self.logprobs + 1)
            top_tokens = [get_reply_from_output_ids([tok]) for tok in top_indices[0]]
            top_probs = [float(x) for x in top_values[0]]
            self.token_alternatives = dict(zip(top_tokens, top_probs))
            debug_msg(repr(self))

        return logits

    def __repr__(self):
        return f"<{self.__class__.__name__}(logprobs={self.logprobs}, token_alternatives={self.token_alternatives})>"


def convert_logprobs_to_tiktoken(model, logprobs):
    # more problems than it's worth.
    # try:
    #     encoder = tiktoken.encoding_for_model(model)
    #     # just pick the first one if it encodes to multiple tokens... 99.9% not required and maybe worse overall.
    #     return dict([(encoder.decode([encoder.encode(token)[0]]), prob) for token, prob in logprobs.items()])
    # except KeyError:
    #     # assume native tokens if we can't find the tokenizer
    #     return logprobs

    return logprobs


def process_parameters(body, is_legacy=False):
    generate_params = body
    max_tokens_str = 'length' if is_legacy else 'max_tokens'
    generate_params['max_new_tokens'] = body.pop(max_tokens_str)
    if generate_params['truncation_length'] == 0:
        generate_params['truncation_length'] = shared.settings['truncation_length']

    if generate_params['temperature'] == 0:
        generate_params['do_sample'] = False
        generate_params['top_k'] = 1

    if body['preset'] is not None:
        preset = load_preset_memoized(body['preset'])
        generate_params.update(preset)

    generate_params['custom_stopping_strings'] = []
    if 'stop' in body:  # str or array, max len 4 (ignored)
        if isinstance(body['stop'], str):
            generate_params['custom_stopping_strings'] = [body['stop']]
        elif isinstance(body['stop'], list):
            generate_params['custom_stopping_strings'] = body['stop']

    logits_processor = []
    logit_bias = body.get('logit_bias', None)
    if logit_bias:  # {str: float, ...}
        logits_processor = [LogitsBiasProcessor(logit_bias)]

    logprobs = None  # coming to chat eventually
    if 'logprobs' in body:
        logprobs = body.get('logprobs', 0)  # maybe cap at topk? don't clamp 0-5.
        generate_params['logprob_proc'] = LogprobProcessor(logprobs)
        logits_processor.extend([generate_params['logprob_proc']])
    else:
        logprobs = None

    if logits_processor:  # requires logits_processor support
        generate_params['logits_processor'] = LogitsProcessorList(logits_processor)

    return generate_params


def convert_history(history):
    '''
    Chat histories in this program are in the format [message, reply].
    This function converts OpenAI histories to that format.
    '''
    chat_dialogue = []
    current_message = ""
    current_reply = ""
    user_input = ""
    user_input_last = True
    system_message = ""

    # Multimodal: convert OpenAI format to multimodal extension format
    if any('content' in entry and isinstance(entry['content'], list) for entry in history):
        new_history = []
        for entry in history:
            if isinstance(entry['content'], list):
                image_url = None
                content = None
                for item in entry['content']:
                    if not isinstance(item, dict):
                        continue

                    if item['type'] == 'image_url' and isinstance(item['image_url'], dict):
                        image_url = item['image_url']['url']
                    elif item['type'] == 'text' and isinstance(item['text'], str):
                        content = item['text']

                if image_url and content:
                    new_history.append({"image_url": image_url, "role": "user"})
                    new_history.append({"content": content, "role": "user"})
            else:
                new_history.append(entry)

        history = new_history

    for entry in history:
        if "image_url" in entry:
            image_url = entry['image_url']
            if "base64" in image_url:
                image_url = re.sub('^data:image/.+;base64,', '', image_url)
                img = Image.open(BytesIO(base64.b64decode(image_url)))
            else:
                try:
                    my_res = requests.get(image_url)
                    img = Image.open(BytesIO(my_res.content))
                except Exception:
                    raise 'Image cannot be loaded from the URL!'

            buffered = BytesIO()
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            content = f'<img src="data:image/jpeg;base64,{img_str}">'
        else:
            content = entry["content"]

        role = entry["role"]

        if role == "user":
            user_input = content
            user_input_last = True
            if current_message:
                chat_dialogue.append([current_message, ''])
                current_message = ""

            current_message = content
        elif role == "assistant":
            current_reply = content
            user_input_last = False
            if current_message:
                chat_dialogue.append([current_message, current_reply])
                current_message = ""
                current_reply = ""
            else:
                chat_dialogue.append(['', current_reply])
        elif role == "system":
            system_message = content

    if not user_input_last:
        user_input = ""

    return user_input, system_message, {'internal': chat_dialogue, 'visible': copy.deepcopy(chat_dialogue)}


def chat_completions_common(body: dict, is_legacy: bool = False, stream=False, prompt_only=False) -> dict:
    if body.get('functions', []):
        raise InvalidRequestError(message="functions is not supported.", param='functions')

    if body.get('function_call', ''):
        raise InvalidRequestError(message="function_call is not supported.", param='function_call')

    if 'messages' not in body:
        raise InvalidRequestError(message="messages is required", param='messages')

    messages = body['messages']
    for m in messages:
        if 'role' not in m:
            raise InvalidRequestError(message="messages: missing role", param='messages')
        elif m['role'] == 'function':
            raise InvalidRequestError(message="role: function is not supported.", param='messages')

        if 'content' not in m and "image_url" not in m:
            raise InvalidRequestError(message="messages: missing content", param='messages')

    # Chat Completions
    object_type = 'chat.completions' if not stream else 'chat.completions.chunk'
    created_time = int(time.time())
    cmpl_id = "chatcmpl-%d" % (int(time.time() * 1000000000))
    resp_list = 'data' if is_legacy else 'choices'

    # generation parameters
    generate_params = process_parameters(body, is_legacy=is_legacy)
    continue_ = body['continue_']

    # Instruction template
    if body['instruction_template_str']:
        instruction_template_str = body['instruction_template_str']
    elif body['instruction_template']:
        instruction_template = body['instruction_template']
        instruction_template = "Alpaca" if instruction_template == "None" else instruction_template
        instruction_template_str = load_instruction_template_memoized(instruction_template)
    else:
        instruction_template_str = shared.settings['instruction_template_str']

    chat_template_str = body['chat_template_str'] or shared.default_settings['chat_template_str']
    chat_instruct_command = body['chat_instruct_command'] or shared.default_settings['chat-instruct_command']

    # Chat character
    character = body['character'] or shared.default_settings['character']
    character = "Assistant" if character == "None" else character
    name1 = body['user_name'] or shared.default_settings['name1']
    name1, name2, _, greeting, context = load_character_memoized(character, name1, '')
    name2 = body['bot_name'] or name2
    context = body['context'] or context
    greeting = body['greeting'] or greeting
    user_bio = body['user_bio'] or ''

    # History
    user_input, custom_system_message, history = convert_history(messages)

    generate_params.update({
        'mode': body['mode'],
        'name1': name1,
        'name2': name2,
        'context': context,
        'greeting': greeting,
        'user_bio': user_bio,
        'instruction_template_str': instruction_template_str,
        'custom_system_message': custom_system_message,
        'chat_template_str': chat_template_str,
        'chat-instruct_command': chat_instruct_command,
        'history': history,
        'stream': stream
    })

    max_tokens = generate_params['max_new_tokens']
    if max_tokens in [None, 0]:
        generate_params['max_new_tokens'] = 512
        generate_params['auto_max_new_tokens'] = True

    requested_model = generate_params.pop('model')
    logprob_proc = generate_params.pop('logprob_proc', None)

    def chat_streaming_chunk(content):
        # begin streaming
        chunk = {
            "id": cmpl_id,
            "object": object_type,
            "created": created_time,
            "model": shared.model_name,
            resp_list: [{
                "index": 0,
                "finish_reason": None,
                "delta": {'role': 'assistant', 'content': content},
            }],
        }

        if logprob_proc:  # not official for chat yet
            top_logprobs = convert_logprobs_to_tiktoken(model=requested_model, logprobs=logprob_proc.token_alternatives)
            chunk[resp_list][0]["logprobs"] = {'top_logprobs': [top_logprobs]}
        # else:
        #    chunk[resp_list][0]["logprobs"] = None
        return chunk

    # generate reply #######################################
    prompt = generate_chat_prompt(user_input, generate_params, _continue=continue_)
    if prompt_only:
        yield {'prompt': prompt}
        return

    debug_msg({'prompt': prompt, 'generate_params': generate_params})

    if stream:
        yield chat_streaming_chunk('')

    generator = generate_chat_reply(
        user_input, generate_params, regenerate=False, _continue=continue_, loading_message=False)

    answer = ''
    seen_content = ''

    for a in generator:
        answer = a['internal'][-1][1]
        if stream:
            len_seen = len(seen_content)
            new_content = answer[len_seen:]

            if not new_content or chr(0xfffd) in new_content:  # partial unicode character, don't send it yet.
                continue

            seen_content = answer
            chunk = chat_streaming_chunk(new_content)
            yield chunk

    token_count = len(encode(prompt)[0])
    completion_token_count = len(encode(answer)[0])
    stop_reason = "stop"
    if token_count + completion_token_count >= generate_params['truncation_length'] or completion_token_count >= generate_params['max_new_tokens']:
        stop_reason = "length"

    if stream:
        chunk = chat_streaming_chunk('')
        chunk[resp_list][0]['finish_reason'] = stop_reason
        chunk['usage'] = {
            "prompt_tokens": token_count,
            "completion_tokens": completion_token_count,
            "total_tokens": token_count + completion_token_count
        }

        yield chunk
    else:
        resp = {
            "id": cmpl_id,
            "object": object_type,
            "created": created_time,
            "model": shared.model_name,
            resp_list: [{
                "index": 0,
                "finish_reason": stop_reason,
                "message": {"role": "assistant", "content": answer}
            }],
            "usage": {
                "prompt_tokens": token_count,
                "completion_tokens": completion_token_count,
                "total_tokens": token_count + completion_token_count
            }
        }
        if logprob_proc:  # not official for chat yet
            top_logprobs = convert_logprobs_to_tiktoken(model=requested_model, logprobs=logprob_proc.token_alternatives)
            resp[resp_list][0]["logprobs"] = {'top_logprobs': [top_logprobs]}
        # else:
        #     resp[resp_list][0]["logprobs"] = None

        yield resp


def completions_common(body: dict, is_legacy: bool = False, stream=False):
    object_type = 'text_completion.chunk' if stream else 'text_completion'
    created_time = int(time.time())
    cmpl_id = "conv-%d" % (int(time.time() * 1000000000))
    resp_list = 'data' if is_legacy else 'choices'

    prompt_str = 'context' if is_legacy else 'prompt'

    # ... encoded as a string, array of strings, array of tokens, or array of token arrays.
    if prompt_str not in body:
        raise InvalidRequestError("Missing required input", param=prompt_str)

    # common params
    generate_params = process_parameters(body, is_legacy=is_legacy)
    max_tokens = generate_params['max_new_tokens']
    generate_params['stream'] = stream
    requested_model = generate_params.pop('model')
    logprob_proc = generate_params.pop('logprob_proc', None)
    suffix = body['suffix'] if body['suffix'] else ''
    echo = body['echo']

    if not stream:
        prompt_arg = body[prompt_str]
        if isinstance(prompt_arg, str) or (isinstance(prompt_arg, list) and isinstance(prompt_arg[0], int)):
            prompt_arg = [prompt_arg]

        resp_list_data = []
        total_completion_token_count = 0
        total_prompt_token_count = 0

        for idx, prompt in enumerate(prompt_arg, start=0):
            if isinstance(prompt[0], int):
                # token lists
                if requested_model == shared.model_name:
                    prompt = decode(prompt)[0]
                else:
                    try:
                        encoder = tiktoken.encoding_for_model(requested_model)
                        prompt = encoder.decode(prompt)
                    except KeyError:
                        prompt = decode(prompt)[0]

            prefix = prompt if echo else ''

            # generate reply #######################################
            debug_msg({'prompt': prompt, 'generate_params': generate_params})
            generator = generate_reply(prompt, generate_params, is_chat=False)
            answer = ''

            for a in generator:
                answer = a

            token_count = len(encode(prompt)[0])
            total_prompt_token_count += token_count
            completion_token_count = len(encode(answer)[0])
            total_completion_token_count += completion_token_count
            stop_reason = "stop"
            if token_count + completion_token_count >= generate_params['truncation_length'] or completion_token_count >= max_tokens:
                stop_reason = "length"

            respi = {
                "index": idx,
                "finish_reason": stop_reason,
                "text": prefix + answer + suffix,
                "logprobs": {'top_logprobs': [logprob_proc.token_alternatives]} if logprob_proc else None,
            }

            resp_list_data.extend([respi])

        resp = {
            "id": cmpl_id,
            "object": object_type,
            "created": created_time,
            "model": shared.model_name,
            resp_list: resp_list_data,
            "usage": {
                "prompt_tokens": total_prompt_token_count,
                "completion_tokens": total_completion_token_count,
                "total_tokens": total_prompt_token_count + total_completion_token_count
            }
        }

        yield resp
    else:
        prompt = body[prompt_str]
        if isinstance(prompt, list):
            if prompt and isinstance(prompt[0], int):
                try:
                    encoder = tiktoken.encoding_for_model(requested_model)
                    prompt = encoder.decode(prompt)
                except KeyError:
                    prompt = decode(prompt)[0]
            else:
                raise InvalidRequestError(message="API Batched generation not yet supported.", param=prompt_str)

        prefix = prompt if echo else ''
        token_count = len(encode(prompt)[0])

        def text_streaming_chunk(content):
            # begin streaming
            chunk = {
                "id": cmpl_id,
                "object": object_type,
                "created": created_time,
                "model": shared.model_name,
                resp_list: [{
                    "index": 0,
                    "finish_reason": None,
                    "text": content,
                    "logprobs": {'top_logprobs': [logprob_proc.token_alternatives]} if logprob_proc else None,
                }],
            }

            return chunk

        yield text_streaming_chunk(prefix)

        # generate reply #######################################
        debug_msg({'prompt': prompt, 'generate_params': generate_params})
        generator = generate_reply(prompt, generate_params, is_chat=False)

        answer = ''
        seen_content = ''
        completion_token_count = 0

        for a in generator:
            answer = a

            len_seen = len(seen_content)
            new_content = answer[len_seen:]

            if not new_content or chr(0xfffd) in new_content:  # partial unicode character, don't send it yet.
                continue

            seen_content = answer
            chunk = text_streaming_chunk(new_content)
            yield chunk

        completion_token_count = len(encode(answer)[0])
        stop_reason = "stop"
        if token_count + completion_token_count >= generate_params['truncation_length'] or completion_token_count >= max_tokens:
            stop_reason = "length"

        chunk = text_streaming_chunk(suffix)
        chunk[resp_list][0]["finish_reason"] = stop_reason
        chunk["usage"] = {
            "prompt_tokens": token_count,
            "completion_tokens": completion_token_count,
            "total_tokens": token_count + completion_token_count
        }

        yield chunk


def chat_completions(body: dict, is_legacy: bool = False) -> dict:
    generator = chat_completions_common(body, is_legacy, stream=False)
    return deque(generator, maxlen=1).pop()


def stream_chat_completions(body: dict, is_legacy: bool = False):
    for resp in chat_completions_common(body, is_legacy, stream=True):
        yield resp


def completions(body: dict, is_legacy: bool = False) -> dict:
    generator = completions_common(body, is_legacy, stream=False)
    return deque(generator, maxlen=1).pop()


def stream_completions(body: dict, is_legacy: bool = False):
    for resp in completions_common(body, is_legacy, stream=True):
        yield resp
