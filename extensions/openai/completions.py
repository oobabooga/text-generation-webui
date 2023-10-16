import time

import tiktoken
import torch
import torch.nn.functional as F
import yaml
from extensions.openai.defaults import clamp, default, get_default_req_params
from extensions.openai.errors import InvalidRequestError
from extensions.openai.utils import debug_msg, end_line
from modules import shared
from modules.text_generation import decode, encode, generate_reply
from transformers import LogitsProcessor, LogitsProcessorList


# Thanks to @Cypherfox [Cypherfoxy] for the logits code, blame to @matatonic
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
            top_tokens = [decode(tok) for tok in top_indices[0]]
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


def marshal_common_params(body):
    # Request Parameters
    # Try to use openai defaults or map them to something with the same intent

    req_params = get_default_req_params()

    # Common request parameters
    req_params['truncation_length'] = shared.settings['truncation_length']
    req_params['add_bos_token'] = shared.settings.get('add_bos_token', req_params['add_bos_token'])
    req_params['seed'] = shared.settings.get('seed', req_params['seed'])
    req_params['custom_stopping_strings'] = shared.settings['custom_stopping_strings']

    # OpenAI API Parameters
    # model - ignored for now, TODO: When we can reliably load a model or lora from a name only change this
    req_params['requested_model'] = body.get('model', shared.model_name)

    req_params['suffix'] = default(body, 'suffix', req_params['suffix'])
    req_params['temperature'] = clamp(default(body, 'temperature', req_params['temperature']), 0.01, 1.99)  # fixup absolute 0.0/2.0
    req_params['top_p'] = clamp(default(body, 'top_p', req_params['top_p']), 0.01, 1.0)
    n = default(body, 'n', 1)
    if n != 1:
        raise InvalidRequestError(message="Only n = 1 is supported.", param='n')

    if 'stop' in body:  # str or array, max len 4 (ignored)
        if isinstance(body['stop'], str):
            req_params['stopping_strings'] = [body['stop']]  # non-standard parameter
        elif isinstance(body['stop'], list):
            req_params['stopping_strings'] = body['stop']

    # presence_penalty - ignored
    # frequency_penalty - ignored

    # pass through unofficial params
    req_params['repetition_penalty'] = default(body, 'repetition_penalty', req_params['repetition_penalty'])
    req_params['encoder_repetition_penalty'] = default(body, 'encoder_repetition_penalty', req_params['encoder_repetition_penalty'])

    # user - ignored

    logits_processor = []
    logit_bias = body.get('logit_bias', None)
    if logit_bias:  # {str: float, ...}
        # XXX convert tokens from tiktoken based on requested model
        # Ex.: 'logit_bias': {'1129': 100, '11442': 100, '16243': 100}
        try:
            encoder = tiktoken.encoding_for_model(req_params['requested_model'])
            new_logit_bias = {}
            for logit, bias in logit_bias.items():
                for x in encode(encoder.decode([int(logit)]), add_special_tokens=False)[0]:
                    if int(x) in [0, 1, 2, 29871]:  # XXX LLAMA tokens
                        continue
                    new_logit_bias[str(int(x))] = bias
            debug_msg('logit_bias_map', logit_bias, '->', new_logit_bias)
            logit_bias = new_logit_bias
        except KeyError:
            pass  # assume native tokens if we can't find the tokenizer

        logits_processor = [LogitsBiasProcessor(logit_bias)]

    logprobs = None  # coming to chat eventually
    if 'logprobs' in body:
        logprobs = default(body, 'logprobs', 0)  # maybe cap at topk? don't clamp 0-5.
        req_params['logprob_proc'] = LogprobProcessor(logprobs)
        logits_processor.extend([req_params['logprob_proc']])
    else:
        logprobs = None

    if logits_processor:  # requires logits_processor support
        req_params['logits_processor'] = LogitsProcessorList(logits_processor)

    return req_params


def messages_to_prompt(body: dict, req_params: dict, max_tokens):
    # functions
    if body.get('functions', []):  # chat only
        raise InvalidRequestError(message="functions is not supported.", param='functions')
    if body.get('function_call', ''):  # chat only, 'none', 'auto', {'name': 'func'}
        raise InvalidRequestError(message="function_call is not supported.", param='function_call')

    if 'messages' not in body:
        raise InvalidRequestError(message="messages is required", param='messages')

    messages = body['messages']

    role_formats = {
        'user': 'User: {message}\n',
        'assistant': 'Assistant: {message}\n',
        'system': '{message}',
        'context': 'You are a helpful assistant. Answer as concisely as possible.\nUser: I want your assistance.\nAssistant: Sure! What can I do for you?',
        'prompt': 'Assistant:',
    }

    if 'stopping_strings' not in req_params:
        req_params['stopping_strings'] = []

    # Instruct models can be much better
    if shared.settings['instruction_template']:
        try:
            instruct = yaml.safe_load(open(f"instruction-templates/{shared.settings['instruction_template']}.yaml", 'r'))

            template = instruct['turn_template']
            system_message_template = "{message}"
            system_message_default = instruct.get('context', '')  # can be missing
            bot_start = template.find('<|bot|>')  # So far, 100% of instruction templates have this token
            user_message_template = template[:bot_start].replace('<|user-message|>', '{message}').replace('<|user|>', instruct.get('user', ''))
            bot_message_template = template[bot_start:].replace('<|bot-message|>', '{message}').replace('<|bot|>', instruct.get('bot', ''))
            bot_prompt = bot_message_template[:bot_message_template.find('{message}')].rstrip(' ')

            role_formats = {
                'user': user_message_template,
                'assistant': bot_message_template,
                'system': system_message_template,
                'context': system_message_default,
                'prompt': bot_prompt,
            }

            if 'Alpaca' in shared.settings['instruction_template']:
                req_params['stopping_strings'].extend(['\n###'])
            elif instruct['user']:  # WizardLM and some others have no user prompt.
                req_params['stopping_strings'].extend(['\n' + instruct['user'], instruct['user']])

            debug_msg(f"Loaded instruction role format: {shared.settings['instruction_template']}")

        except Exception as e:
            req_params['stopping_strings'].extend(['\nUser:', 'User:'])  # XXX User: prompt here also

            print(f"Exception: When loading instruction-templates/{shared.settings['instruction_template']}.yaml: {repr(e)}")
            print("Warning: Loaded default instruction-following template for model.")

    else:
        req_params['stopping_strings'].extend(['\nUser:', 'User:'])  # XXX User: prompt here also
        print("Warning: Loaded default instruction-following template for model.")

    system_msgs = []
    chat_msgs = []

    # You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible. Knowledge cutoff: {knowledge_cutoff} Current date: {current_date}
    context_msg = role_formats['system'].format(message=role_formats['context']) if role_formats['context'] else ''
    context_msg = end_line(context_msg)

    # Maybe they sent both? This is not documented in the API, but some clients seem to do this.
    if 'prompt' in body:
        context_msg = end_line(role_formats['system'].format(message=body['prompt'])) + context_msg

    for m in messages:
        if 'role' not in m:
            raise InvalidRequestError(message="messages: missing role", param='messages')
        if 'content' not in m:
            raise InvalidRequestError(message="messages: missing content", param='messages')

        role = m['role']
        content = m['content']
        # name = m.get('name', None)
        # function_call = m.get('function_call', None) # user name or function name with output in content
        msg = role_formats[role].format(message=content)
        if role == 'system':
            system_msgs.extend([msg])
        elif role == 'function':
            raise InvalidRequestError(message="role: function is not supported.", param='messages')
        else:
            chat_msgs.extend([msg])

    system_msg = '\n'.join(system_msgs)
    system_msg = end_line(system_msg)

    prompt = system_msg + context_msg + ''.join(chat_msgs) + role_formats['prompt']

    token_count = len(encode(prompt)[0])

    if token_count >= req_params['truncation_length']:
        err_msg = f"This model maximum context length is {req_params['truncation_length']} tokens. However, your messages resulted in over {token_count} tokens."
        raise InvalidRequestError(message=err_msg, param='messages')

    if max_tokens > 0 and token_count + max_tokens > req_params['truncation_length']:
        err_msg = f"This model maximum context length is {req_params['truncation_length']} tokens. However, your messages resulted in over {token_count} tokens and max_tokens is {max_tokens}."
        print(f"Warning: ${err_msg}")
        # raise InvalidRequestError(message=err_msg, params='max_tokens')

    return prompt, token_count


def chat_completions(body: dict, is_legacy: bool = False) -> dict:
    # Chat Completions
    object_type = 'chat.completions'
    created_time = int(time.time())
    cmpl_id = "chatcmpl-%d" % (int(time.time() * 1000000000))
    resp_list = 'data' if is_legacy else 'choices'

    # common params
    req_params = marshal_common_params(body)
    req_params['stream'] = False
    requested_model = req_params.pop('requested_model')
    logprob_proc = req_params.pop('logprob_proc', None)
    req_params['top_k'] = 20  # There is no best_of/top_k param for chat, but it is much improved with a higher top_k.

    # chat default max_tokens is 'inf', but also flexible
    max_tokens = 0
    max_tokens_str = 'length' if is_legacy else 'max_tokens'
    if max_tokens_str in body:
        max_tokens = default(body, max_tokens_str, req_params['truncation_length'])
        req_params['max_new_tokens'] = max_tokens
    else:
        req_params['max_new_tokens'] = req_params['truncation_length']

    # format the prompt from messages
    prompt, token_count = messages_to_prompt(body, req_params, max_tokens)  # updates req_params['stopping_strings']

    # set real max, avoid deeper errors
    if req_params['max_new_tokens'] + token_count >= req_params['truncation_length']:
        req_params['max_new_tokens'] = req_params['truncation_length'] - token_count

    stopping_strings = req_params.pop('stopping_strings', [])

    # generate reply #######################################
    debug_msg({'prompt': prompt, 'req_params': req_params})
    generator = generate_reply(prompt, req_params, stopping_strings=stopping_strings, is_chat=False)

    answer = ''
    for a in generator:
        answer = a

    # strip extra leading space off new generated content
    if answer and answer[0] == ' ':
        answer = answer[1:]

    completion_token_count = len(encode(answer)[0])
    stop_reason = "stop"
    if token_count + completion_token_count >= req_params['truncation_length'] or completion_token_count >= req_params['max_new_tokens']:
        stop_reason = "length"

    resp = {
        "id": cmpl_id,
        "object": object_type,
        "created": created_time,
        "model": shared.model_name,  # TODO: add Lora info?
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

    return resp


# generator
def stream_chat_completions(body: dict, is_legacy: bool = False):

    # Chat Completions
    stream_object_type = 'chat.completions.chunk'
    created_time = int(time.time())
    cmpl_id = "chatcmpl-%d" % (int(time.time() * 1000000000))
    resp_list = 'data' if is_legacy else 'choices'

    # common params
    req_params = marshal_common_params(body)
    req_params['stream'] = True
    requested_model = req_params.pop('requested_model')
    logprob_proc = req_params.pop('logprob_proc', None)
    req_params['top_k'] = 20  # There is no best_of/top_k param for chat, but it is much improved with a higher top_k.

    # chat default max_tokens is 'inf', but also flexible
    max_tokens = 0
    max_tokens_str = 'length' if is_legacy else 'max_tokens'
    if max_tokens_str in body:
        max_tokens = default(body, max_tokens_str, req_params['truncation_length'])
        req_params['max_new_tokens'] = max_tokens
    else:
        req_params['max_new_tokens'] = req_params['truncation_length']

    # format the prompt from messages
    prompt, token_count = messages_to_prompt(body, req_params, max_tokens)  # updates req_params['stopping_strings']

    # set real max, avoid deeper errors
    if req_params['max_new_tokens'] + token_count >= req_params['truncation_length']:
        req_params['max_new_tokens'] = req_params['truncation_length'] - token_count

    def chat_streaming_chunk(content):
        # begin streaming
        chunk = {
            "id": cmpl_id,
            "object": stream_object_type,
            "created": created_time,
            "model": shared.model_name,
            resp_list: [{
                "index": 0,
                "finish_reason": None,
                # So yeah... do both methods? delta and messages.
                "message": {'role': 'assistant', 'content': content},
                "delta": {'role': 'assistant', 'content': content},
            }],
        }

        if logprob_proc:  # not official for chat yet
            top_logprobs = convert_logprobs_to_tiktoken(model=requested_model, logprobs=logprob_proc.token_alternatives)
            chunk[resp_list][0]["logprobs"] = {'top_logprobs': [top_logprobs]}
        # else:
        #    chunk[resp_list][0]["logprobs"] = None
        return chunk

    yield chat_streaming_chunk('')

    # generate reply #######################################
    debug_msg({'prompt': prompt, 'req_params': req_params})

    stopping_strings = req_params.pop('stopping_strings', [])

    generator = generate_reply(prompt, req_params, stopping_strings=stopping_strings, is_chat=False)

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

        # strip extra leading space off new generated content
        if len_seen == 0 and new_content[0] == ' ':
            new_content = new_content[1:]

        chunk = chat_streaming_chunk(new_content)

        yield chunk

    # to get the correct token_count, strip leading space if present
    if answer and answer[0] == ' ':
        answer = answer[1:]

    completion_token_count = len(encode(answer)[0])
    stop_reason = "stop"
    if token_count + completion_token_count >= req_params['truncation_length'] or completion_token_count >= req_params['max_new_tokens']:
        stop_reason = "length"

    chunk = chat_streaming_chunk('')
    chunk[resp_list][0]['finish_reason'] = stop_reason
    chunk['usage'] = {
        "prompt_tokens": token_count,
        "completion_tokens": completion_token_count,
        "total_tokens": token_count + completion_token_count
    }

    yield chunk


def completions(body: dict, is_legacy: bool = False):
    # Legacy
    # Text Completions
    object_type = 'text_completion'
    created_time = int(time.time())
    cmpl_id = "conv-%d" % (int(time.time() * 1000000000))
    resp_list = 'data' if is_legacy else 'choices'

    # ... encoded as a string, array of strings, array of tokens, or array of token arrays.
    prompt_str = 'context' if is_legacy else 'prompt'
    if prompt_str not in body:
        raise InvalidRequestError("Missing required input", param=prompt_str)

    prompt_arg = body[prompt_str]
    if isinstance(prompt_arg, str) or (isinstance(prompt_arg, list) and isinstance(prompt_arg[0], int)):
        prompt_arg = [prompt_arg]

    # common params
    req_params = marshal_common_params(body)
    req_params['stream'] = False
    max_tokens_str = 'length' if is_legacy else 'max_tokens'
    max_tokens = default(body, max_tokens_str, req_params['max_new_tokens'])
    req_params['max_new_tokens'] = max_tokens
    requested_model = req_params.pop('requested_model')
    logprob_proc = req_params.pop('logprob_proc', None)
    stopping_strings = req_params.pop('stopping_strings', [])
    # req_params['suffix'] = default(body, 'suffix', req_params['suffix'])
    req_params['echo'] = default(body, 'echo', req_params['echo'])
    req_params['top_k'] = default(body, 'best_of', req_params['top_k'])

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

        token_count = len(encode(prompt)[0])
        total_prompt_token_count += token_count

        if token_count + max_tokens > req_params['truncation_length']:
            err_msg = f"The token count of your prompt ({token_count}) plus max_tokens ({max_tokens}) cannot exceed the model's context length ({req_params['truncation_length']})."
            # print(f"Warning: ${err_msg}")
            raise InvalidRequestError(message=err_msg, param=max_tokens_str)

        # generate reply #######################################
        debug_msg({'prompt': prompt, 'req_params': req_params})
        generator = generate_reply(prompt, req_params, stopping_strings=stopping_strings, is_chat=False)
        answer = ''

        for a in generator:
            answer = a

        # strip extra leading space off new generated content
        if answer and answer[0] == ' ':
            answer = answer[1:]

        completion_token_count = len(encode(answer)[0])
        total_completion_token_count += completion_token_count
        stop_reason = "stop"
        if token_count + completion_token_count >= req_params['truncation_length'] or completion_token_count >= max_tokens:
            stop_reason = "length"

        respi = {
            "index": idx,
            "finish_reason": stop_reason,
            "text": answer,
            "logprobs": {'top_logprobs': [logprob_proc.token_alternatives]} if logprob_proc else None,
        }

        resp_list_data.extend([respi])

    resp = {
        "id": cmpl_id,
        "object": object_type,
        "created": created_time,
        "model": shared.model_name,  # TODO: add Lora info?
        resp_list: resp_list_data,
        "usage": {
            "prompt_tokens": total_prompt_token_count,
            "completion_tokens": total_completion_token_count,
            "total_tokens": total_prompt_token_count + total_completion_token_count
        }
    }

    return resp


# generator
def stream_completions(body: dict, is_legacy: bool = False):
    # Legacy
    # Text Completions
    # object_type = 'text_completion'
    stream_object_type = 'text_completion.chunk'
    created_time = int(time.time())
    cmpl_id = "conv-%d" % (int(time.time() * 1000000000))
    resp_list = 'data' if is_legacy else 'choices'

    # ... encoded as a string, array of strings, array of tokens, or array of token arrays.
    prompt_str = 'context' if is_legacy else 'prompt'
    if prompt_str not in body:
        raise InvalidRequestError("Missing required input", param=prompt_str)

    prompt = body[prompt_str]
    req_params = marshal_common_params(body)
    requested_model = req_params.pop('requested_model')
    if isinstance(prompt, list):
        if prompt and isinstance(prompt[0], int):
            try:
                encoder = tiktoken.encoding_for_model(requested_model)
                prompt = encoder.decode(prompt)
            except KeyError:
                prompt = decode(prompt)[0]
        else:
            raise InvalidRequestError(message="API Batched generation not yet supported.", param=prompt_str)

    # common params
    req_params['stream'] = True
    max_tokens_str = 'length' if is_legacy else 'max_tokens'
    max_tokens = default(body, max_tokens_str, req_params['max_new_tokens'])
    req_params['max_new_tokens'] = max_tokens
    logprob_proc = req_params.pop('logprob_proc', None)
    stopping_strings = req_params.pop('stopping_strings', [])
    # req_params['suffix'] = default(body, 'suffix', req_params['suffix'])
    req_params['echo'] = default(body, 'echo', req_params['echo'])
    req_params['top_k'] = default(body, 'best_of', req_params['top_k'])

    token_count = len(encode(prompt)[0])

    if token_count + max_tokens > req_params['truncation_length']:
        err_msg = f"The token count of your prompt ({token_count}) plus max_tokens ({max_tokens}) cannot exceed the model's context length ({req_params['truncation_length']})."
        # print(f"Warning: ${err_msg}")
        raise InvalidRequestError(message=err_msg, param=max_tokens_str)

    def text_streaming_chunk(content):
        # begin streaming
        chunk = {
            "id": cmpl_id,
            "object": stream_object_type,
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

    yield text_streaming_chunk('')

    # generate reply #######################################
    debug_msg({'prompt': prompt, 'req_params': req_params})
    generator = generate_reply(prompt, req_params, stopping_strings=stopping_strings, is_chat=False)

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

        # strip extra leading space off new generated content
        if len_seen == 0 and new_content[0] == ' ':
            new_content = new_content[1:]

        chunk = text_streaming_chunk(new_content)

        yield chunk

    # to get the correct count, we strip the leading space if present
    if answer and answer[0] == ' ':
        answer = answer[1:]

    completion_token_count = len(encode(answer)[0])
    stop_reason = "stop"
    if token_count + completion_token_count >= req_params['truncation_length'] or completion_token_count >= max_tokens:
        stop_reason = "length"

    chunk = text_streaming_chunk('')
    chunk[resp_list][0]["finish_reason"] = stop_reason
    chunk["usage"] = {
        "prompt_tokens": token_count,
        "completion_tokens": completion_token_count,
        "total_tokens": token_count + completion_token_count
    }

    yield chunk
