import base64
import json
import os
import time
import requests
import yaml
import numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from modules.utils import get_available_models
from modules.models import load_model, unload_model
from modules.models_settings import (get_model_settings_from_yamls,
                                     update_model_parameters)

from modules import shared
from modules.text_generation import encode, generate_reply

params = {
    'port': int(os.environ.get('OPENEDAI_PORT')) if 'OPENEDAI_PORT' in os.environ else 5001,
}

debug = True if 'OPENEDAI_DEBUG' in os.environ else False

# Slightly different defaults for OpenAI's API
# Data type is important, Ex. use 0.0 for a float 0
default_req_params = {
    'max_new_tokens': 200,
    'temperature': 1.0,
    'top_p': 1.0,
    'top_k': 1,
    'repetition_penalty': 1.18,
    'repetition_penalty_range': 0,
    'encoder_repetition_penalty': 1.0,
    'suffix': None,
    'stream': False,
    'echo': False,
    'seed': -1,
    # 'n' : default(body, 'n', 1),  # 'n' doesn't have a direct map
    'truncation_length': 2048,
    'add_bos_token': True,
    'do_sample': True,
    'typical_p': 1.0,
    'epsilon_cutoff': 0.0,  # In units of 1e-4
    'eta_cutoff': 0.0,  # In units of 1e-4
    'tfs': 1.0,
    'top_a': 0.0,
    'min_length': 0,
    'no_repeat_ngram_size': 0,
    'num_beams': 1,
    'penalty_alpha': 0.0,
    'length_penalty': 1.0,
    'early_stopping': False,
    'mirostat_mode': 0,
    'mirostat_tau': 5.0,
    'mirostat_eta': 0.1,
    'ban_eos_token': False,
    'skip_special_tokens': True,
    'custom_stopping_strings': '',
}

# Optional, install the module and download the model to enable
# v1/embeddings
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    pass

st_model = os.environ["OPENEDAI_EMBEDDING_MODEL"] if "OPENEDAI_EMBEDDING_MODEL" in os.environ else "all-mpnet-base-v2"
embedding_model = None

# little helper to get defaults if arg is present but None and should be the same type as default.
def default(dic, key, default):
    val = dic.get(key, default)
    if type(val) != type(default):
        # maybe it's just something like 1 instead of 1.0
        try:
            v = type(default)(val)
            if type(val)(v) == val:  # if it's the same value passed in, it's ok.
                return v
        except:
            pass

        val = default
    return val


def clamp(value, minvalue, maxvalue):
    return max(minvalue, min(value, maxvalue))


def float_list_to_base64(float_list):
    # Convert the list to a float32 array that the OpenAPI client expects
    float_array = np.array(float_list, dtype="float32")

    # Get raw bytes
    bytes_array = float_array.tobytes()

    # Encode bytes into base64
    encoded_bytes = base64.b64encode(bytes_array)

    # Turn raw base64 encoded bytes into ASCII
    ascii_string = encoded_bytes.decode('ascii')
    return ascii_string


class Handler(BaseHTTPRequestHandler):
    def send_access_control_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Credentials", "true")
        self.send_header(
            "Access-Control-Allow-Methods",
            "GET,HEAD,OPTIONS,POST,PUT"
        )
        self.send_header(
            "Access-Control-Allow-Headers",
            "Origin, Accept, X-Requested-With, Content-Type, "
            "Access-Control-Request-Method, Access-Control-Request-Headers, "
            "Authorization"
        )

    def openai_error(self, message, code = 500, error_type = 'APIError', param = '', internal_message = ''):
        self.send_response(code)
        self.send_access_control_headers()
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        error_resp = {
            'error': {
                'message': message,
                'code': code,
                'type': error_type,
                'param': param,
            }
        }
        if internal_message:
            error_resp['internal_message'] = internal_message

        response = json.dumps(error_resp)
        self.wfile.write(response.encode('utf-8'))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_access_control_headers()
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write("OK".encode('utf-8'))

    def do_GET(self):
        if self.path.startswith('/v1/engines') or self.path.startswith('/v1/models'):
            current_model_list = [ shared.model_name ] # The real chat/completions model, maybe "None"
            embeddings_model_list = [ st_model ] if embedding_model else [] # The real sentence transformer embeddings model
            pseudo_model_list = [ # these are expected by so much, so include some here as a dummy
                'gpt-3.5-turbo', # /v1/chat/completions
                'text-curie-001', # /v1/completions, 2k context
                'text-davinci-002' # /v1/embeddings text-embedding-ada-002:1536, text-davinci-002:768
            ]

            is_legacy = 'engines' in self.path
            is_list = self.path in ['/v1/engines', '/v1/models']

            resp = ''

            if is_legacy and not is_list: # load model
                model_name = self.path[self.path.find('/v1/engines/') + len('/v1/engines/'):]

                resp = {
                    "id": model_name,
                    "object": "engine",
                    "owner": "self",
                    "ready": True,
                }
                if model_name not in pseudo_model_list + embeddings_model_list + current_model_list: # Real model only
                    # No args. Maybe it works anyways!
                    # TODO: hack some heuristics into args for better results

                    shared.model_name = model_name
                    unload_model()

                    model_settings = get_model_settings_from_yamls(shared.model_name)
                    shared.settings.update(model_settings)
                    update_model_parameters(model_settings, initial=True)

                    if shared.settings['mode'] != 'instruct':
                        shared.settings['instruction_template'] = None

                    shared.model, shared.tokenizer = load_model(shared.model_name)

                    if not shared.model: # load failed.
                        shared.model_name = "None"
                        resp['id'] = "None"
                        resp['ready'] = False

            elif is_list:
                # TODO: Lora's?
                available_model_list = get_available_models()
                all_model_list = current_model_list + embeddings_model_list + pseudo_model_list + available_model_list

                models = {}

                if is_legacy:
                    models = [{ "id": id, "object": "engine", "owner": "user", "ready": True } for id in all_model_list ]
                    if not shared.model:
                        models[0]['ready'] = False
                else:
                    models = [{ "id": id, "object": "model", "owned_by": "user", "permission": [] } for id in all_model_list ]

                resp = {
                    "object": "list",
                    "data": models,
                }

            else:
                the_model_name = self.path[len('/v1/models/'):]
                resp = {
                    "id": the_model_name,
                    "object": "model",
                    "owned_by": "user",
                    "permission": []
                }

            self.send_response(200)
            self.send_access_control_headers()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = json.dumps(resp)
            self.wfile.write(response.encode('utf-8'))

        elif '/billing/usage' in self.path:
            # Ex. /v1/dashboard/billing/usage?start_date=2023-05-01&end_date=2023-05-31
            self.send_response(200)
            self.send_access_control_headers()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()

            response = json.dumps({
                "total_usage": 0,
            })
            self.wfile.write(response.encode('utf-8'))

        else:
            self.send_error(404)

    def do_POST(self):
        if debug:
            print(self.headers)  # did you know... python-openai sends your linux kernel & python version?
        content_length = int(self.headers['Content-Length'])
        body = json.loads(self.rfile.read(content_length).decode('utf-8'))

        if debug:
            print(body)

        if '/completions' in self.path or '/generate' in self.path:

            if not shared.model:
                self.openai_error("No model loaded.")
                return

            is_legacy = '/generate' in self.path
            is_chat_request = 'chat' in self.path
            resp_list = 'data' if is_legacy else 'choices'

            # XXX model is ignored for now
            # model = body.get('model', shared.model_name) # ignored, use existing for now
            model = shared.model_name
            created_time = int(time.time())

            cmpl_id = "chatcmpl-%d" % (created_time) if is_chat_request else "conv-%d" % (created_time)

            # Request Parameters
            # Try to use openai defaults or map them to something with the same intent
            req_params = default_req_params.copy()
            stopping_strings = []

            if 'stop' in body:
                if isinstance(body['stop'], str):
                    stopping_strings.extend([body['stop']])
                elif isinstance(body['stop'], list):
                    stopping_strings.extend(body['stop'])

            truncation_length = default(shared.settings, 'truncation_length', 2048)
            truncation_length = clamp(default(body, 'truncation_length', truncation_length), 1, truncation_length)

            default_max_tokens = truncation_length if is_chat_request else 16  # completions default, chat default is 'inf' so we need to cap it.

            max_tokens_str = 'length' if is_legacy else 'max_tokens'
            max_tokens = default(body, max_tokens_str, default(shared.settings, 'max_new_tokens', default_max_tokens))
            # if the user assumes OpenAI, the max_tokens is way too large - try to ignore it unless it's small enough

            req_params['max_new_tokens'] = max_tokens
            req_params['truncation_length'] = truncation_length
            req_params['temperature'] = clamp(default(body, 'temperature', default_req_params['temperature']), 0.001, 1.999) # fixup absolute 0.0
            req_params['top_p'] = clamp(default(body, 'top_p', default_req_params['top_p']), 0.001, 1.0)
            req_params['top_k'] = default(body, 'best_of', default_req_params['top_k'])
            req_params['suffix'] = default(body, 'suffix', default_req_params['suffix'])
            req_params['stream'] = default(body, 'stream', default_req_params['stream'])
            req_params['echo'] = default(body, 'echo', default_req_params['echo'])
            req_params['seed'] = shared.settings.get('seed', default_req_params['seed'])
            req_params['add_bos_token'] = shared.settings.get('add_bos_token', default_req_params['add_bos_token'])

            is_streaming = req_params['stream']

            self.send_response(200)
            self.send_access_control_headers()
            if is_streaming:
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                # self.send_header('Connection', 'keep-alive')
            else:
                self.send_header('Content-Type', 'application/json')
            self.end_headers()

            token_count = 0
            completion_token_count = 0
            prompt = ''
            stream_object_type = ''
            object_type = ''

            if is_chat_request:
                # Chat Completions
                stream_object_type = 'chat.completions.chunk'
                object_type = 'chat.completions'

                messages = body['messages']

                role_formats = {
                    'user': 'user: {message}\n',
                    'assistant': 'assistant: {message}\n',
                    'system': '{message}',
                    'context': 'You are a helpful assistant. Answer as concisely as possible.',
                    'prompt': 'assistant:',
                }

                # Instruct models can be much better
                if shared.settings['instruction_template']:
                    try:
                        instruct = yaml.safe_load(open(f"characters/instruction-following/{shared.settings['instruction_template']}.yaml", 'r'))

                        template = instruct['turn_template']
                        system_message_template = "{message}"
                        system_message_default = instruct['context']
                        bot_start = template.find('<|bot|>') # So far, 100% of instruction templates have this token
                        user_message_template = template[:bot_start].replace('<|user-message|>', '{message}').replace('<|user|>', instruct['user'])
                        bot_message_template = template[bot_start:].replace('<|bot-message|>', '{message}').replace('<|bot|>', instruct['bot'])
                        bot_prompt = bot_message_template[:bot_message_template.find('{message}')].rstrip(' ')
                
                        role_formats = {
                            'user': user_message_template,
                            'assistant': bot_message_template,
                            'system': system_message_template,
                            'context': system_message_default,
                            'prompt': bot_prompt,
                        }

                        if 'Alpaca' in shared.settings['instruction_template']:
                            stopping_strings.extend(['\n###'])
                        elif instruct['user']: # WizardLM and some others have no user prompt.
                            stopping_strings.extend(['\n' + instruct['user'], instruct['user']])

                        if debug:
                            print(f"Loaded instruction role format: {shared.settings['instruction_template']}")

                    except Exception as e:
                        stopping_strings.extend(['\nuser:'])

                        print(f"Exception: When loading characters/instruction-following/{shared.settings['instruction_template']}.yaml: {repr(e)}")
                        print("Warning: Loaded default instruction-following template for model.")

                else:
                    stopping_strings.extend(['\nuser:'])
                    print("Warning: Loaded default instruction-following template for model.")

                system_msgs = []
                chat_msgs = []

                # You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible. Knowledge cutoff: {knowledge_cutoff} Current date: {current_date}
                context_msg = role_formats['system'].format(message=role_formats['context']) if role_formats['context'] else ''
                if context_msg:
                    system_msgs.extend([context_msg])

                # Maybe they sent both? This is not documented in the API, but some clients seem to do this.
                if 'prompt' in body:
                    prompt_msg = role_formats['system'].format(message=body['prompt'])
                    system_msgs.extend([prompt_msg])

                for m in messages:
                    role = m['role']
                    content = m['content']
                    msg = role_formats[role].format(message=content)
                    if role == 'system':
                        system_msgs.extend([msg])
                    else:
                        chat_msgs.extend([msg])

                # can't really truncate the system messages
                system_msg = '\n'.join(system_msgs)
                if system_msg and system_msg[-1] != '\n':
                    system_msg = system_msg + '\n'

                system_token_count = len(encode(system_msg)[0])
                remaining_tokens = truncation_length - system_token_count
                chat_msg = ''

                while chat_msgs:
                    new_msg = chat_msgs.pop()
                    new_size = len(encode(new_msg)[0])
                    if new_size <= remaining_tokens:
                        chat_msg = new_msg + chat_msg
                        remaining_tokens -= new_size
                    else:
                        print(f"Warning: too many messages for context size, dropping {len(chat_msgs) + 1} oldest message(s).")
                        break

                prompt = system_msg + chat_msg + role_formats['prompt']

                token_count = len(encode(prompt)[0])

            else:
                # Text Completions
                stream_object_type = 'text_completion.chunk'
                object_type = 'text_completion'

                # ... encoded as a string, array of strings, array of tokens, or array of token arrays.
                if is_legacy:
                    prompt = body['context']  # Older engines.generate API
                else:
                    prompt = body['prompt']  # XXX this can be different types

                if isinstance(prompt, list):
                    self.openai_error("API Batched generation not yet supported.")
                    return

                token_count = len(encode(prompt)[0])
                if token_count >= truncation_length:
                    new_len = int(len(prompt) * shared.settings['truncation_length'] / token_count)
                    prompt = prompt[-new_len:]
                    new_token_count = len(encode(prompt)[0])
                    print(f"Warning: truncating prompt to {new_len} characters, was {token_count} tokens. Now: {new_token_count} tokens.")
                    token_count = new_token_count

            if truncation_length - token_count < req_params['max_new_tokens']:
                print(f"Warning: Ignoring max_new_tokens ({req_params['max_new_tokens']}), too large for the remaining context. Remaining tokens: {truncation_length - token_count}")
                req_params['max_new_tokens'] = truncation_length - token_count
                print(f"Warning: Set max_new_tokens = {req_params['max_new_tokens']}")

            if is_streaming:
                # begin streaming
                chunk = {
                    "id": cmpl_id,
                    "object": stream_object_type,
                    "created": created_time,
                    "model": shared.model_name,
                    resp_list: [{
                        "index": 0,
                        "finish_reason": None,
                    }],
                }

                if stream_object_type == 'text_completion.chunk':
                    chunk[resp_list][0]["text"] = ""
                else:
                    # So yeah... do both methods? delta and messages.
                    chunk[resp_list][0]["message"] = {'role': 'assistant', 'content': ''}
                    chunk[resp_list][0]["delta"] = {'role': 'assistant', 'content': ''}

                response = 'data: ' + json.dumps(chunk) + '\r\n\r\n'
                self.wfile.write(response.encode('utf-8'))

            # generate reply #######################################
            if debug:
                print({'prompt': prompt, 'req_params': req_params})
            generator = generate_reply(prompt, req_params, stopping_strings=stopping_strings, is_chat=False)

            answer = ''
            seen_content = ''
            longest_stop_len = max([len(x) for x in stopping_strings] + [0])

            for a in generator:
                answer = a

                stop_string_found = False
                len_seen = len(seen_content)
                search_start = max(len_seen - longest_stop_len, 0)

                for string in stopping_strings:
                    idx = answer.find(string, search_start)
                    if idx != -1:
                        answer = answer[:idx]  # clip it.
                        stop_string_found = True

                if stop_string_found:
                    break

                # If something like "\nYo" is generated just before "\nYou:"
                # is completed, buffer and generate more, don't send it
                buffer_and_continue = False

                for string in stopping_strings:
                    for j in range(len(string) - 1, 0, -1):
                        if answer[-j:] == string[:j]:
                            buffer_and_continue = True
                            break
                    else:
                        continue
                    break

                if buffer_and_continue:
                    continue

                if is_streaming:
                    # Streaming
                    new_content = answer[len_seen:]

                    if not new_content or chr(0xfffd) in new_content:  # partial unicode character, don't send it yet.
                        continue

                    seen_content = answer
                    chunk = {
                        "id": cmpl_id,
                        "object": stream_object_type,
                        "created": created_time,
                        "model": shared.model_name,
                        resp_list: [{
                            "index": 0,
                            "finish_reason": None,
                        }],
                    }

                    # strip extra leading space off new generated content
                    if len_seen == 0 and new_content[0] == ' ':
                        new_content = new_content[1:]

                    if stream_object_type == 'text_completion.chunk':
                        chunk[resp_list][0]['text'] = new_content
                    else:
                        # So yeah... do both methods? delta and messages.
                        chunk[resp_list][0]['message'] = {'content': new_content}
                        chunk[resp_list][0]['delta'] = {'content': new_content}
                    response = 'data: ' + json.dumps(chunk) + '\r\n\r\n'
                    self.wfile.write(response.encode('utf-8'))
                    completion_token_count += len(encode(new_content)[0])

            if is_streaming:
                chunk = {
                    "id": cmpl_id,
                    "object": stream_object_type,
                    "created": created_time,
                    "model": model,  # TODO: add Lora info?
                    resp_list: [{
                        "index": 0,
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": token_count,
                        "completion_tokens": completion_token_count,
                        "total_tokens": token_count + completion_token_count
                    }
                }
                if stream_object_type == 'text_completion.chunk':
                    chunk[resp_list][0]['text'] = ''
                else:
                    # So yeah... do both methods? delta and messages.
                    chunk[resp_list][0]['message'] = {'content': ''}
                    chunk[resp_list][0]['delta'] = {'content': ''}

                response = 'data: ' + json.dumps(chunk) + '\r\n\r\ndata: [DONE]\r\n\r\n'
                self.wfile.write(response.encode('utf-8'))
                # Finished if streaming.
                if debug:
                    if answer and answer[0] == ' ':
                        answer = answer[1:]
                    print({'answer': answer}, chunk)
                return

            # strip extra leading space off new generated content
            if answer and answer[0] == ' ':
                answer = answer[1:]

            if debug:
                print({'response': answer})

            completion_token_count = len(encode(answer)[0])
            stop_reason = "stop"
            if token_count + completion_token_count >= truncation_length:
                stop_reason = "length"

            resp = {
                "id": cmpl_id,
                "object": object_type,
                "created": created_time,
                "model": model,  # TODO: add Lora info?
                resp_list: [{
                    "index": 0,
                    "finish_reason": stop_reason,
                }],
                "usage": {
                    "prompt_tokens": token_count,
                    "completion_tokens": completion_token_count,
                    "total_tokens": token_count + completion_token_count
                }
            }

            if is_chat_request:
                resp[resp_list][0]["message"] = {"role": "assistant", "content": answer}
            else:
                resp[resp_list][0]["text"] = answer

            response = json.dumps(resp)
            self.wfile.write(response.encode('utf-8'))

        elif '/edits' in self.path:
            if not shared.model:
                self.openai_error("No model loaded.")
                return

            self.send_response(200)
            self.send_access_control_headers()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()

            created_time = int(time.time())

            # Using Alpaca format, this may work with other models too.
            instruction = body['instruction']
            input = body.get('input', '')

            # Request parameters
            req_params = default_req_params.copy()
            stopping_strings = []

            # Alpaca is verbose so a good default prompt
            default_template = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
            )

            instruction_template = default_template
            
            # Use the special instruction/input/response template for anything trained like Alpaca
            if shared.settings['instruction_template']:
                if 'Alpaca' in shared.settings['instruction_template']:
                    stopping_strings.extend(['\n###'])
                else:
                    try:
                        instruct = yaml.safe_load(open(f"characters/instruction-following/{shared.settings['instruction_template']}.yaml", 'r'))

                        template = instruct['turn_template']
                        template = template\
                            .replace('<|user|>', instruct.get('user', ''))\
                            .replace('<|bot|>', instruct.get('bot', ''))\
                            .replace('<|user-message|>', '{instruction}\n{input}')

                        instruction_template = instruct.get('context', '') + template[:template.find('<|bot-message|>')].rstrip(' ')
                        if instruct['user']:
                            stopping_strings.extend(['\n' + instruct['user'], instruct['user'] ])

                    except Exception as e:
                        instruction_template = default_template
                        print(f"Exception: When loading characters/instruction-following/{shared.settings['instruction_template']}.yaml: {repr(e)}")
                        print("Warning: Loaded default instruction-following template (Alpaca) for model.")
            else:
                stopping_strings.extend(['\n###'])
                print("Warning: Loaded default instruction-following template (Alpaca) for model.")
                

            edit_task = instruction_template.format(instruction=instruction, input=input)

            truncation_length = default(shared.settings, 'truncation_length', 2048)
            token_count = len(encode(edit_task)[0])
            max_tokens = truncation_length - token_count

            req_params['max_new_tokens'] = max_tokens
            req_params['truncation_length'] = truncation_length
            req_params['temperature'] = clamp(default(body, 'temperature', default_req_params['temperature']), 0.001, 1.999) # fixup absolute 0.0
            req_params['top_p'] = clamp(default(body, 'top_p', default_req_params['top_p']), 0.001, 1.0)
            req_params['seed'] = shared.settings.get('seed', default_req_params['seed'])
            req_params['add_bos_token'] = shared.settings.get('add_bos_token', default_req_params['add_bos_token'])

            if debug:
                print({'edit_template': edit_task, 'req_params': req_params, 'token_count': token_count})
            
            generator = generate_reply(edit_task, req_params, stopping_strings=stopping_strings, is_chat=False)

            longest_stop_len = max([len(x) for x in stopping_strings] + [0])
            answer = ''
            seen_content = ''
            for a in generator:
                answer = a

                stop_string_found = False
                len_seen = len(seen_content)
                search_start = max(len_seen - longest_stop_len, 0)

                for string in stopping_strings:
                    idx = answer.find(string, search_start)
                    if idx != -1:
                        answer = answer[:idx]  # clip it.
                        stop_string_found = True

                if stop_string_found:
                    break


            # some reply's have an extra leading space to fit the instruction template, just clip it off from the reply.
            if edit_task[-1] != '\n' and answer and answer[0] == ' ':
                answer = answer[1:]

            completion_token_count = len(encode(answer)[0])

            resp = {
                "object": "edit",
                "created": created_time,
                "choices": [{
                    "text": answer,
                    "index": 0,
                }],
                "usage": {
                    "prompt_tokens": token_count,
                    "completion_tokens": completion_token_count,
                    "total_tokens": token_count + completion_token_count
                }
            }

            if debug:
                print({'answer': answer, 'completion_token_count': completion_token_count})

            response = json.dumps(resp)
            self.wfile.write(response.encode('utf-8'))

        elif '/images/generations' in self.path and 'SD_WEBUI_URL' in os.environ:
            # Stable Diffusion callout wrapper for txt2img
            # Low effort implementation for compatibility. With only "prompt" being passed and assuming DALL-E
            # the results will be limited and likely poor. SD has hundreds of models and dozens of settings.
            # If you want high quality tailored results you should just use the Stable Diffusion API directly.
            # it's too general an API to try and shape the result with specific tags like "masterpiece", etc,
            # Will probably work best with the stock SD models.
            # SD configuration is beyond the scope of this API.
            # At this point I will not add the edits and variations endpoints (ie. img2img) because they
            # require changing the form data handling to accept multipart form data, also to properly support
            # url return types will require file management and a web serving files... Perhaps later!

            self.send_response(200)
            self.send_access_control_headers()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()

            width, height = [ int(x) for x in default(body, 'size', '1024x1024').split('x') ]  # ignore the restrictions on size
            response_format = default(body, 'response_format', 'url')  # or b64_json
            
            payload = {
                'prompt': body['prompt'],  # ignore prompt limit of 1000 characters
                'width': width,
                'height': height,
                'batch_size': default(body, 'n', 1)  # ignore the batch limits of max 10
            }

            resp = {
                'created': int(time.time()),
                'data': []
            }

            # TODO: support SD_WEBUI_AUTH username:password pair.
            sd_url = f"{os.environ['SD_WEBUI_URL']}/sdapi/v1/txt2img"

            response = requests.post(url=sd_url, json=payload)
            r = response.json()
            # r['parameters']...
            for b64_json in r['images']:
                if response_format == 'b64_json':
                    resp['data'].extend([{'b64_json': b64_json}])
                else:
                    resp['data'].extend([{'url': f'data:image/png;base64,{b64_json}'}])  # yeah it's lazy. requests.get() will not work with this

            response = json.dumps(resp)
            self.wfile.write(response.encode('utf-8'))

        elif '/embeddings' in self.path and embedding_model is not None:
            self.send_response(200)
            self.send_access_control_headers()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()

            input = body['input'] if 'input' in body else body['text']
            if type(input) is str:
                input = [input]

            embeddings = embedding_model.encode(input).tolist()

            def enc_emb(emb):
                # If base64 is specified, encode. Otherwise, do nothing.
                if body.get("encoding_format", "") == "base64":
                    return float_list_to_base64(emb)
                else:
                    return emb
            data = [{"object": "embedding", "embedding": enc_emb(emb), "index": n} for n, emb in enumerate(embeddings)]

            response = json.dumps({
                "object": "list",
                "data": data,
                "model": st_model,  # return the real model
                "usage": {
                    "prompt_tokens": 0,
                    "total_tokens": 0,
                }
            })

            if debug:
                print(f"Embeddings return size: {len(embeddings[0])}, number: {len(embeddings)}")
            self.wfile.write(response.encode('utf-8'))

        elif '/moderations' in self.path:
            # for now do nothing, just don't error.
            self.send_response(200)
            self.send_access_control_headers()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()

            response = json.dumps({
                "id": "modr-5MWoLO",
                "model": "text-moderation-001",
                "results": [{
                    "categories": {
                        "hate": False,
                        "hate/threatening": False,
                        "self-harm": False,
                        "sexual": False,
                        "sexual/minors": False,
                        "violence": False,
                        "violence/graphic": False
                    },
                    "category_scores": {
                        "hate": 0.0,
                        "hate/threatening": 0.0,
                        "self-harm": 0.0,
                        "sexual": 0.0,
                        "sexual/minors": 0.0,
                        "violence": 0.0,
                        "violence/graphic": 0.0
                    },
                    "flagged": False
                }]
            })
            self.wfile.write(response.encode('utf-8'))

        elif self.path == '/api/v1/token-count':
            # NOT STANDARD. lifted from the api extension, but it's still very useful to calculate tokenized length client side.
            self.send_response(200)
            self.send_access_control_headers()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()

            tokens = encode(body['prompt'])[0]
            response = json.dumps({
                'results': [{
                    'tokens': len(tokens)
                }]
            })
            self.wfile.write(response.encode('utf-8'))

        else:
            print(self.path, self.headers)
            self.send_error(404)


def run_server():
    global embedding_model
    try:
        embedding_model = SentenceTransformer(st_model)
        print(f"\nLoaded embedding model: {st_model}, max sequence length: {embedding_model.max_seq_length}")
    except:
        print(f"\nFailed to load embedding model: {st_model}")
        pass

    server_addr = ('0.0.0.0' if shared.args.listen else '127.0.0.1', params['port'])
    server = ThreadingHTTPServer(server_addr, Handler)
    if shared.args.share:
        try:
            from flask_cloudflared import _run_cloudflared
            public_url = _run_cloudflared(params['port'], params['port'] + 1)
            print(f'Starting OpenAI compatible api at\nOPENAI_API_BASE={public_url}/v1')
        except ImportError:
            print('You should install flask_cloudflared manually')
    else:
        print(f'Starting OpenAI compatible api:\nOPENAI_API_BASE=http://{server_addr[0]}:{server_addr[1]}/v1')
        
    server.serve_forever()


def setup():
    Thread(target=run_server, daemon=True).start()
