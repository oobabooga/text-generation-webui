import base64
import json
import numpy as np
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from modules import shared
from modules.text_generation import encode, generate_reply

params = {
    'port': int(os.environ.get('OPENEDAI_PORT')) if 'OPENEDAI_PORT' in os.environ else 5001,
}

debug = True if 'OPENEDAI_DEBUG' in os.environ else False

# Optional, install the module and download the model to enable
# v1/embeddings
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    pass

st_model = os.environ["OPENEDAI_EMBEDDING_MODEL"] if "OPENEDAI_EMBEDDING_MODEL" in os.environ else "all-mpnet-base-v2"
embedding_model = None

standard_stopping_strings = ['\nsystem:', '\nuser:', '\nhuman:', '\nassistant:', '\n###', ]

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
    def do_GET(self):
        if self.path.startswith('/v1/models'):

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()

            # TODO: list all models and allow model changes via API? Lora's?
            # This API should list capabilities, limits and pricing...
            models = [{
                "id": shared.model_name,  # The real chat/completions model
                "object": "model",
                "owned_by": "user",
                "permission": []
            }, {
                "id": st_model,  # The real sentence transformer embeddings model
                "object": "model",
                "owned_by": "user",
                "permission": []
            }, {  # these are expected by so much, so include some here as a dummy
                "id": "gpt-3.5-turbo",  # /v1/chat/completions
                "object": "model",
                "owned_by": "user",
                "permission": []
            }, {
                "id": "text-curie-001",  # /v1/completions, 2k context
                "object": "model",
                "owned_by": "user",
                "permission": []
            }, {
                "id": "text-davinci-002",  # /v1/embeddings text-embedding-ada-002:1536, text-davinci-002:768
                "object": "model",
                "owned_by": "user",
                "permission": []
            }]

            response = ''
            if self.path == '/v1/models':
                response = json.dumps({
                    "object": "list",
                    "data": models,
                })
            else:
                the_model_name = self.path[len('/v1/models/'):]
                response = json.dumps({
                    "id": the_model_name,
                    "object": "model",
                    "owned_by": "user",
                    "permission": []
                })

            self.wfile.write(response.encode('utf-8'))
        else:
            self.send_error(404)

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = json.loads(self.rfile.read(content_length).decode('utf-8'))

        if debug:
            print(self.headers)  # did you know... python-openai sends your linux kernel & python version?
        if debug:
            print(body)

        if '/completions' in self.path or '/generate' in self.path:
            is_legacy = '/generate' in self.path
            is_chat = 'chat' in self.path
            resp_list = 'data' if is_legacy else 'choices'

            # XXX model is ignored for now
            # model = body.get('model', shared.model_name) # ignored, use existing for now
            model = shared.model_name
            created_time = int(time.time())
            cmpl_id = "conv-%d" % (created_time)

            # Try to use openai defaults or map them to something with the same intent
            stopping_strings = default(shared.settings, 'custom_stopping_strings', [])
            if 'stop' in body:
                if isinstance(body['stop'], str):
                    stopping_strings = [body['stop']]
                elif isinstance(body['stop'], list):
                    stopping_strings = body['stop']

            truncation_length = default(shared.settings, 'truncation_length', 2048)
            truncation_length = clamp(default(body, 'truncation_length', truncation_length), 1, truncation_length)

            default_max_tokens = truncation_length if is_chat else 16  # completions default, chat default is 'inf' so we need to cap it., the default for chat is "inf"

            max_tokens_str = 'length' if is_legacy else 'max_tokens'
            max_tokens = default(body, max_tokens_str, default(shared.settings, 'max_new_tokens', default_max_tokens))

            # hard scale this, assuming the given max is for GPT3/4, perhaps inspect the requested model and lookup the context max
            while truncation_length <= max_tokens:
                max_tokens = max_tokens // 2

            req_params = {
                'max_new_tokens': max_tokens,
                'temperature': default(body, 'temperature', 1.0),
                'top_p': default(body, 'top_p', 1.0),
                'top_k': default(body, 'best_of', 1),
                # XXX not sure about this one, seems to be the right mapping, but the range is different (-2..2.0) vs 0..2
                # 0 is default in openai, but 1.0 is default in other places. Maybe it's scaled? scale it.
                'repetition_penalty': 1.18,  # (default(body, 'presence_penalty', 0) + 2.0 ) / 2.0, # 0 the real default, 1.2 is the model default, but 1.18 works better.
                # XXX not sure about this one either, same questions. (-2..2.0), 0 is default not 1.0, scale it.
                'encoder_repetition_penalty': 1.0,  # (default(body, 'frequency_penalty', 0) + 2.0) / 2.0,
                'suffix': body.get('suffix', None),
                'stream': default(body, 'stream', False),
                'echo': default(body, 'echo', False),
                #####################################################
                'seed': shared.settings.get('seed', -1),
                # int(body.get('n', 1)) # perhaps this should be num_beams or chat_generation_attempts? 'n' doesn't have a direct map
                # unofficial, but it needs to get set anyways.
                'truncation_length': truncation_length,
                # no more args.
                'add_bos_token': shared.settings.get('add_bos_token', True),
                'do_sample': True,
                'typical_p': 1.0,
                'min_length': 0,
                'no_repeat_ngram_size': 0,
                'num_beams': 1,
                'penalty_alpha': 0.0,
                'length_penalty': 1,
                'early_stopping': False,
                'ban_eos_token': False,
                'skip_special_tokens': True,
            }

            # fixup absolute 0.0's
            for par in ['temperature', 'repetition_penalty', 'encoder_repetition_penalty']:
                req_params[par] = clamp(req_params[par], 0.001, 1.999)

            self.send_response(200)
            if req_params['stream']:
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

            if is_chat:
                stream_object_type = 'chat.completions.chunk'
                object_type = 'chat.completions'

                messages = body['messages']

                system_msg = ''  # You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible. Knowledge cutoff: {knowledge_cutoff} Current date: {current_date}
                if 'prompt' in body:  # Maybe they sent both? This is not documented in the API, but some clients seem to do this.
                    system_msg = body['prompt']

                chat_msgs = []

                for m in messages:
                    role = m['role']
                    content = m['content']
                    # name = m.get('name', 'user')
                    if role == 'system':
                        system_msg += content
                    else:
                        chat_msgs.extend([f"\n{role}: {content.strip()}"])  # Strip content? linefeed?

                system_token_count = len(encode(system_msg)[0])
                remaining_tokens = req_params['truncation_length'] - req_params['max_new_tokens'] - system_token_count
                chat_msg = ''

                while chat_msgs:
                    new_msg = chat_msgs.pop()
                    new_size = len(encode(new_msg)[0])
                    if new_size <= remaining_tokens:
                        chat_msg = new_msg + chat_msg
                        remaining_tokens -= new_size
                    else:
                        # TODO: clip a message to fit?
                        # ie. user: ...<clipped message>
                        break

                if len(chat_msgs) > 0:
                    print(f"truncating chat messages, dropping {len(chat_msgs)} messages.")

                if system_msg:
                    prompt = 'system: ' + system_msg + '\n' + chat_msg + '\nassistant: '
                else:
                    prompt = chat_msg + '\nassistant: '

                token_count = len(encode(prompt)[0])

                # pass with some expected stop strings.
                # some strange cases of "##| Instruction: " sneaking through.
                stopping_strings += standard_stopping_strings
                req_params['custom_stopping_strings'] = stopping_strings
            else:
                stream_object_type = 'text_completion.chunk'
                object_type = 'text_completion'

                # ... encoded as a string, array of strings, array of tokens, or array of token arrays.
                if is_legacy:
                    prompt = body['context']  # Older engines.generate API
                else:
                    prompt = body['prompt']  # XXX this can be different types

                if isinstance(prompt, list):
                    prompt = ''.join(prompt)  # XXX this is wrong... need to split out to multiple calls?

                token_count = len(encode(prompt)[0])
                if token_count >= req_params['truncation_length']:
                    new_len = int(len(prompt) * (float(shared.settings['truncation_length']) - req_params['max_new_tokens']) / token_count)
                    prompt = prompt[-new_len:]
                    print(f"truncating prompt to {new_len} characters, was {token_count} tokens. Now: {len(encode(prompt)[0])} tokens.")

                # pass with some expected stop strings.
                # some strange cases of "##| Instruction: " sneaking through.
                stopping_strings += standard_stopping_strings
                req_params['custom_stopping_strings'] = stopping_strings

            if req_params['stream']:
                shared.args.chat = True
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
                    # This is coming back as "system" to the openapi cli, not sure why.
                    # So yeah... do both methods? delta and messages.
                    chunk[resp_list][0]["message"] = {'role': 'assistant', 'content': ''}
                    chunk[resp_list][0]["delta"] = {'role': 'assistant', 'content': ''}
                    # { "role": "assistant" }

                response = 'data: ' + json.dumps(chunk) + '\n'
                self.wfile.write(response.encode('utf-8'))

            # generate reply #######################################
            if debug:
                print({'prompt': prompt, 'req_params': req_params, 'stopping_strings': stopping_strings})
            generator = generate_reply(prompt, req_params, stopping_strings=stopping_strings)

            answer = ''
            seen_content = ''
            longest_stop_len = max([len(x) for x in stopping_strings])

            for a in generator:
                if isinstance(a, str):
                    answer = a
                else:
                    answer = a[0]

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

                if req_params['stream']:
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
                    if stream_object_type == 'text_completion.chunk':
                        chunk[resp_list][0]['text'] = new_content
                    else:
                        # So yeah... do both methods? delta and messages.
                        chunk[resp_list][0]['message'] = {'content': new_content}
                        chunk[resp_list][0]['delta'] = {'content': new_content}
                    response = 'data: ' + json.dumps(chunk) + '\n'
                    self.wfile.write(response.encode('utf-8'))
                    completion_token_count += len(encode(new_content)[0])

            if req_params['stream']:
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
                    chunk[resp_list][0]['delta'] = {}
                response = 'data: ' + json.dumps(chunk) + '\ndata: [DONE]\n'
                self.wfile.write(response.encode('utf-8'))
                # Finished if streaming.
                if debug:
                    print({'response': answer})
                return

            if debug:
                print({'response': answer})

            completion_token_count = len(encode(answer)[0])
            stop_reason = "stop"
            if token_count + completion_token_count >= req_params['truncation_length']:
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

            if is_chat:
                resp[resp_list][0]["message"] = {"role": "assistant", "content": answer}
            else:
                resp[resp_list][0]["text"] = answer

            response = json.dumps(resp)
            self.wfile.write(response.encode('utf-8'))
        elif '/embeddings' in self.path and embedding_model is not None:
            self.send_response(200)
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
            print(f'Starting OpenAI compatible api at {public_url}/')
        except ImportError:
            print('You should install flask_cloudflared manually')
    else:
        print(f'Starting OpenAI compatible api at http://{server_addr[0]}:{server_addr[1]}/')
    server.serve_forever()


def setup():
    Thread(target=run_server, daemon=True).start()
