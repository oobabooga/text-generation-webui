import ast
import json, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from modules import shared
from modules.text_generation import encode, generate_reply

params = {
    'port': 5001,
}

# little helper to get defaults if arg is present but None and should be the same type as default.
def default(dic, key, default):
    val = dic.get(key, default)
    if type(val) != type(default):
        # maybe it's just something like 1 instead of 1.0
        try:
            v = type(default)(val)
            if type(val)(v) == val: # if it's the same value passed in, it's ok.
                return v
        except:
            pass

        val = default
    return val

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/v1/models':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = json.dumps({
                "object": "list",
                "data": [
                    {
                    "id": shared.model_name,
                    "object": "model",
                    "owned_by": "free",
                    "permission": []
                    },
                    { # this is expected by so much, so include it here as a dummy
                    "id": "gpt-35-turbo",
                    "object": "model",
                    "owned_by": "openai",
                    "permission": []
                    }
                    # TODO: list all models and allow model changes via API? Lora's?
                ],
            })

            self.wfile.write(response.encode('utf-8'))
        else:
            self.send_error(404)

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = json.loads(self.rfile.read(content_length).decode('utf-8'))

#       print(self.headers) # did you know... python-openai sends your linux kernel & python version?
#       print(body)

        if self.path == '/v1/completions' or self.path == '/v1/chat/completions':
            # XXX model is ignored for now
            #model = body.get('model', shared.model_name) # ignored, use existing for now
            model = shared.model_name
            created_time = int(time.time())
            cmpl_id = "conv-%d" % (created_time)

            # Try to use openai defaults or map them to something with the same intent

            req_params = {
                'max_new_tokens': default(body, 'max_tokens', default(shared.settings, 'max_new_tokens', 16)),
                'temperature': default(body, 'temperature', 1.0),
                'top_p': default(body, 'top_p', 1.0),
                'top_k': default(body, 'best_of', 1),
                ### XXX not sure about this one, seems to be the right mapping, but the range is different (-2..2.0)
                # 0 is default in openai, but 1.0 is default in other places. Maybe it's scaled?
                'repetition_penalty': default(body, 'presence_penalty', 1.18), # 1.2 is the real default
                ### XXX not sure about this one either, same questions. (-2..2.0), 0 is default not 1.0
                'encoder_repetition_penalty': default(body, 'frequency_penalty', 1.0),
                # stopping strings are tricky to handle... not sure this ends up as expected wrt \n, quotes and spaces.
                #'stopping_strings': default(body, 'stop', default(shared.settings, 'stopping_strings', '')),
                'custom_stopping_strings': default(body, 'stop', default(shared.settings, 'custom_stopping_strings', [])),
                'suffix': body.get('suffix', None),
                'stream': default(body, 'stream', False),
                'echo': default(body, 'echo', False),
                #####################################################
                'seed': shared.settings.get('seed', -1),
                #int(body.get('n', 1)) # perhaps this should be num_beams or chat_generation_attempts? 'n' doesn't have a direct map.
                # unofficial, but it needs to get set anyways.
                'truncation_length': default(body, 'truncation_length', default(shared.settings, 'truncation_length', 2048)),
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

#            print ({'req_params': req_params})

            self.send_response(200)
            if req_params['stream']:
                self.send_header('Content-Type', 'text/event-stream')
            else:
                self.send_header('Content-Type', 'application/json')
            self.end_headers()

            token_count = 0
            completion_token_count = 0
            prompt = ''
            stream_object_type = ''
            object_type = ''

            if self.path == '/v1/completions':
                stream_object_type = 'text_completion.chunk'
                object_type = 'text_completion'

                prompt = body['prompt']

                token_count = len(encode(prompt)[0])
                if token_count >= req_params['truncation_length']:
                    new_len = int(len(prompt) * (float(shared.settings['truncation_length'])  - req_params['max_new_tokens']) / token_count)
                    prompt = prompt[-new_len:]
                    print(f"truncating prompt to {new_len} characters, was {token_count} tokens. Now: {len(encode(prompt)[0])} tokens.")

            elif self.path == '/v1/chat/completions':
                stream_object_type = 'chat.completions.chunk'
                object_type = 'chat.completions'

                messages = body['messages']

                system_msg = ''
                chat_msgs = []

                for m in messages:
                    role = m['role']
                    content = m['content']
                    #name = m.get('name', 'user')
                    if role == 'system':
                        system_msg += content
                    else:
                        chat_msgs.extend([f"\n{role}: {content.strip()}"]) ### Strip content? linefeed?
                
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

                prompt = system_msg + '\n' +  chat_msg
                token_count = len(encode(prompt)[0])

            # XXX this is borken
            #stopping_strings = '' #ast.literal_eval(f"[{req_params['custom_stopping_strings']}]")

            shared.args.no_stream = not req_params['stream']
            if not shared.args.no_stream:
                shared.args.chat = True
                # begin streaming
                chunk = {
                    "choices": [{
                        "finish_reason": None,
                        "index": 0
                    }],
                    "created": created_time,
                    "id": cmpl_id,
                    "model": shared.model_name,
                    "object": stream_object_type,
                }

                if stream_object_type == 'text_completion.chunk':
                    chunk["choices"][0]["text"] = ""
                else:
                    # This is coming back as "system" to the openapi cli, not sure why.
                    # So yeah... do both methods? delta and messages. 
                    chunk["choices"][0]["message"] = {'role': 'assistant', 'content': ''}
                    chunk["choices"][0]["delta"] = {'role': 'assistant', 'content': ''}
                    #{ "role": "assistant" }

                response = 'data: ' + json.dumps(chunk) + '\n'
                self.wfile.write(response.encode('utf-8'))
                
            generator = generate_reply(prompt, req_params, stopping_strings=req_params['custom_stopping_strings'])

            answer = ''
            seen_content = ''
            
            for a in generator:
                if isinstance(a, str):
                    answer = a
                else:
                    answer = a[0]
                if not shared.args.no_stream:
                    # Streaming
                    new_content = answer[len(seen_content):]
                    seen_content = answer
                    chunk = {
                        "id": cmpl_id,
                        "object": stream_object_type,
                        "created": created_time,
                        "model": shared.model_name,
                        "choices": [{
                            "index": 0,
                            "finish_reason": None,
                        }],
                    }
                    if stream_object_type == 'text_completion.chunk':
                        chunk['choices'][0]['text'] = new_content
                    else:
                        # So yeah... do both methods? delta and messages. 
                        chunk['choices'][0]['message'] = { 'content': new_content }
                        chunk['choices'][0]['delta'] = { 'content': new_content }
                    response = 'data: ' + json.dumps(chunk) + '\n'
                    self.wfile.write(response.encode('utf-8'))
                    completion_token_count += len(encode(new_content)[0])

            if not shared.args.no_stream:
                chunk = {
                    "id": cmpl_id,
                    "object": stream_object_type,
                    "created": created_time,
                    "model": model, # TODO: add Lora info?
                    "choices": [{
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
                    chunk['choices'][0]['text'] = ''
                else:
                    # So yeah... do both methods? delta and messages. 
                    chunk['choices'][0]['message'] = {'content': '' }
                    chunk['choices'][0]['delta'] = {}
                response = 'data: ' + json.dumps(chunk) + '\ndata: [DONE]\n'
                self.wfile.write(response.encode('utf-8'))
                ###### Finished if streaming.
#                print({'prompt': prompt}, req_params)
                return
            
#            print({'prompt': prompt, 'answer': answer}, req_params)

            completion_token_count = len(encode(answer)[0])
            stop_reason = "stop"
            if token_count + completion_token_count >= req_params['truncation_length']:
                stop_reason = "length"

            resp = {
                "id": cmpl_id,
                "object": object_type,
                "created": created_time,
                "model": model, # TODO: add Lora info?
                "choices": [{
                    "index": 0,
                    "finish_reason": stop_reason,
                }],
                "usage": {
                    "prompt_tokens": token_count,
                    "completion_tokens": completion_token_count,
                    "total_tokens": token_count + completion_token_count
                }
            }

            if self.path == '/v1/completions':
                resp["choices"][0]["text"] = answer
            elif self.path == '/v1/chat/completions':
                resp["choices"][0]["message"] = {"role": "assistant", "content": answer }

            response = json.dumps(resp)
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
