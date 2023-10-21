import time

import yaml
from extensions.openai.defaults import get_default_req_params
from extensions.openai.errors import InvalidRequestError
from extensions.openai.utils import debug_msg
from modules import shared
from modules.text_generation import encode, generate_reply


def edits(instruction: str, input: str, temperature=1.0, top_p=1.0) -> dict:

    created_time = int(time.time() * 1000)

    # Request parameters
    req_params = get_default_req_params()
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
                instruct = yaml.safe_load(open(f"instruction-templates/{shared.settings['instruction_template']}.yaml", 'r'))

                template = instruct['turn_template']
                template = template\
                    .replace('<|user|>', instruct.get('user', ''))\
                    .replace('<|bot|>', instruct.get('bot', ''))\
                    .replace('<|user-message|>', '{instruction}\n{input}')

                instruction_template = instruct.get('context', '') + template[:template.find('<|bot-message|>')].rstrip(' ')
                if instruct['user']:
                    stopping_strings.extend(['\n' + instruct['user'], instruct['user']])

            except Exception as e:
                instruction_template = default_template
                print(f"Exception: When loading instruction-templates/{shared.settings['instruction_template']}.yaml: {repr(e)}")
                print("Warning: Loaded default instruction-following template (Alpaca) for model.")
    else:
        stopping_strings.extend(['\n###'])
        print("Warning: Loaded default instruction-following template (Alpaca) for model.")

    edit_task = instruction_template.format(instruction=instruction, input=input)

    truncation_length = shared.settings['truncation_length']

    token_count = len(encode(edit_task)[0])
    max_tokens = truncation_length - token_count

    if max_tokens < 1:
        err_msg = f"This model maximum context length is {truncation_length} tokens. However, your messages resulted in over {truncation_length - max_tokens} tokens."
        raise InvalidRequestError(err_msg, param='input')

    req_params['max_new_tokens'] = max_tokens
    req_params['truncation_length'] = truncation_length
    req_params['temperature'] = temperature
    req_params['top_p'] = top_p
    req_params['seed'] = shared.settings.get('seed', req_params['seed'])
    req_params['add_bos_token'] = shared.settings.get('add_bos_token', req_params['add_bos_token'])
    req_params['custom_stopping_strings'] = shared.settings['custom_stopping_strings']

    debug_msg({'edit_template': edit_task, 'req_params': req_params, 'token_count': token_count})

    generator = generate_reply(edit_task, req_params, stopping_strings=stopping_strings, is_chat=False)

    answer = ''
    for a in generator:
        answer = a

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

    return resp
