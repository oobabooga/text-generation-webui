import json

import gradio as gr

from modules import shared
from modules.text_generation import generate_reply

# set this to True to rediscover the fn_index using the browser DevTools
VISIBLE = False


def generate_reply_wrapper(string):

    # Provide defaults so as to not break the API on the client side when new parameters are added
    generate_params = {
        'max_new_tokens': 200,
        'do_sample': True,
        'temperature': 0.5,
        'top_p': 1,
        'typical_p': 1,
        'repetition_penalty': 1.1,
        'encoder_repetition_penalty': 1,
        'top_k': 0,
        'min_length': 0,
        'no_repeat_ngram_size': 0,
        'num_beams': 1,
        'penalty_alpha': 0,
        'length_penalty': 1,
        'early_stopping': False,
        'seed': -1,
        'add_bos_token': True,
        'custom_stopping_strings': '',
        'truncation_length': 2048,
        'ban_eos_token': False,
        'skip_special_tokens': True,
        'stopping_strings': [],
    }
    params = json.loads(string)
    generate_params.update(params[1])
    stopping_strings = generate_params.pop('stopping_strings')
    for i in generate_reply(params[0], generate_params, stopping_strings=stopping_strings):
        yield i


def create_apis():
    t1 = gr.Textbox(visible=VISIBLE)
    t2 = gr.Textbox(visible=VISIBLE)
    dummy = gr.Button(visible=VISIBLE)

    input_params = [t1]
    output_params = [t2] + [shared.gradio[k] for k in ['markdown', 'html']]
    dummy.click(generate_reply_wrapper, input_params, output_params, api_name='textgen')
