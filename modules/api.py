import json

import gradio as gr

from modules import shared
from modules.text_generation import generate_reply


def generate_reply_wrapper(string):
    generate_params = {
        'do_sample': True,
        'temperature': 1,
        'top_p': 1,
        'typical_p': 1,
        'repetition_penalty': 1,
        'encoder_repetition_penalty': 1,
        'top_k': 50,
        'num_beams': 1,
        'penalty_alpha': 0,
        'min_length': 0,
        'length_penalty': 1,
        'no_repeat_ngram_size': 0,
        'early_stopping': False,
    }
    params = json.loads(string)
    for k in params[1]:
        generate_params[k] = params[1][k]
    for i in generate_reply(params[0], generate_params):
        yield i


def create_apis():
    t1 = gr.Textbox(visible=False)
    t2 = gr.Textbox(visible=False)
    dummy = gr.Button(visible=False)

    input_params = [t1]
    output_params = [t2] + [shared.gradio[k] for k in ['markdown', 'html']]
    dummy.click(generate_reply_wrapper, input_params, output_params, api_name='textgen')
