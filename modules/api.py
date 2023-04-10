import json

import gradio as gr

from modules import shared
from modules.text_generation import generate_reply
from modules.chat import chatbot_wrapper, save_history

chat_api = False

def generate_reply_wrapper(string):
    global chat_api

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
        'stop_at_newline': False,
        "chat_prompt_size": 2048,
        "chat_generation_attempts": 1,
    }
    params = json.loads(string)
    for k in params[1]:
        generate_params[k] = params[1][k]
    
    if chat_api:
        # Back up the old no_stream value and set no_stream to True (required for API to work correctly)
        no_stream = shared.args.no_stream
        shared.args.no_stream = True

        # Need to figure out why I need to use the .value here and why aren't they being updated when changing values on the UI anymore?
        for i in chatbot_wrapper(params[0], generate_params, shared.gradio['name1'].value, shared.gradio['name2'].value, shared.gradio['context'].value, shared.gradio['Chat mode'].value, shared.gradio['end_of_turn'].value, True):
            # I'm not sure how to do this properly in Python, this is just basically letting the generator finish. I did the yield shared.history['visible'][-1] here, but then I couldn't force the save or reset the streaming variable
            pass
        
        # Reset no_stream to backed up value
        shared.args.no_stream = no_stream

        # Save prompt and reply to persistent chat log
        save_history(timestamp=False)

        yield shared.history['visible'][-1]
    else:
        for i in generate_reply(params[0], generate_params):
            yield i


def create_apis():
    global chat_api

    t1 = gr.Textbox(visible=False)
    t2 = gr.Textbox(visible=False)
    dummy = gr.Button(visible=False)

    input_params = [t1]
    output_params = [t2] + [shared.gradio[k] for k in ['display']] if chat_api else [shared.gradio[k] for k in ['markdown', 'html']]
    dummy.click(generate_reply_wrapper, input_params, output_params, api_name='textgen')

def create_chat_apis():
    global chat_api
    
    chat_api = True
    create_apis()
