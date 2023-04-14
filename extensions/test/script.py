import gradio as gr
import modules.shared as shared
from modules.chat import chatbot_wrapper
import modules.ui
import json

custom_state = {}
custom_output = []

def get_params(*args):
    global custom_state
    custom_state = modules.ui.gather_interface_values(*args)
    return json.dumps(custom_state)

def pause_here(text):
    global custom_state
    global custom_output
    text = "what's it like to fly?"
    for new in chatbot_wrapper(text, custom_state):
        custom_output = new
    temp

def poop_test(flubby):
    return flubby

def ui():
    butt="name1"
    prompt = gr.Textbox(value="name1", label='you\'re a fucker!', interactive=True)
    with gr.Accordion("flippity floppity", open=True):
        make_state = gr.Button("make_state")
        test_output = gr.Button("test_output")
        tester = gr.HTML(value="what the fuck is happening?")
        custom_chat = gr.HTML(value="for the love of God, is this actually going to work???")

    prompt.change(poop_test, prompt, tester)
    make_state.click(get_params, [shared.gradio[k] for k in shared.input_elements], tester)
    test_output.click(pause_here, prompt)
