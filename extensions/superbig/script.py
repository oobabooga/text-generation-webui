import gradio as gr

from modules import shared

from .provider import PseudocontextProvider, make_instruct_provider

provider = PseudocontextProvider()
instruct_provider = make_instruct_provider()

data = ''

def input_modifier(string):
    """
    This function is applied to your text inputs before
    they are fed into the model.
    """

    # Use data here

    return instruct_provider.with_pseudocontext(string)

def ui():
    if shared.is_chat():
        # Chat mode has to be handled differently, probably using a custom_generate_chat_prompt
        pass
    else:
        data_input = gr.Textbox(lines=20, label='Input data')
        data_input.change(lambda x: globals().update(data=x))
