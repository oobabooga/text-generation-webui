import base64
from io import BytesIO

import gradio as gr

import modules.chat as chat
import modules.shared as shared
from modules.bot_picture import caption_image

params = {
}

# If 'state' is True, will hijack the next chat generation with
# custom input text
input_hijack = {
    'state': False,
    'value': ["", ""]
}

def generate_chat_picture(picture, name1, name2):
    text = f'*{name1} sends {name2} a picture that contains the following: "{caption_image(picture)}"*'
    buffer = BytesIO()
    picture.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    visible_text = f'<img src="data:image/jpeg;base64,{img_str}">'
    return text, visible_text

def input_modifier(string):
    """
    This function is applied to your text inputs before
    they are fed into the model.
    """ 

    return string

def output_modifier(string):
    """
    This function is applied to the model outputs.
    """

    return string

def bot_prefix_modifier(string):
    """
    This function is only applied in chat mode. It modifies
    the prefix text for the Bot and can be used to bias its
    behavior.
    """

    return string

def ui():
    picture_select = gr.Image(label='Send a picture', type='pil')

    function_call = 'chat.cai_chatbot_wrapper' if shared.args.cai_chat else 'chat.chatbot_wrapper'
    picture_select.upload(lambda picture, name1, name2: input_hijack.update({"state": True, "value": generate_chat_picture(picture, name1, name2)}), [picture_select, shared.gradio['name1'], shared.gradio['name2']], None)
    picture_select.upload(eval(function_call), shared.input_params, shared.gradio['display'], show_progress=shared.args.no_stream)
    picture_select.upload(lambda : None, [], [picture_select], show_progress=False)
