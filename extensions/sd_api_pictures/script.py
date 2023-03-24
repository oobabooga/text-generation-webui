import base64
import io
import re
from pathlib import Path

import gradio as gr
import modules.chat as chat
import modules.shared as shared
import requests
import torch
from PIL import Image

torch._C._jit_set_profiling_mode(False)

# parameters which can be customized in settings.json of webui  
params = {
    'enable_SD_api': False,
    'address': 'http://127.0.0.1:7860',
    'save_img': False,
    'SD_model': 'NeverEndingDream', # not really used right now
    'prompt_prefix': '(Masterpiece:1.1), (solo:1.3), detailed, intricate, colorful',
    'negative_prompt': '(worst quality, low quality:1.3)',
    'side_length': 512,
    'restore_faces': False
}

SD_models = ['NeverEndingDream'] # TODO: get with http://{address}}/sdapi/v1/sd-models and allow user to select

streaming_state = shared.args.no_stream # remember if chat streaming was enabled
picture_response = False # specifies if the next model response should appear as a picture
pic_id = 0

def remove_surrounded_chars(string):
    # this expression matches to 'as few symbols as possible (0 upwards) between any asterisks' OR
    # 'as few symbols as possible (0 upwards) between an asterisk and the end of the string'
    return re.sub('\*[^\*]*?(\*|$)','',string)

# I don't even need input_hijack for this as visible text will be commited to history as the unmodified string
def input_modifier(string):
    """
    This function is applied to your text inputs before
    they are fed into the model.
    """
    global params, picture_response
    if not params['enable_SD_api']:
        return string

    commands = ['send', 'mail', 'me']
    mediums = ['image', 'pic', 'picture', 'photo']
    subjects = ['yourself', 'own']
    lowstr = string.lower()

    # TODO: refactor out to separate handler and also replace detection with a regexp
    if any(command in lowstr for command in commands) and any(case in lowstr for case in mediums): # trigger the generation if a command signature and a medium signature is found
        picture_response = True
        shared.args.no_stream = True                                                               # Disable streaming cause otherwise the SD-generated picture would return as a dud
        shared.processing_message = "*Is sending a picture...*"
        string = "Please provide a detailed description of your surroundings, how you look and the situation you're in and what you are doing right now"
        if any(target in lowstr for target in subjects):                                           # the focus of the image should be on the sending character
            string = "Please provide a detailed and vivid description of how you look and what you are wearing"

    return string

# Get and save the Stable Diffusion-generated picture
def get_SD_pictures(description):

    global params, pic_id

    payload = {
        "prompt": params['prompt_prefix'] + description,
        "seed": -1,
        "sampler_name": "DPM++ 2M Karras",
        "steps": 32,
        "cfg_scale": 7,
        "width": params['side_length'],
        "height": params['side_length'],
        "restore_faces": params['restore_faces'],
        "negative_prompt": params['negative_prompt']
    }
    
    response = requests.post(url=f'{params["address"]}/sdapi/v1/txt2img', json=payload)
    r = response.json()

    visible_result = ""
    for img_str in r['images']:
        image = Image.open(io.BytesIO(base64.b64decode(img_str.split(",",1)[0])))
        if params['save_img']:
            output_file = Path(f'extensions/sd_api_pictures/outputs/{pic_id:06d}.png')
            image.save(output_file.as_posix())
            pic_id += 1
        # lower the resolution of received images for the chat, otherwise the log size gets out of control quickly with all the base64 values in visible history
        image.thumbnail((300, 300))
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        buffered.seek(0)
        image_bytes = buffered.getvalue()
        img_str = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode()
        visible_result = visible_result + f'<img src="{img_str}" alt="{description}">\n'
    
    return visible_result

# TODO: how do I make the UI history ignore the resulting pictures (I don't want HTML to appear in history)
# and replace it with 'text' for the purposes of logging?
def output_modifier(string):
    """
    This function is applied to the model outputs.
    """
    global pic_id, picture_response, streaming_state

    if not picture_response:
        return string

    string = remove_surrounded_chars(string)
    string = string.replace('"', '')
    string = string.replace('“', '')
    string = string.replace('\n', ' ')
    string = string.strip()

    if string == '':
        string = 'no viable description in reply, try regenerating'

    # I can't for the love of all that's holy get the name from shared.gradio['name1'], so for now it will be like this
    text = f'*Description: "{string}"*'

    image = get_SD_pictures(string)

    picture_response = False

    shared.processing_message = "*Is typing...*"
    shared.args.no_stream = streaming_state
    return image + "\n" + text

def bot_prefix_modifier(string):
    """
    This function is only applied in chat mode. It modifies
    the prefix text for the Bot and can be used to bias its
    behavior.
    """

    return string

def force_pic():
    global picture_response
    picture_response = True

def ui():

    # Gradio elements
    with gr.Accordion("Stable Diffusion api integration", open=True):
        with gr.Row():
            with gr.Column():
                enable = gr.Checkbox(value=params['enable_SD_api'], label='Activate SD Api integration')
                save_img = gr.Checkbox(value=params['save_img'], label='Keep original received images in the outputs subdir')
            with gr.Column():
                address = gr.Textbox(placeholder=params['address'], value=params['address'], label='Stable Diffusion host address')
        
        with gr.Row():
            force_btn = gr.Button("Force the next response to be a picture")
            generate_now_btn = gr.Button("Generate an image response to the input")

        with gr.Accordion("Generation parameters", open=False):
            prompt_prefix = gr.Textbox(placeholder=params['prompt_prefix'], value=params['prompt_prefix'], label='Prompt Prefix (best used to describe the look of the character)')
            with gr.Row():
                negative_prompt = gr.Textbox(placeholder=params['negative_prompt'], value=params['negative_prompt'], label='Negative Prompt')
                dimensions = gr.Slider(256,702,value=params['side_length'],step=64,label='Image dimensions')
                # model = gr.Dropdown(value=SD_models[0], choices=SD_models, label='Model')
    
    # Event functions to update the parameters in the backend
    enable.change(lambda x: params.update({"enable_SD_api": x}), enable, None)
    save_img.change(lambda x: params.update({"save_img": x}), save_img, None)
    address.change(lambda x: params.update({"address": x}), address, None)
    prompt_prefix.change(lambda x: params.update({"prompt_prefix": x}), prompt_prefix, None)
    negative_prompt.change(lambda x: params.update({"negative_prompt": x}), negative_prompt, None)
    dimensions.change(lambda x: params.update({"side_length": x}), dimensions, None)
    # model.change(lambda x: params.update({"SD_model": x}), model, None)

    force_btn.click(force_pic)
    generate_now_btn.click(force_pic)
    generate_now_btn.click(eval('chat.cai_chatbot_wrapper'), shared.input_params, shared.gradio['display'], show_progress=shared.args.no_stream)