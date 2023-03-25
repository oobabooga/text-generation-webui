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
    'address': '127.0.0.1:7860',
    'mode': 0, # modes of operation: 0 (Manual only), 1 (Immersive/Interactive - looks for words to trigger), 2 (Picturebook Adventure - Always on)
    'manage_VRAM': True,
    'save_img': False,
    'SD_model': 'NeverEndingDream', # not used right now
    'prompt_prefix': '(Masterpiece:1.1), detailed, intricate, colorful',
    'negative_prompt': '(worst quality, low quality:1.3)',
    'width': 512,
    'height': 512,
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

def input_modifier(string):
    """
    This function is applied to your text inputs before
    they are fed into the model.
    """
    global params
    if (not params['mode']==1):
        return string

    # TODO: refactor out to separate handler and also replace detection with a regexp
    commands = ['send', 'mail', 'me']
    mediums = ['image', 'pic', 'picture', 'photo']
    subjects = ['yourself', 'own']
    lowstr = string.lower()

    if (params['mode']==1) and any(command in lowstr for command in commands) and any(case in lowstr for case in mediums): # trigger the generation if a command signature and a medium signature is found
        toggle_generation(True)
        string = "Please provide a detailed description of your surroundings, how you look and the situation you're in and what you are doing right now"
        if any(target in lowstr for target in subjects):                                           # the focus of the image should be on the sending character
            string = "Please provide a detailed and vivid description of how you look and what you are wearing"

    return string

# Get and save the Stable Diffusion-generated picture
def get_SD_pictures(description):

    global params, pic_id

    if params['manage_VRAM']:
        shared.gradio['unload_model_fn']()
        response = requests.post(url=f'http://{params["address"]}/sdapi/v1/reload-checkpoint', json='')
        response.raise_for_status()
        r = response.json()
        del response
        del r

    payload = {
        "prompt": params['prompt_prefix'] + description,
        "seed": -1,
        "sampler_name": "DPM++ 2M Karras",
        "steps": 32,
        "cfg_scale": 7,
        "width": params['width'],
        "height": params['height'],
        "restore_faces": params['restore_faces'],
        "negative_prompt": params['negative_prompt']
    }
    
    print(f'Prompting the image generator via the API on {params["address"]}…')
    response = requests.post(url=f'http://{params["address"]}/sdapi/v1/txt2img', json=payload)
    response.raise_for_status()
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

    if params['manage_VRAM']:
        response = requests.post(url=f'http://{params["address"]}/sdapi/v1/unload-checkpoint', json='')
        response.raise_for_status()
        r = response.json()
        del response
        del r
        shared.gradio['reload_model_fn']()
    
    return visible_result

# TODO: how do I make the UI history ignore the resulting pictures (I don't want HTML to appear in history)
# and replace it with 'text' for the purposes of logging?
def output_modifier(string):
    """
    This function is applied to the model outputs.
    """
    global pic_id, picture_response, params

    if not picture_response:
        return string

    string = remove_surrounded_chars(string)
    string = string.replace('"', '')
    string = string.replace('“', '')
    string = string.replace('\n', ' ')
    string = string.strip()

    if string == '':
        string = 'no viable description in reply, try regenerating'

    text = ""
    if (params['mode']<2):
        toggle_generation(False)
        text = f'*Sends a picture which portrays: “{string}”*'
    else:
        text = string

    string = get_SD_pictures(string) + "\n" + text

    return string 

def bot_prefix_modifier(string):
    """
    This function is only applied in chat mode. It modifies
    the prefix text for the Bot and can be used to bias its
    behavior.
    """

    return string

def toggle_generation(*args):
    global picture_response, shared, streaming_state
    if not args:
        picture_response = not picture_response
    else:
        picture_response = args[0]

    shared.args.no_stream = True if picture_response else streaming_state # Disable streaming cause otherwise the SD-generated picture would return as a dud
    shared.processing_message = "*Is sending a picture...*" if picture_response else "*Is typing...*"
    btn_text = "Suppress the picture response" if picture_response else "Force the picture response"
    
    return btn_text

def filter_address(address):
    address = address.strip()
    address = re.sub('http(s)?:\/\/|\/$','',address) # remove starting http:// OR https:// OR trailing slash
    return address

def SD_api_address_update(address):
    
    global params
    
    msg = "✔️ SD API is found on:"
    address = filter_address(address)
    params.update({"address": address})
    try:
        response = requests.get(url=f'http://{params["address"]}/sdapi/v1/sd-models')
        response.raise_for_status()
        r = response.json()
    except:
        msg = "❌ No SD API endpoint on:"

    return gr.Textbox.update(label=msg)



def ui():

    # Gradio elements
    gr.Markdown('### Stable Diffusion API Pictures')
    with gr.Accordion("Parameters", open=True):
        with gr.Row():
            address = gr.Textbox(placeholder=params['address'], value=params['address'], label='Automatic1111\'s WebUI address')
            mode = gr.Dropdown(["Manual", "Immersive \ Interactive", "Picturebook \ Adventure"], value="Manual", label="Mode of operation", type="index")
            with gr.Column():
                manage_VRAM = gr.Checkbox(value=params['manage_VRAM'], label='Manage VRAM')
                save_img = gr.Checkbox(value=params['save_img'], label='Keep original received images in the outputs subdir')

            toggle_gen = gr.Button("Force (Suppress) the picture response")

        with gr.Accordion("Generation parameters", open=False):
            prompt_prefix = gr.Textbox(placeholder=params['prompt_prefix'], value=params['prompt_prefix'], label='Prompt Prefix (best used to describe the look of the character)')
            with gr.Row():
                negative_prompt = gr.Textbox(placeholder=params['negative_prompt'], value=params['negative_prompt'], label='Negative Prompt')
                with gr.Column():
                    width = gr.Slider(256,704,value=params['width'],step=64,label='Width')
                    height = gr.Slider(256,704,value=params['height'],step=64,label='Height')
    
    # Event functions to update the parameters in the backend
    address.change(lambda x: params.update({"address": filter_address(x)}), address, None)
    mode.change(lambda x: params.update({"mode": x }), mode, None)
    manage_VRAM.change(lambda x: params.update({"manage_VRAM": x}), manage_VRAM, None)
    save_img.change(lambda x: params.update({"save_img": x}), save_img, None)
    
    address.submit(fn=SD_api_address_update, inputs=address, outputs=address)
    prompt_prefix.change(lambda x: params.update({"prompt_prefix": x}), prompt_prefix, None)
    negative_prompt.change(lambda x: params.update({"negative_prompt": x}), negative_prompt, None)
    width.change(lambda x: params.update({"width": x}), width, None)
    height.change(lambda x: params.update({"height": x}), height, None)
    
    toggle_gen.click(fn=toggle_generation, inputs=None, outputs=toggle_gen)
