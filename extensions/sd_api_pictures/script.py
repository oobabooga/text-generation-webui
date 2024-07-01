import base64
import io
import re
import time
from datetime import date
from pathlib import Path

import gradio as gr
import requests
import torch
from PIL import Image

from modules import shared
from modules.models import reload_model, unload_model
from modules.ui import create_refresh_button

torch._C._jit_set_profiling_mode(False)

# parameters which can be customized in settings.json of webui
params = {
    'address': 'http://127.0.0.1:7860',
    'mode': 0,  # modes of operation: 0 (Manual only), 1 (Immersive/Interactive - looks for words to trigger), 2 (Picturebook Adventure - Always on)
    'manage_VRAM': False,
    'save_img': False,
    'SD_model': 'NeverEndingDream',  # not used right now
    'prompt_prefix': '(Masterpiece:1.1), detailed, intricate, colorful',
    'negative_prompt': '(worst quality, low quality:1.3)',
    'width': 512,
    'height': 512,
    'denoising_strength': 0.61,
    'restore_faces': False,
    'enable_hr': False,
    'hr_upscaler': 'ESRGAN_4x',
    'hr_scale': '1.0',
    'seed': -1,
    'sampler_name': 'DPM++ 2M',
    'steps': 32,
    'cfg_scale': 7,
    'textgen_prefix': 'Please provide a detailed and vivid description of [subject]',
    'sd_checkpoint': ' ',
    'checkpoint_list': [" "]
}


def give_VRAM_priority(actor):
    global shared, params

    if actor == 'SD':
        unload_model()
        print("Requesting Auto1111 to re-load last checkpoint used...")
        response = requests.post(url=f'{params["address"]}/sdapi/v1/reload-checkpoint', json='')
        response.raise_for_status()

    elif actor == 'LLM':
        print("Requesting Auto1111 to vacate VRAM...")
        response = requests.post(url=f'{params["address"]}/sdapi/v1/unload-checkpoint', json='')
        response.raise_for_status()
        reload_model()

    elif actor == 'set':
        print("VRAM mangement activated -- requesting Auto1111 to vacate VRAM...")
        response = requests.post(url=f'{params["address"]}/sdapi/v1/unload-checkpoint', json='')
        response.raise_for_status()

    elif actor == 'reset':
        print("VRAM mangement deactivated -- requesting Auto1111 to reload checkpoint")
        response = requests.post(url=f'{params["address"]}/sdapi/v1/reload-checkpoint', json='')
        response.raise_for_status()

    else:
        raise RuntimeError(f'Managing VRAM: "{actor}" is not a known state!')

    response.raise_for_status()
    del response


if params['manage_VRAM']:
    give_VRAM_priority('set')

SD_models = ['NeverEndingDream']  # TODO: get with http://{address}}/sdapi/v1/sd-models and allow user to select

picture_response = False  # specifies if the next model response should appear as a picture


def remove_surrounded_chars(string):
    # this expression matches to 'as few symbols as possible (0 upwards) between any asterisks' OR
    # 'as few symbols as possible (0 upwards) between an asterisk and the end of the string'
    return re.sub('\*[^\*]*?(\*|$)', '', string)


def triggers_are_in(string):
    string = remove_surrounded_chars(string)
    # regex searches for send|main|message|me (at the end of the word) followed by
    # a whole word of image|pic|picture|photo|snap|snapshot|selfie|meme(s),
    # (?aims) are regex parser flags
    return bool(re.search('(?aims)(send|mail|message|me)\\b.+?\\b(image|pic(ture)?|photo|snap(shot)?|selfie|meme)s?\\b', string))


def state_modifier(state):
    if picture_response:
        state['stream'] = False

    return state


def input_modifier(string):
    """
    This function is applied to your text inputs before
    they are fed into the model.
    """

    global params

    if not params['mode'] == 1:  # if not in immersive/interactive mode, do nothing
        return string

    if triggers_are_in(string):  # if we're in it, check for trigger words
        toggle_generation(True)
        string = string.lower()
        if "of" in string:
            subject = string.split('of', 1)[1]  # subdivide the string once by the first 'of' instance and get what's coming after it
            string = params['textgen_prefix'].replace("[subject]", subject)
        else:
            string = params['textgen_prefix'].replace("[subject]", "your appearance, your surroundings and what you are doing right now")

    return string

# Get and save the Stable Diffusion-generated picture
def get_SD_pictures(description, character):

    global params

    if params['manage_VRAM']:
        give_VRAM_priority('SD')

    description = re.sub('<audio.*?</audio>', ' ', description)
    description = f"({description}:1)"

    payload = {
        "prompt": params['prompt_prefix'] + description,
        "seed": params['seed'],
        "sampler_name": params['sampler_name'],
        "enable_hr": params['enable_hr'],
        "hr_scale": params['hr_scale'],
        "hr_upscaler": params['hr_upscaler'],
        "denoising_strength": params['denoising_strength'],
        "steps": params['steps'],
        "cfg_scale": params['cfg_scale'],
        "width": params['width'],
        "height": params['height'],
        "restore_faces": params['restore_faces'],
        "override_settings_restore_afterwards": True,
        "negative_prompt": params['negative_prompt']
    }

    print(f'Prompting the image generator via the API on {params["address"]}...')
    response = requests.post(url=f'{params["address"]}/sdapi/v1/txt2img', json=payload)
    response.raise_for_status()
    r = response.json()

    visible_result = ""
    for img_str in r['images']:
        if params['save_img']:
            img_data = base64.b64decode(img_str)

            variadic = f'{date.today().strftime("%Y_%m_%d")}/{character}_{int(time.time())}'
            output_file = Path(f'extensions/sd_api_pictures/outputs/{variadic}.png')
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file.as_posix(), 'wb') as f:
                f.write(img_data)

            visible_result = visible_result + f'<img src="/file/extensions/sd_api_pictures/outputs/{variadic}.png" alt="{description}" style="max-width: unset; max-height: unset;">\n'
        else:
            image = Image.open(io.BytesIO(base64.b64decode(img_str.split(",", 1)[0])))
            # lower the resolution of received images for the chat, otherwise the log size gets out of control quickly with all the base64 values in visible history
            image.thumbnail((300, 300))
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            buffered.seek(0)
            image_bytes = buffered.getvalue()
            img_str = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode()
            visible_result = visible_result + f'<img src="{img_str}" alt="{description}">\n'

    if params['manage_VRAM']:
        give_VRAM_priority('LLM')

    return visible_result

# TODO: how do I make the UI history ignore the resulting pictures (I don't want HTML to appear in history)
# and replace it with 'text' for the purposes of logging?
def output_modifier(string, state):
    """
    This function is applied to the model outputs.
    """

    global picture_response, params

    if not picture_response:
        return string

    string = remove_surrounded_chars(string)
    string = string.replace('"', '')
    string = string.replace('“', '')
    string = string.replace('\n', ' ')
    string = string.strip()

    if string == '':
        string = 'no viable description in reply, try regenerating'
        return string

    text = ""
    if (params['mode'] < 2):
        toggle_generation(False)
        text = f'*Sends a picture which portrays: “{string}”*'
    else:
        text = string

    string = get_SD_pictures(string, state['character_menu']) + "\n" + text

    return string


def bot_prefix_modifier(string):
    """
    This function is only applied in chat mode. It modifies
    the prefix text for the Bot and can be used to bias its
    behavior.
    """

    return string


def toggle_generation(*args):
    global picture_response, shared

    if not args:
        picture_response = not picture_response
    else:
        picture_response = args[0]

    shared.processing_message = "*Is sending a picture...*" if picture_response else "*Is typing...*"


def filter_address(address):
    address = address.strip()
    # address = re.sub('http(s)?:\/\/|\/$','',address) # remove starting http:// OR https:// OR trailing slash
    address = re.sub('\/$', '', address)  # remove trailing /s
    if not address.startswith('http'):
        address = 'http://' + address
    return address


def SD_api_address_update(address):
    global params

    msg = "✔️ SD API is found on:"
    address = filter_address(address)
    params.update({"address": address})
    try:
        response = requests.get(url=f'{params["address"]}/sdapi/v1/sd-models')
        response.raise_for_status()
        # r = response.json()
    except:
        msg = "❌ No SD API endpoint on:"

    return gr.Textbox.update(label=msg)


def custom_css():
    path_to_css = Path(__file__).parent.resolve() / 'style.css'
    return open(path_to_css, 'r').read()


def get_checkpoints():
    global params

    try:
        models = requests.get(url=f'{params["address"]}/sdapi/v1/sd-models')
        options = requests.get(url=f'{params["address"]}/sdapi/v1/options')
        options_json = options.json()
        params['sd_checkpoint'] = options_json['sd_model_checkpoint']
        params['checkpoint_list'] = [result["title"] for result in models.json()]
    except:
        params['sd_checkpoint'] = ""
        params['checkpoint_list'] = []

    return gr.update(choices=params['checkpoint_list'], value=params['sd_checkpoint'])


def load_checkpoint(checkpoint):
    payload = {
        "sd_model_checkpoint": checkpoint
    }

    try:
        requests.post(url=f'{params["address"]}/sdapi/v1/options', json=payload)
    except:
        pass


def get_samplers():
    try:
        response = requests.get(url=f'{params["address"]}/sdapi/v1/samplers')
        response.raise_for_status()
        samplers = [x["name"] for x in response.json()]
    except:
        samplers = []

    return samplers


def ui():

    # Gradio elements
    # gr.Markdown('### Stable Diffusion API Pictures') # Currently the name of extension is shown as the title
    with gr.Accordion("Parameters", open=True, elem_classes="SDAP"):
        with gr.Row():
            address = gr.Textbox(placeholder=params['address'], value=params['address'], label='Auto1111\'s WebUI address')
            modes_list = ["Manual", "Immersive/Interactive", "Picturebook/Adventure"]
            mode = gr.Dropdown(modes_list, value=modes_list[params['mode']], label="Mode of operation", type="index")
            with gr.Column(scale=1, min_width=300):
                manage_VRAM = gr.Checkbox(value=params['manage_VRAM'], label='Manage VRAM')
                save_img = gr.Checkbox(value=params['save_img'], label='Keep original images and use them in chat')

            force_pic = gr.Button("Force the picture response")
            suppr_pic = gr.Button("Suppress the picture response")
        with gr.Row():
            checkpoint = gr.Dropdown(params['checkpoint_list'], value=params['sd_checkpoint'], label="Checkpoint", type="value")
            update_checkpoints = gr.Button("Get list of checkpoints")

        with gr.Accordion("Generation parameters", open=False):
            prompt_prefix = gr.Textbox(placeholder=params['prompt_prefix'], value=params['prompt_prefix'], label='Prompt Prefix (best used to describe the look of the character)')
            textgen_prefix = gr.Textbox(placeholder=params['textgen_prefix'], value=params['textgen_prefix'], label='textgen prefix (type [subject] where the subject should be placed)')
            negative_prompt = gr.Textbox(placeholder=params['negative_prompt'], value=params['negative_prompt'], label='Negative Prompt')
            with gr.Row():
                with gr.Column():
                    width = gr.Slider(64, 2048, value=params['width'], step=64, label='Width')
                    height = gr.Slider(64, 2048, value=params['height'], step=64, label='Height')
                with gr.Column(variant="compact", elem_id="sampler_col"):
                    with gr.Row(elem_id="sampler_row"):
                        sampler_name = gr.Dropdown(value=params['sampler_name'], allow_custom_value=True, label='Sampling method', elem_id="sampler_box")
                        create_refresh_button(sampler_name, lambda: None, lambda: {'choices': get_samplers()}, 'refresh-button')
                    steps = gr.Slider(1, 150, value=params['steps'], step=1, label="Sampling steps", elem_id="steps_box")
            with gr.Row():
                seed = gr.Number(label="Seed", value=params['seed'], elem_id="seed_box")
                cfg_scale = gr.Number(label="CFG Scale", value=params['cfg_scale'], elem_id="cfg_box")
                with gr.Column() as hr_options:
                    restore_faces = gr.Checkbox(value=params['restore_faces'], label='Restore faces')
                    enable_hr = gr.Checkbox(value=params['enable_hr'], label='Hires. fix')
            with gr.Row(visible=params['enable_hr'], elem_classes="hires_opts") as hr_options:
                hr_scale = gr.Slider(1, 4, value=params['hr_scale'], step=0.1, label='Upscale by')
                denoising_strength = gr.Slider(0, 1, value=params['denoising_strength'], step=0.01, label='Denoising strength')
                hr_upscaler = gr.Textbox(placeholder=params['hr_upscaler'], value=params['hr_upscaler'], label='Upscaler')

    # Event functions to update the parameters in the backend
    address.change(lambda x: params.update({"address": filter_address(x)}), address, None)
    mode.select(lambda x: params.update({"mode": x}), mode, None)
    mode.select(lambda x: toggle_generation(x > 1), inputs=mode, outputs=None)
    manage_VRAM.change(lambda x: params.update({"manage_VRAM": x}), manage_VRAM, None)
    manage_VRAM.change(lambda x: give_VRAM_priority('set' if x else 'reset'), inputs=manage_VRAM, outputs=None)
    save_img.change(lambda x: params.update({"save_img": x}), save_img, None)

    address.submit(fn=SD_api_address_update, inputs=address, outputs=address)
    prompt_prefix.change(lambda x: params.update({"prompt_prefix": x}), prompt_prefix, None)
    textgen_prefix.change(lambda x: params.update({"textgen_prefix": x}), textgen_prefix, None)
    negative_prompt.change(lambda x: params.update({"negative_prompt": x}), negative_prompt, None)
    width.change(lambda x: params.update({"width": x}), width, None)
    height.change(lambda x: params.update({"height": x}), height, None)
    hr_scale.change(lambda x: params.update({"hr_scale": x}), hr_scale, None)
    denoising_strength.change(lambda x: params.update({"denoising_strength": x}), denoising_strength, None)
    restore_faces.change(lambda x: params.update({"restore_faces": x}), restore_faces, None)
    hr_upscaler.change(lambda x: params.update({"hr_upscaler": x}), hr_upscaler, None)
    enable_hr.change(lambda x: params.update({"enable_hr": x}), enable_hr, None)
    enable_hr.change(lambda x: hr_options.update(visible=params["enable_hr"]), enable_hr, hr_options)
    update_checkpoints.click(get_checkpoints, None, checkpoint)
    checkpoint.change(lambda x: params.update({"sd_checkpoint": x}), checkpoint, None)
    checkpoint.change(load_checkpoint, checkpoint, None)

    sampler_name.change(lambda x: params.update({"sampler_name": x}), sampler_name, None)
    steps.change(lambda x: params.update({"steps": x}), steps, None)
    seed.change(lambda x: params.update({"seed": x}), seed, None)
    cfg_scale.change(lambda x: params.update({"cfg_scale": x}), cfg_scale, None)

    force_pic.click(lambda x: toggle_generation(True), inputs=force_pic, outputs=None)
    suppr_pic.click(lambda x: toggle_generation(False), inputs=suppr_pic, outputs=None)
