import base64
import io
import re
import time
from datetime import date
from pathlib import Path

import gradio as gr
import modules.shared as shared
import requests
import torch
import json
import yaml
from modules.models import reload_model, unload_model
from PIL import Image

torch._C._jit_set_profiling_mode(False)

# parameters which can be customized in settings.json of webui
params = {
    'address': 'http://127.0.0.1:7860',
    'mode': 0,  # modes of operation: 0 (Manual only), 1 (Immersive/Interactive - looks for words to trigger), 2 (Picturebook Adventure - Always on)
    'manage_VRAM': False,
    'save_img': False,
    'SD_model': 'NeverEndingDream',  # not used right now
    'prompt_prefix': '(Masterpiece:1.1), detailed, intricate',
    'negative_prompt': '(worst quality, low quality:1.3)',
    'width': 512,
    'height': 512,
    'seed': -1,
    'sampler_name': 'DDIM',
    'steps': 32,
    'cfg_scale': 7,
    'characterfocus': False,
    'secondary_prompt': False,
    'translations': False,
    'character': "",
    'secondary_negative_prompt': '',
    'secondary_positive_prompt': '',
    'prompt_translation_positive': "",
    'prompt_translation_negative': "",
    'positive_suffix': "",
    'negative_suffix': ""

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

samplers = ['DDIM', 'DPM++ 2M Karras']  # TODO: get the availible samplers with http://{address}}/sdapi/v1/samplers
SD_models = ['NeverEndingDream']  # TODO: get with http://{address}}/sdapi/v1/sd-models and allow user to select
initial_string = ""

streaming_state = shared.args.no_stream  # remember if chat streaming was enabled
picture_response = False  # specifies if the next model response should appear as a picture


def add_translations(description,triggered_array):
    global params
    tpatterns = json.loads(open(Path(f'extensions/sd_api_pictures/translations.json'), 'r', encoding='utf-8').read())
    i = 0
    for word_pair in tpatterns['pairs']:
        if triggered_array[i] != 1:
            if any(target in description for target in word_pair['descriptive_word']):
                params['positive_suffix'] = params['positive_suffix'] + ", " + word_pair['SD_positive_translation']
                params['negative_suffix'] = params['negative_suffix'] + ", " + word_pair['SD_negative_translation']
                triggered_array[i] = 1
        i = i + 1
    return triggered_array

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

def request_generation(case,string):
    if case == 1: # Character has been asked to send a picture of itself
        toggle_generation(True)
        params['characterfocus'] = True
        string = string.replace("yourself","you")
        after_you = string.split("you", 1)[1] # subdivide the string once by the first 'you' instance and get what's coming after it
        if after_you != '':
            string = "Describe what you are currently wearing, your environment and yourself performing the following action: " + after_you.strip()
        else:
            string = "Describe what you are currently wearing, your environment and yourself."
    elif case == 2: # Character has been asked to send a picture of something else
        toggle_generation(True)
        subject = string.split('of', 1)[1]  # subdivide the string once by the first 'of' instance and get what's coming after it
        string = "Please provide a detailed and vivid description of " + subject.strip()
    elif case == 3: # Character has been asked to send a picture of nothing in particular
        toggle_generation(True)
        string = "Please provide a detailed description of your appearance, your surroundings and what you are doing right now."
    return string

def string_evaluation(string):
    orig_string = string
    input_type = 0
    subjects = ['yourself', 'you']
    params['characterfocus'] = False
    params['prompt_translation_positive'] = ""
    params['prompt_translation_negative'] = ""
    if triggers_are_in(string):  # check for trigger words for generation
        string = string.lower()
        if "of" in string:
            if any(target in string for target in subjects): # the focus of the image should be on the sending character
                input_type = 1 
            else:
                input_type = 2
        else:
            input_type = 3
    return request_generation(input_type,string)

def input_modifier(string):
    """
    This function is applied to your text inputs before
    they are fed into the model.
    """

    global params, initial_string
    initial_string = string
    if params['mode'] == 1:  # For immersive/interactive mode, send to string evaluation
            return string_evaluation(string)
    if params['mode'] == 2:
        params['characterfocus'] = False
        string = string.lower()
        return string
    if params['mode'] == 0:
        return string

# Add NSFW tags if NSFW is enabled, add character sheet tags if character is describing itself
def create_suffix():
    global params
    params['positive_suffix'] = ""
    params['negative_suffix'] = ""

    # load character data from json, yaml, or yml file
    if shared.character != 'None':
        found_file = False
        folder1 = 'characters'
        folder2 = 'characters/instruction-following'
        for folder in [folder1, folder2]:
            for extension in ["yml", "yaml", "json"]:
                filepath = Path(f'{folder}/{shared.character}.{extension}')
                if filepath.exists():
                    found_file = True
                    break
            if found_file:
                break
        file_contents = open(filepath, 'r', encoding='utf-8').read()
        data = json.loads(file_contents) if extension == "json" else yaml.safe_load(file_contents)

    if params['secondary_prompt']:
        params['positive_suffix'] = params['secondary_positive_prompt'] + ", " + params['prompt_translation_positive']
        params['negative_suffix'] = params['secondary_negative_prompt'] + ", " + params['prompt_translation_negative']
    if params['characterfocus'] and shared.character != 'None':
        params['positive_suffix'] = data['sd_tags_positive'] + ", " + params['prompt_translation_positive'] if 'sd_tags_positive' in data else params['prompt_translation_positive']
        params['negative_suffix'] = data['sd_tags_negative'] + ", " + params['prompt_translation_negative'] if 'sd_tags_negative' in data else params['prompt_translation_negative']
        if params['secondary_prompt']:
            params['positive_suffix'] = params['secondary_positive_prompt'] + ", " + data['sd_tags_positive'] + ", " + params['prompt_translation_positive'] if 'sd_tags_positive' in data else params['secondary_positive_prompt'] + ", " + params['prompt_translation_positive']
            params['negative_suffix'] = params['secondary_negative_prompt'] + ", " + data['sd_tags_negative'] + ", " + params['prompt_translation_negative'] if 'sd_tags_negative' in data else params['secondary_negative_prompt'] + ", " + params['prompt_translation_negative']


# Get and save the Stable Diffusion-generated picture
def get_SD_pictures(description):

    global params, initial_string

    if params['manage_VRAM']:
        give_VRAM_priority('SD')

    create_suffix()
    if params['translations']:
        tpatterns = json.loads(open(Path(f'extensions/sd_api_pictures/translations.json'), 'r', encoding='utf-8').read())
        triggered_array = [0] * len(tpatterns['pairs'])
        triggered_array = add_translations(initial_string,triggered_array)
        add_translations(description,triggered_array)

    payload = {
        "prompt": params['prompt_prefix'] + ", " + description + ", " + params['positive_suffix'],
        "seed": params['seed'],
        "sampler_name": params['sampler_name'],
        "steps": params['steps'],
        "cfg_scale": params['cfg_scale'],
        "width": params['width'],
        "height": params['height'],
        "restore_faces": params['restore_faces'],
        "override_settings_restore_afterwards": True,
        "negative_prompt": params['negative_prompt'] + ", " + params['negative_suffix']
    }

    print(f'Prompting the image generator via the API on {params["address"]}...')
    response = requests.post(url=f'{params["address"]}/sdapi/v1/txt2img', json=payload)
    response.raise_for_status()
    r = response.json()

    visible_result = ""
    for img_str in r['images']:
        if params['save_img']:
            img_data = base64.b64decode(img_str)

            variadic = f'{date.today().strftime("%Y_%m_%d")}/{shared.character}_{int(time.time())}'
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
def output_modifier(string):
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

    shared.args.no_stream = True if picture_response else streaming_state  # Disable streaming cause otherwise the SD-generated picture would return as a dud
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

def ui():

    # Gradio elements
    # gr.Markdown('### Stable Diffusion API Pictures') # Currently the name of extension is shown as the title
    with gr.Accordion("Parameters", open=True):
        with gr.Row():
            address = gr.Textbox(placeholder=params['address'], value=params['address'], label='Auto1111\'s WebUI address')
            mode = gr.Dropdown(["Manual", "Immersive/Interactive", "Picturebook/Adventure"], value="Manual", label="Mode of operation", type="index")
            with gr.Column(scale=1, min_width=300):
                manage_VRAM = gr.Checkbox(value=params['manage_VRAM'], label='Manage VRAM')
                save_img = gr.Checkbox(value=params['save_img'], label='Keep original images and use them in chat')
                secondary_prompt = gr.Checkbox(value=params['secondary_prompt'], label='Add secondary tags in prompt')
                translations = gr.Checkbox(value=params['translations'], label='Activate SD translations')

            force_pic = gr.Button("Force the picture response")
            suppr_pic = gr.Button("Suppress the picture response")

        with gr.Accordion("Generation parameters", open=False):
            prompt_prefix = gr.Textbox(placeholder=params['prompt_prefix'], value=params['prompt_prefix'], label='Prompt Prefix (best used to describe the look of the character)')
            with gr.Row():
                with gr.Column():
                    negative_prompt = gr.Textbox(placeholder=params['negative_prompt'], value=params['negative_prompt'], label='Negative Prompt')
                    sampler_name = gr.Textbox(placeholder=params['sampler_name'], value=params['sampler_name'], label='Sampler')
                with gr.Column():
                    width = gr.Slider(256, 768, value=params['width'], step=64, label='Width')
                    height = gr.Slider(256, 768, value=params['height'], step=64, label='Height')
            with gr.Row():
                with gr.Column():
                    secondary_positive_prompt = gr.Textbox(placeholder=params['secondary_positive_prompt'], value=params['secondary_positive_prompt'], label='Secondary positive prompt')
                with gr.Column():
                    secondary_negative_prompt = gr.Textbox(placeholder=params['secondary_negative_prompt'], value=params['secondary_negative_prompt'], label='Secondary negative prompt')
            with gr.Row():
                steps = gr.Number(label="Steps:", value=params['steps'])
                seed = gr.Number(label="Seed:", value=params['seed'])
                cfg_scale = gr.Number(label="CFG Scale:", value=params['cfg_scale'])

    # Event functions to update the parameters in the backend
    address.change(lambda x: params.update({"address": filter_address(x)}), address, None)
    mode.select(lambda x: params.update({"mode": x}), mode, None)
    mode.select(lambda x: toggle_generation(x > 1), inputs=mode, outputs=None)
    manage_VRAM.change(lambda x: params.update({"manage_VRAM": x}), manage_VRAM, None)
    manage_VRAM.change(lambda x: give_VRAM_priority('set' if x else 'reset'), inputs=manage_VRAM, outputs=None)
    save_img.change(lambda x: params.update({"save_img": x}), save_img, None)

    address.submit(fn=SD_api_address_update, inputs=address, outputs=address)
    negative_prompt.change(lambda x: params.update({"negative_prompt": x}), negative_prompt, None)
    translations.change(lambda x: params.update({"translations": x}), translations, None)
    prompt_prefix.change(lambda x: params.update({"prompt_prefix": x}), prompt_prefix, None)
    negative_prompt.change(lambda x: params.update({"negative_prompt": x}), negative_prompt, None)
    secondary_positive_prompt.change(lambda x: params.update({"secondary_positive_prompt": x}), secondary_positive_prompt, None)
    secondary_negative_prompt.change(lambda x: params.update({"secondary_negative_prompt": x}), secondary_negative_prompt, None)
    width.change(lambda x: params.update({"width": x}), width, None)
    height.change(lambda x: params.update({"height": x}), height, None)

    sampler_name.change(lambda x: params.update({"sampler_name": x}), sampler_name, None)
    steps.change(lambda x: params.update({"steps": x}), steps, None)
    seed.change(lambda x: params.update({"seed": x}), seed, None)
    cfg_scale.change(lambda x: params.update({"cfg_scale": x}), cfg_scale, None)

    force_pic.click(lambda x: toggle_generation(True), inputs=force_pic, outputs=None)
    suppr_pic.click(lambda x: toggle_generation(False), inputs=suppr_pic, outputs=None)
