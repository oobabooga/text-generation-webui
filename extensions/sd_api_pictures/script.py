import base64
import io
import re
from pathlib import Path
from datetime import date
import time

import gradio as gr
import modules.shared as shared
import modules.chat as chat
import requests
import torch
from modules.models import reload_model, unload_model
from PIL import Image

torch._C._jit_set_profiling_mode(False)

# parameters which can be customized in settings.json of webui
params = {
    "address": "http://127.0.0.1:7860",
    "mode": 0,  # modes of operation: 0 (Manual only), 1 (Immersive/Interactive - looks for words to trigger), 2 (Picturebook Adventure - Always on)
    "manage_VRAM": False,
    "save_img": False,
    "sd_model_chekpoint": "dreamlikeDiffusion10_10.ckpt [0aecbcfa2c]",
    "prompt_prefix": "(Masterpiece:1.1), detailed, intricate, colorful",
    "negative_prompt": "(worst quality, low quality:1.3)",
    "enable_hr": False,
    "denoising_strength": 0.36,
    "hr_scale": 2,
    "hr_upscaler": "4x_NMKD-Superscale-SP_178000_G",
    "hr_second_pass_steps": 0,
    "seed": -1,
    "batch_size": 1,
    "steps": 36,
    "cfg_scale": 7,
    "width": 512,
    "height": 768,
    "restore_faces": False,
    "sampler_index": "DPM++ SDE Karras"  
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


#list all upscalers to pass them on the buttom
res_upscalers = requests.get(url=f'{params["address"]}/sdapi/v1/upscalers')
upscalers = [upscaler['name'] for upscaler in res_upscalers.json()]
#list samplers
res_samplers = requests.get(url=f'{params["address"]}/sdapi/v1/samplers')
samplers = [sampler['name'] for sampler in res_samplers.json()]
#format the name of the checkpoint to pass on the options
def format_model_name_and_hash(item):
    return f"{item['model_name']} {item['hash']}"
#get the models 
res_models = requests.get(url=f'{params["address"]}/sdapi/v1/sd-models')
sd_models = [format_model_name_and_hash(model) for model in res_models.json()]

streaming_state = shared.args.no_stream  # remember if chat streaming was enabled
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
            string = "Please provide a detailed and vivid description of " + subject
        else:
            string = "Please provide a detailed description of your appearance, your surroundings and what you are doing right now"

    return string

# Get and save the Stable Diffusion-generated picture
def get_SD_pictures(description):
   
    global params

    if params['manage_VRAM']:
        give_VRAM_priority('SD')

    payload_options = {
        "sd_model_checkpoint": params['sd_model_checkpoint']
        #"sd_vae": params['sd_vae']
    }


    print(f'Updating parameters via the API on {params["address"]}...')
    response = requests.post(url=f'{params["address"]}/sdapi/v1/options', json=payload_options)
    response.raise_for_status()
    r = response.json()

    payload = {

        "prompt": params['prompt_prefix'] + description,
        "negative_prompt": params['negative_prompt'],
        "seed": params['seed'],
        "sampler_index": params['sampler_index'],
        "steps": params['steps'], 
        "cfg_scale": params['cfg_scale'],
        "width": params['width'],
        "height": params['height'],
        "restore_faces": params['restore_faces'],
        "enable_hr": params['enable_hr'],
        "hr_upscaler": params['hr_upscaler'],
        "hr_scale": params['hr_scale'],
        "restore_faces": params['restore_faces'],
        "batch_size": params['batch_size'],
        "denoising_strength": params['denoising_strength'],
        "hr_second_pass_steps": params['hr_second_pass_steps']
                 
    }

    print(f'Prompting the image generator via the API on {params["address"]}...')
    response = requests.post(url=f'{params["address"]}/sdapi/v1/txt2img', json=payload)
    response.raise_for_status()
    r = response.json()


    visible_result = ""
    for img_str in r['images']:
        image = Image.open(io.BytesIO(base64.b64decode(img_str.split(",", 1)[0])))
        if params['save_img']:
            variadic = f'{date.today().strftime("%Y_%m_%d")}/{shared.character}_{int(time.time())}'
            output_file = Path(f'extensions/sd_api_pictures/outputs/{variadic}.png')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_file.as_posix())
            visible_result = visible_result + f'<img src="/file/extensions/sd_api_pictures/outputs/{variadic}.png" alt="{description}" style="max-width: unset; max-height: unset;">\n'
        else:
            # lower the resolution of received images for the chat, otherwise the log size gets out of control quickly with all the base64 values in visible history
            image.thumbnail((400, 400))
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

        with gr.Row():
            generate_now_btn = gr.Button("Generate from input")
        with gr.Row():    
            force_pic = gr.Button("Force the picture response")
            suppr_pic = gr.Button("Suppress the picture response")


        with gr.Accordion("Generation parameters", open=False):
            with gr.Row():
                with gr.Column():
                    width = gr.Slider(256, 1024, value=params['width'], step=64, label='Image width')
                    height = gr.Slider(256, 1024, value=params['height'], step=64, label='Image height')
                    seed = gr.Number(label="Seed:", value=params['seed'])
                    cfg_scale = gr.Number(label="CFG Scale:", value=params['cfg_scale'])
                    steps = gr.Number(label="Steps:", value=params['steps'])
                    batch_size = gr.Slider(1, 6, value=params['batch_size'], step=1, label='batch size')

                with gr.Column():
                    restore_faces = gr.Checkbox(value=params['restore_faces'], label='Restore faces')
                    enable_hr = gr.Checkbox(value=params['enable_hr'], label='Enable High Resolution')
                    hr_upscaler = gr.Dropdown(label="Upscaler", choices=upscalers)
                    denoising_strength = gr.Slider(0, 1, value=params['denoising_strength'], step=0.01, label='Denoising strength')
                    hr_second_pass_steps = gr.Slider(0, 30, value=params['hr_second_pass_steps'], step=1, label='HR second pass steps')
                    hr_scale = gr.Slider(0, 4, value=params['hr_scale'], step=0.5, label='HR scale')
                
            with gr.Row():
                sd_model_checkpoint = gr.Dropdown(label="model", choices=sd_models)
                sampler_index = gr.Dropdown(label="sampler", choices=samplers)
                  
            with gr.Accordion("Prompt settigns", open=False):
                prompt_prefix = gr.Textbox(placeholder=params['prompt_prefix'], value=params['prompt_prefix'], label='Prompt Prefix (best used to describe the look of the character)')
                with gr.Row():
                    negative_prompt = gr.Textbox(placeholder=params['negative_prompt'], value=params['negative_prompt'], label='Negative Prompt')
             
    # Event functions to update the parameters in the backend
    address.change(lambda x: params.update({"address": filter_address(x)}), address, None)
    mode.select(lambda x: params.update({"mode": x}), mode, None)
    mode.select(lambda x: toggle_generation(x > 1), inputs=mode, outputs=None)
    manage_VRAM.change(lambda x: params.update({"manage_VRAM": x}), manage_VRAM, None)
    manage_VRAM.change(lambda x: give_VRAM_priority('set' if x else 'reset'), inputs=manage_VRAM, outputs=None)
    save_img.change(lambda x: params.update({"save_img": x}), save_img, None)
    sd_model_checkpoint.change(lambda x: params.update({"sd_model_checkpoint": x}), sd_model_checkpoint, None)
    address.submit(fn=SD_api_address_update, inputs=address, outputs=address)
    prompt_prefix.change(lambda x: params.update({"prompt_prefix": x}), prompt_prefix, None)
    negative_prompt.change(lambda x: params.update({"negative_prompt": x}), negative_prompt, None)
    width.change(lambda x: params.update({"width": x}), width, None)
    height.change(lambda x: params.update({"height": x}), height, None)    
    batch_size.change(lambda x: params.update({"batch_size": x}), batch_size, None)
    hr_upscaler.change(lambda x: params.update({"hr_upscaler": x}), hr_upscaler, None)
    restore_faces.change(lambda x: params.update({"restore_faces": x}), restore_faces, None)
    enable_hr.change(lambda x: params.update({"enable_hr": x}), enable_hr, None)
    denoising_strength.change(lambda x: params.update({"denoising_strength": x}), denoising_strength, None)
    hr_second_pass_steps.change(lambda x: params.update({"hr_second_pass_steps": x}), hr_second_pass_steps, None)
    hr_scale.change(lambda x: params.update({"hr_scale": x}), hr_scale, None)
    sampler_index.change(lambda x: params.update({"sampler_index": x}), sampler_index, None)
    steps.change(lambda x: params.update({"steps": x}), steps, None)
    seed.change(lambda x: params.update({"seed": x}), seed, None)
    cfg_scale.change(lambda x: params.update({"cfg_scale": x}), cfg_scale, None)

    #update the buttoms
    force_pic.click(lambda x: toggle_generation(True), inputs=force_pic, outputs=None)
    suppr_pic.click(lambda x: toggle_generation(False), inputs=suppr_pic, outputs=None)
    generate_now_btn.click(lambda x: toggle_generation(True), inputs=generate_now_btn, outputs=None)    
    generate_now_btn.click(eval('chat.chatbot_wrapper'), shared.input_params, shared.gradio['display'], show_progress=shared.args.no_stream)


      
    
