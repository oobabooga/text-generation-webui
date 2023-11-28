import html
import json
import random
import time
import subprocess
import os
from pathlib import Path

import gradio as gr

from modules import chat, shared, ui_chat
from modules.logging_colors import logger
from modules.ui import create_refresh_button
from modules.utils import gradio

import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


try:
    from TTS.api import TTS
    from TTS.utils.synthesizer import Synthesizer
except ModuleNotFoundError:
    logger.error(
        "[COQUI TTS] Could not find the TTS module. Make sure to install the requirements for the coqui_tts extension."
        "\n"
        "\nLinux / Mac:\npip install -r extensions/coqui_tts/requirements.txt\n"
        "\nWindows:\npip install -r extensions\\coqui_tts\\requirements.txt\n"
        "\n"
        "[COQUI TTS] If you used the one-click installer, paste the command above in the terminal window launched after running the \"cmd_\" script. On Windows, that's \"cmd_windows.bat\"."
    )
    raise

try:
    import deepspeed
except:
    deepspeed_installed = False
    print("[COQUI TTS] DEEPSPEED: Not Detected. See https://github.com/microsoft/DeepSpeed") 
else: 
    deepspeed_installed = True
    print("[COQUI TTS] DEEPSPEED: Detected")
    print("[COQUI TTS] DEEPSPEED: Activate in Coqui settings")

params = {
    "activate": True,
    "autoplay": True,
    "show_text": True,
    "low_vram": False,
    "remove_trailing_dots": False,
    "voice": "female_01.wav",
    "language": "English",
    "model_name": "tts_models/multilingual/multi-dataset/xtts_v2", #This is the version called through the TTS.api
    "model_version": "xttsv2_2.0.2", #This is the model that is downloaded into your /coqui_tts/models/
    "deepspeed_activate": False,
    # Set the different methods for Generation of TTS
    "tts_method_api_tts": False,
    "tts_method_api_local": False,
    "tts_method_xtts_local": True,
    "model_loaded": False
}

#Clear model
model = None

# Device Setup
device = "cuda" if torch.cuda.is_available() else "cpu"

this_dir = Path(__file__).parent.resolve()

with open(this_dir / 'languages.json', encoding='utf8') as f:
    languages = json.load(f)

# Check for model having been downloaded
def check_required_files():
    this_dir = Path(__file__).parent.resolve()
    download_script_path = this_dir / 'modeldownload.py'
    subprocess.run(['python', str(download_script_path)])
    print("[COQUI TTS] STARTUP: All required files are present.")

# Call the function when your main script loads
check_required_files()

# Pick model loader depending on Params
def setup():
    global model
    generate_start_time = time.time()  # Record the start time of loading the model
    if params["tts_method_api_tts"]:
        print(f"[COQUI TTS] MODEL: \033[94mAPI TTS Loading\033[0m {params['model_name']} into\033[93m", device, "\033[0m")
        model = api_load_model()
    elif params["tts_method_api_local"]:
        print(f"[COQUI TTS] MODEL: \033[94mAPI Local Loading\033[0m {params['model_version']} into\033[93m", device, "\033[0m")
        model = api_manual_load_model()
    elif params["tts_method_xtts_local"]:
        print(f"[COQUI TTS] MODEL: \033[94mXTTSv2 Local Loading\033[0m {params['model_version']} into\033[93m", device, "\033[0m")
        model = xtts_manual_load_model()
        
    generate_end_time = time.time()  # Record the end time of loading the model
    generate_elapsed_time = generate_end_time - generate_start_time
    params["model_loaded"] = True
    print(f"[COQUI TTS] MODEL: \033[94mModel Loaded in \033[0m{generate_elapsed_time:.2f} seconds.")
    Path(f"{this_dir}/outputs").mkdir(parents=True, exist_ok=True)

#Model Loaders
def api_load_model():
    model = TTS(params["model_name"]).to(device)
    return model
 
def api_manual_load_model():
    model = TTS(model_path=this_dir / 'models' / params['model_version'],config_path=this_dir / 'models' / params['model_version'] / 'config.json').to(device)
    return model
    
def xtts_manual_load_model():
    config = XttsConfig()
    config_path = this_dir / 'models' / params['model_version'] / 'config.json'
    checkpoint_dir = this_dir / 'models' / params['model_version']
    config.load_json(str(config_path))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=str(checkpoint_dir), use_deepspeed=params['deepspeed_activate'])
    model.cuda()
    model.to(device)
    return model

#Unload or clear the model, and return None
def unload_model(model):
    del model
    params["model_loaded"] = False
    return None

#Move model between VRAM and RAM if Low VRAM set.
def switch_device():
    global model, device
    if not params["low_vram"]:
        return
    if device == "cuda":
        device = "cpu"
        model.to(device) 
        torch.cuda.empty_cache()
    else:    
        device == "cpu"
        device = "cuda"
        model.to(device)

#Display license information
print("[COQUI TTS] LICENSE: \033[94mCoqui Public Model License\033[0m")
print("[COQUI TTS] LICENSE: \033[94mhttps://coqui.ai/cpml.txt\033[0m")

def get_available_voices():
    return sorted([voice.name for voice in Path(f"{this_dir}/voices").glob("*.wav")])

def preprocess(raw_input):
    raw_input = html.unescape(raw_input)
    # raw_input = raw_input.strip("\"")
    return raw_input

def new_split_into_sentences(self, text):
    sentences = self.seg.segment(text)
    if params['remove_trailing_dots']:
        sentences_without_dots = []
        for sentence in sentences:
            if sentence.endswith('.') and not sentence.endswith('...'):
                sentence = sentence[:-1]

            sentences_without_dots.append(sentence)

        return sentences_without_dots
    else:
        return sentences

Synthesizer.split_into_sentences = new_split_into_sentences


def remove_tts_from_history(history):
    for i, entry in enumerate(history['internal']):
        history['visible'][i] = [history['visible'][i][0], entry[1]]

    return history

def toggle_text_in_history(history):
    for i, entry in enumerate(history['visible']):
        visible_reply = entry[1]
        if visible_reply.startswith('<audio'):
            if params['show_text']:
                reply = history['internal'][i][1]
                history['visible'][i] = [history['visible'][i][0], f"{visible_reply.split('</audio>')[0]}</audio>\n\n{reply}"]
            else:
                history['visible'][i] = [history['visible'][i][0], f"{visible_reply.split('</audio>')[0]}</audio>"]
    return history


def random_sentence():
    with open(Path("extensions/coqui_tts/harvard_sentences.txt")) as f:
        return random.choice(list(f))


#Preview Voice Generation Function
def voice_preview(string):
    #Check model is loaded before continuing
    if not params["model_loaded"]:
        print("[COQUI TTS] \033[91mWARNING\033[0m Model is still loading, please wait before trying to generate TTS")
        return
    string = html.unescape(string) or random_sentence()
    # Replace double quotes with single, asterisks, carriage returns, and line feeds
    string = string.replace('"', "'").replace(".'", "'.").replace('*', '').replace('\r', '').replace('\n', '')
    output_file = Path('extensions/coqui_tts/outputs/voice_preview.wav')
    if params["low_vram"] and device == "cpu":
        switch_device()
        print("[COQUI TTS] LOW VRAM: Moving model to:\033[93m", device, "\033[0m")
  
    #XTTSv2 LOCAL Method
    if params["tts_method_xtts_local"]:     
        generate_start_time = time.time()  # Record the start time of generating TTS
        print("[COQUI TTS] GENERATING TTS: {}".format(string))
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[f"{this_dir}/voices/{params['voice']}"])
        out = model.inference(
        string, 
        languages[params["language"]],
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=0.7
        )
        torchaudio.save(output_file, torch.tensor(out["wav"]).unsqueeze(0), 24000)
        generate_end_time = time.time()  # Record the end time to generate TTS
        generate_elapsed_time = generate_end_time - generate_start_time
        print(f"[COQUI TTS] PROCESSING TIME: \033[91m{generate_elapsed_time:.2f}\033[0m seconds.")
    #API TTS and API LOCAL Methods
    elif params["tts_method_api_tts"] or params["tts_method_api_local"]:
        #Set the correct output path (different from the if statement)
        model.tts_to_file(
            text=string,
            file_path=output_file,
            speaker_wav=[f"{this_dir}/voices/{params['voice']}"],
            language=languages[params["language"]]
        )       
 
    if params["low_vram"] and device == "cuda":
        switch_device()
        print("[COQUI TTS] LOW VRAM: Moving model to:\033[93m", device, "\033[0m")
 
 
    return f'<audio src="file/{output_file.as_posix()}?{int(time.time())}" controls autoplay></audio>'


def history_modifier(history):
    # Remove autoplay from the last reply
    if len(history['internal']) > 0:
        history['visible'][-1] = [
            history['visible'][-1][0],
            history['visible'][-1][1].replace('controls autoplay>', 'controls>')
        ]

    return history


def state_modifier(state):
    if not params['activate']:
        return state

    state['stream'] = False
    return state


def input_modifier(string, state):
    if not params['activate']:
        return string

    shared.processing_message = "*Is recording a voice message...*"
    return string

#Standard Voice Generation Function
def output_modifier(string, state):
    if not params["model_loaded"]:
        print("[COQUI TTS] \033[91mWARNING\033[0m Model is still loading, please wait before trying to generate TTS")
        return
    if params["low_vram"] and device == "cpu":
        switch_device()
        print("[COQUI TTS] LOW VRAM: Moving model to:\033[93m", device, "\033[0m")
    if not params['activate']: 
        return string
   
    original_string = string
    string = preprocess(html.unescape(string))
    
    if string == '':
        return '*Empty string*'
        
    # Replace double quotes with single, asterisks, carriage returns, and line feeds
    string = string.replace('"', "'").replace(".'", "'.").replace('*', '').replace('\r', '').replace('\n', '')  
    output_file = Path(f'extensions/coqui_tts/outputs/{state["character_menu"]}_{int(time.time())}.wav')

    #XTTSv2 LOCAL Method
    if params["tts_method_xtts_local"]:     
        generate_start_time = time.time()  # Record the start time of generating TTS
        print("[COQUI TTS] GENERATING TTS: {}".format(string))
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[f"{this_dir}/voices/{params['voice']}"])
        out = model.inference(
        string, 
        languages[params["language"]],
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=0.7
        )
        torchaudio.save(output_file, torch.tensor(out["wav"]).unsqueeze(0), 24000)
        generate_end_time = time.time()  # Record the end time to generate TTS
        generate_elapsed_time = generate_end_time - generate_start_time
        print(f"[COQUI TTS] PROCESSING TIME: \033[91m{generate_elapsed_time:.2f}\033[0m seconds.") 

        autoplay = 'autoplay' if params['autoplay'] else ''
        string = f'<audio src="file/{output_file.as_posix()}" controls {autoplay}></audio>'
        
        if params['show_text']:
            string += f'\n\n{original_string}'
        
        shared.processing_message = "*Is typing...*"
    
    #API TTS and API LOCAL Methods    
    elif params["tts_method_api_tts"] or params["tts_method_api_local"]:
        #Set the correct output path (different from the if statement)
        model.tts_to_file(
            text=string,
            file_path=output_file,
            speaker_wav=[f"{this_dir}/voices/{params['voice']}"],
            language=languages[params["language"]]
        )       
        
        autoplay = 'autoplay' if params['autoplay'] else ''
        string = f'<audio src="file/{output_file.as_posix()}" controls {autoplay}></audio>'
    
      
        if params['show_text']:
            string += f'\n\n{original_string}'
        
        shared.processing_message = "*Is typing...*"
        
    if params["low_vram"] and device == "cuda":
        switch_device()
        print("[COQUI TTS] LOW VRAM: Moving model to:\033[93m", device, "\033[0m")

        
    return string

def custom_css():
    path_to_css = Path(f"{this_dir}/style.css")
    return open(path_to_css, 'r').read()

#Low VRAM Gradio Checkbox handling
def handle_low_vram(value):
   global model, device
   if value:
       model = unload_model(model)
       device = "cpu"
       print("[COQUI TTS] MODEL: \033[94mChanging model \033[92m(Please wait 15 seconds)\033[0m")
       print("[COQUI TTS] LOW VRAM: \033[94mEnabled.\033[0m Model will move between \033[93mVRAM(cuda) <> System RAM(cpu)\033[0m")
       setup() 
   else:
       model = unload_model(model)
       device = "cuda"
       print("[COQUI TTS] MODEL: \033[94mChanging model \033[92m(Please wait 15 seconds)\033[0m")
       print("[COQUI TTS] LOW VRAM: \033[94mDisabled.\033[0m Model will remain in \033[93mVRAM\033[0m")
       setup()

#Reload the model when DeepSpeed checkbox is enabled/disabled
def handle_deepspeed_activate_checkbox_change(value):
    global model
    
    if value:
        # DeepSpeed enabled
        print("[COQUI TTS] DEEPSPEED: \033[93mActivating)\033[0m")
        print("[COQUI TTS] MODEL: \033[94mChanging model \033[92m(Please wait 15 seconds)\033[0m")
        model = unload_model(model)
        params["tts_method_api_tts"] = False
        params["tts_method_api_local"] = False
        params["tts_method_xtts_local"] = True
        params["deepspeed_activate"] = True
        gr.update(tts_radio_buttons={"value": "XTTSv2 Local"})
        setup()
    else:
        # DeepSpeed disabled
        print("[COQUI TTS] DEEPSPEED: \033[93mDe-Activating)\033[0m")
        print("[COQUI TTS] MODEL: \033[94mChanging model \033[92m(Please wait 15 seconds)\033[0m")
        params["deepspeed_activate"] = False 
        model = unload_model(model)
        setup()

    return value # Return new checkbox value


# Allow DeepSpeed Checkbox to appear if tts_method_api_tts and deepspeed_installed are True
deepspeed_condition = params["tts_method_xtts_local"] == "True" and deepspeed_installed

def handle_tts_method_change(choice):
    # Update the params dictionary based on the selected radio button
    print("[COQUI TTS] MODEL: \033[94mChanging model \033[92m(Please wait 15 seconds)\033[0m")

    # Set other parameters to False
    if choice == "API TTS":
        params["tts_method_api_local"] = False
        params["tts_method_xtts_local"] = False
        params["tts_method_api_tts"] = True
        params["deepspeed_activate"] = False
        gr.update(deepspeed_checkbox={"value": False})
    elif choice == "API Local":
        params["tts_method_api_tts"] = False
        params["tts_method_xtts_local"] = False
        params["tts_method_api_local"] = True
        params["deepspeed_activate"] = False
        gr.update(deepspeed_checkbox={"value": False})
    elif choice == "XTTSv2 Local":
        params["tts_method_api_tts"] = False
        params["tts_method_api_local"] = False
        params["tts_method_xtts_local"] = True

    # Unload the current model
    global model
    model = unload_model(model)

    # Load the correct model based on the updated params
    setup()

def ui():
    with gr.Accordion("Coqui TTS (XTTSv2)"):
        with gr.Row():
            activate = gr.Checkbox(value=params['activate'], label='Activate TTS')
            autoplay = gr.Checkbox(value=params['autoplay'], label='Play TTS automatically')

        with gr.Row():
            show_text = gr.Checkbox(value=params['show_text'], label='Show message text under audio player')
            remove_trailing_dots = gr.Checkbox(value=params['remove_trailing_dots'], label='Remove trailing "." from text segments before generation')
                        
        with gr.Row():
            low_vram = gr.Checkbox(value=params['low_vram'], label='Low VRAM mode (Read NOTE)')
            deepspeed_checkbox = gr.Checkbox(value=params['deepspeed_activate'], label='Activate DeepSpeed (Read NOTE)', visible=deepspeed_installed)
        
        with gr.Row():
            tts_radio_buttons = gr.Radio(
            choices=["API TTS", "API Local", "XTTSv2 Local"],
            label="Select TTS Generation Method (Read NOTE)",
            value="XTTSv2 Local"  # Set the default value
            )

            explanation_text = gr.HTML("<p>NOTE: Switching Model Type, Low VRAM & DeepSpeed takes 15 seconds. Each TTS generation method has a slightly different sound. DeepSpeed checkbox is only visible if DeepSpeed is present on your system and it only uses XTTSv2 Local.</p>")
            
        with gr.Row():
            with gr.Row():
                voice = gr.Dropdown(get_available_voices(), label="Voice wav", value=params["voice"])
                create_refresh_button(voice, lambda: None, lambda: {'choices': get_available_voices(), 'value': params["voice"]}, 'refresh-button')

            language = gr.Dropdown(languages.keys(), label="Language", value=params["language"])

        with gr.Row():
            preview_text = gr.Text(show_label=False, placeholder="Preview text", elem_id="silero_preview_text")
            preview_play = gr.Button("Preview")
            preview_audio = gr.HTML(visible=False)

        with gr.Row():
            convert = gr.Button('Permanently replace audios with the message texts')
            convert_cancel = gr.Button('Cancel', visible=False)
            convert_confirm = gr.Button('Confirm (cannot be undone)', variant="stop", visible=False)

    # Convert history with confirmation
    convert_arr = [convert_confirm, convert, convert_cancel]
    convert.click(lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)], None, convert_arr)
    convert_confirm.click(
        lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)], None, convert_arr).then(
        remove_tts_from_history, gradio('history'), gradio('history')).then(
        chat.save_history, gradio('history', 'unique_id', 'character_menu', 'mode'), None).then(
        chat.redraw_html, gradio(ui_chat.reload_arr), gradio('display'))

    convert_cancel.click(lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)], None, convert_arr)

    # Toggle message text in history
    show_text.change(
        lambda x: params.update({"show_text": x}), show_text, None).then(
        toggle_text_in_history, gradio('history'), gradio('history')).then(
        chat.save_history, gradio('history', 'unique_id', 'character_menu', 'mode'), None).then(
        chat.redraw_html, gradio(ui_chat.reload_arr), gradio('display'))

    # Event functions to update the parameters in the backend
    activate.change(lambda x: params.update({"activate": x}), activate, None)
    autoplay.change(lambda x: params.update({"autoplay": x}), autoplay, None)
    low_vram.change(lambda x: params.update({"low_vram": x}), low_vram, None)
    low_vram.change(handle_low_vram, low_vram, None)
    tts_radio_buttons.change(handle_tts_method_change, tts_radio_buttons, None)
    deepspeed_checkbox.change(handle_deepspeed_activate_checkbox_change, deepspeed_checkbox, None)
    remove_trailing_dots.change(lambda x: params.update({"remove_trailing_dots": x}), remove_trailing_dots, None)
    voice.change(lambda x: params.update({"voice": x}), voice, None)
    language.change(lambda x: params.update({"language": x}), language, None)

    # Play preview
    preview_text.submit(voice_preview, preview_text, preview_audio)
    preview_play.click(voice_preview, preview_text, preview_audio)