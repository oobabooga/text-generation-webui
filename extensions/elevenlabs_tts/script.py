import re
from pathlib import Path

import gradio as gr
import modules.shared as shared
from elevenlabslib import ElevenLabsUser
from elevenlabslib.helpers import save_bytes_to_path

params = {
    'activate': True,
    'api_key': '12345',
    'selected_voice': 'None',
}

initial_voice = ['None']
wav_idx = 0
user = ElevenLabsUser(params['api_key'])
user_info = None

if not shared.args.no_stream:
    print("Please add --no-stream. This extension is not meant to be used with streaming.")
    raise ValueError
    
# Check if the API is valid and refresh the UI accordingly.
def check_valid_api():
    
    global user, user_info, params

    user = ElevenLabsUser(params['api_key'])
    user_info = user._get_subscription_data()
    print('checking api')
    if params['activate'] == False:
        return gr.update(value='Disconnected')
    elif user_info is None:
        print('Incorrect API Key')
        return gr.update(value='Disconnected')
    else:
        print('Got an API Key!')
        return gr.update(value='Connected')
    
# Once the API is verified, get the available voices and update the dropdown list
def refresh_voices():
    
    global user, user_info
    
    your_voices = [None]
    if user_info is not None:
        for voice in user.get_available_voices():
            your_voices.append(voice.initialName)
        return  gr.Dropdown.update(choices=your_voices)
    else:
        return

def remove_surrounded_chars(string):
    # this expression matches to 'as few symbols as possible (0 upwards) between any asterisks' OR
    # 'as few symbols as possible (0 upwards) between an asterisk and the end of the string'
    return re.sub('\*[^\*]*?(\*|$)','',string)

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

    global params, wav_idx, user, user_info
    
    if params['activate'] == False:
        return string
    elif user_info == None:
        return string

    string = remove_surrounded_chars(string)
    string = string.replace('"', '')
    string = string.replace('“', '')
    string = string.replace('\n', ' ')
    string = string.strip()

    if string == '':
        string = 'empty reply, try regenerating'
        
    output_file = Path(f'extensions/elevenlabs_tts/outputs/{wav_idx:06d}.wav'.format(wav_idx))
    voice = user.get_voices_by_name(params['selected_voice'])[0]
    audio_data = voice.generate_audio_bytes(string)
    save_bytes_to_path(Path(f'extensions/elevenlabs_tts/outputs/{wav_idx:06d}.wav'), audio_data)

    string = f'<audio src="file/{output_file.as_posix()}" controls></audio>'
    wav_idx += 1
    return string

def ui():

    # Gradio elements
    with gr.Row():
        activate = gr.Checkbox(value=params['activate'], label='Activate TTS')
        connection_status = gr.Textbox(value='Disconnected', label='Connection Status')
    voice = gr.Dropdown(value=params['selected_voice'], choices=initial_voice, label='TTS Voice')
    with gr.Row():
        api_key = gr.Textbox(placeholder="Enter your API key.", label='API Key')
        connect = gr.Button(value='Connect')

    # Event functions to update the parameters in the backend
    activate.change(lambda x: params.update({'activate': x}), activate, None)
    voice.change(lambda x: params.update({'selected_voice': x}), voice, None)
    api_key.change(lambda x: params.update({'api_key': x}), api_key, None)
    connect.click(check_valid_api, [], connection_status)
    connect.click(refresh_voices, [], voice)