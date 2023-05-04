import re
from pathlib import Path

import gradio as gr
from elevenlabslib import ElevenLabsUser
from elevenlabslib.helpers import save_audio_bytes

from modules import chat, shared
from modules.html_generator import chat_html_wrapper

params = {
    'activate': True,
    'api_key': '12345',
    'selected_voice': 'None',
    'autoplay': True,
    'show_text': True,
}

initial_voice = ['None']
wav_idx = 0
# user = ElevenLabsUser(params['api_key'])
user = None
user_info = None
streaming_state = shared.args.no_stream  # remember if chat streaming was enabled

# Check if the API is valid and refresh the UI accordingly.


def check_valid_api():

    global user, user_info, params

    user = ElevenLabsUser(params['api_key'])
    user_info = user._get_subscription_data()
    print('checking api')
    if not params['activate']:
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
        return gr.Dropdown.update(choices=your_voices)
    else:
        return

def remove_tts_from_history(name1, name2, mode):
    for i, entry in enumerate(shared.history['internal']):
        shared.history['visible'][i] = [shared.history['visible'][i][0], entry[1]]
    return chat_html_wrapper(shared.history['visible'], name1, name2, mode)


def toggle_text_in_history(name1, name2, mode):
    for i, entry in enumerate(shared.history['visible']):
        visible_reply = entry[1]
        if visible_reply.startswith('<audio'):
            if params['show_text']:
                reply = shared.history['internal'][i][1]
                shared.history['visible'][i] = [shared.history['visible'][i][0], f"{visible_reply.split('</audio>')[0]}</audio>\n\n{reply}"]
            else:
                shared.history['visible'][i] = [shared.history['visible'][i][0], f"{visible_reply.split('</audio>')[0]}</audio>"]
    return chat_html_wrapper(shared.history['visible'], name1, name2, mode)

def remove_surrounded_chars(string):
    # this expression matches to 'as few symbols as possible (0 upwards) between any asterisks' OR
    # 'as few symbols as possible (0 upwards) between an asterisk and the end of the string'
    return re.sub('\*[^\*]*?(\*|$)', '', string)


def input_modifier(string):
    """
    This function is applied to your text inputs before
    they are fed into the model.
    """
    # Remove autoplay from the last reply
    if shared.is_chat() and len(shared.history['internal']) > 0:
        shared.history['visible'][-1] = [shared.history['visible'][-1][0], shared.history['visible'][-1][1].replace('controls autoplay>', 'controls>')]
    shared.processing_message = "*Is recording a voice message...*"
    shared.args.no_stream = True  # Disable streaming
    return string


def output_modifier(string):
    """
    This function is applied to the model outputs.
    """

    global params, wav_idx, user, user_info, streaming_state

    if not params['activate']:
        return string
    elif user_info is None:
        return string

    original_string = string
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
    save_audio_bytes(audio_data, str(output_file), outputFormat="wav")

    autoplay = 'autoplay' if params['autoplay'] else ''
    string = f'<audio src="file/{output_file.as_posix()}" controls {autoplay}></audio>'
    if params['show_text']:
        string += f'\n\n{original_string}'
    wav_idx += 1
    shared.processing_message = "*Is typing...*"
    shared.args.no_stream = streaming_state  # restore the streaming option 
    return string


def ui():

    # Gradio elements
    with gr.Row():
        activate = gr.Checkbox(value=params['activate'], label='Activate TTS')
        autoplay = gr.Checkbox(value=params['autoplay'], label='Play TTS automatically')
        show_text = gr.Checkbox(value=params['show_text'], label='Show message text under audio player')
    with gr.Row():
        voice = gr.Dropdown(value=params['selected_voice'], choices=initial_voice, label='TTS Voice')
        connection_status = gr.Textbox(value='Disconnected', label='Connection Status')
    with gr.Row():
        api_key = gr.Textbox(placeholder="Enter your API key.", label='API Key')
        connect = gr.Button(value='Connect')
    with gr.Row():
        convert = gr.Button('Permanently replace audios with the message texts')
        convert_cancel = gr.Button('Cancel', visible=False)
        convert_confirm = gr.Button('Confirm (cannot be undone)', variant="stop", visible=False)
        
    # Convert history with confirmation
    convert_arr = [convert_confirm, convert, convert_cancel]
    convert.click(lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)], None, convert_arr)
    convert_confirm.click(lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)], None, convert_arr)
    convert_confirm.click(remove_tts_from_history, [shared.gradio[k] for k in ['name1', 'name2', 'mode']], shared.gradio['display'])
    convert_confirm.click(lambda: chat.save_history(timestamp=False), [], [], show_progress=False)
    convert_cancel.click(lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)], None, convert_arr)
    # Event functions to update the parameters in the backend
    activate.change(lambda x: params.update({'activate': x}), activate, None)
    voice.change(lambda x: params.update({'selected_voice': x}), voice, None)
    api_key.change(lambda x: params.update({'api_key': x}), api_key, None)
    connect.click(check_valid_api, [], connection_status)
    connect.click(refresh_voices, [], voice)
    # Toggle message text in history
    show_text.change(lambda x: params.update({"show_text": x}), show_text, None)
    show_text.change(toggle_text_in_history, [shared.gradio[k] for k in ['name1', 'name2', 'mode']], shared.gradio['display'])
    show_text.change(lambda: chat.save_history(timestamp=False), [], [], show_progress=False)
    # Event functions to update the parameters in the backend
    autoplay.change(lambda x: params.update({"autoplay": x}), autoplay, None)
