import re
from pathlib import Path

import elevenlabs
import gradio as gr

from modules import chat, shared
from modules.html_generator import chat_html_wrapper

params = {
    'activate': True,
    'api_key': None,
    'selected_voice': 'None',
    'autoplay': False,
    'show_text': True,
}

voices = None
wav_idx = 0


def refresh_voices():
    global params
    your_voices = elevenlabs.voices(api_key=params['api_key'])
    voice_names = [voice.name for voice in your_voices]
    return voice_names


def refresh_voices_dd():
    all_voices = refresh_voices()
    return gr.Dropdown.update(value=all_voices[0], choices=all_voices)


def remove_tts_from_history(name1, name2, mode, style):
    for i, entry in enumerate(shared.history['internal']):
        shared.history['visible'][i] = [shared.history['visible'][i][0], entry[1]]

    return chat_html_wrapper(shared.history['visible'], name1, name2, mode, style)


def toggle_text_in_history(name1, name2, mode, style):
    for i, entry in enumerate(shared.history['visible']):
        visible_reply = entry[1]
        if visible_reply.startswith('<audio'):
            if params['show_text']:
                reply = shared.history['internal'][i][1]
                shared.history['visible'][i] = [
                    shared.history['visible'][i][0], f"{visible_reply.split('</audio>')[0]}</audio>\n\n{reply}"
                ]
            else:
                shared.history['visible'][i] = [
                    shared.history['visible'][i][0], f"{visible_reply.split('</audio>')[0]}</audio>"
                ]

    return chat_html_wrapper(shared.history['visible'], name1, name2, mode, style)


def remove_surrounded_chars(string):
    # this expression matches to 'as few symbols as possible (0 upwards) between any asterisks' OR
    # 'as few symbols as possible (0 upwards) between an asterisk and the end of the string'
    return re.sub('\*[^\*]*?(\*|$)', '', string)


def state_modifier(state):
    state['stream'] = False
    return state


def input_modifier(string):
    """
    This function is applied to your text inputs before
    they are fed into the model.
    """
    # Remove autoplay from the last reply
    if shared.is_chat() and len(shared.history['internal']) > 0:
        shared.history['visible'][-1] = [
            shared.history['visible'][-1][0],
            shared.history['visible'][-1][1].replace('controls autoplay>', 'controls>')
        ]

    if params['activate']:
        shared.processing_message = "*Is recording a voice message...*"

    return string


def output_modifier(string):
    """
    This function is applied to the model outputs.
    """

    global params, wav_idx

    if not params['activate']:
        return string

    original_string = string
    string = remove_surrounded_chars(string)
    string = string.replace('"', '')
    string = string.replace('“', '')
    string = string.replace('\n', ' ')
    string = string.strip()
    if string == '':
        string = 'empty reply, try regenerating'

    output_file = Path(f'extensions/elevenlabs_tts/outputs/{wav_idx:06d}.mp3'.format(wav_idx))
    print(f'Outputing audio to {str(output_file)}')
    try:
        audio = elevenlabs.generate(text=string, voice=params['selected_voice'], model="eleven_monolingual_v1")
        elevenlabs.save(audio, str(output_file))

        autoplay = 'autoplay' if params['autoplay'] else ''
        string = f'<audio src="file/{output_file.as_posix()}" controls {autoplay}></audio>'
        wav_idx += 1
    except elevenlabs.api.error.UnauthenticatedRateLimitError:
        string = "🤖 ElevenLabs Unauthenticated Rate Limit Reached - Please create an API key to continue\n\n"
    except elevenlabs.api.error.RateLimitError:
        string = "🤖 ElevenLabs API Tier Limit Reached\n\n"
    except elevenlabs.api.error.APIError as err:
        string = f"🤖 ElevenLabs Error: {err}\n\n"

    if params['show_text']:
        string += f'\n\n{original_string}'

    shared.processing_message = "*Is typing...*"
    return string


def ui():
    global voices
    if not voices:
        voices = refresh_voices()
        params['selected_voice'] = voices[0]

    # Gradio elements
    with gr.Row():
        activate = gr.Checkbox(value=params['activate'], label='Activate TTS')
        autoplay = gr.Checkbox(value=params['autoplay'], label='Play TTS automatically')
        show_text = gr.Checkbox(value=params['show_text'], label='Show message text under audio player')

    with gr.Row():
        voice = gr.Dropdown(value=params['selected_voice'], choices=voices, label='TTS Voice')
        refresh = gr.Button(value='Refresh')

    with gr.Row():
        api_key = gr.Textbox(placeholder="Enter your API key.", label='API Key')

    with gr.Row():
        convert = gr.Button('Permanently replace audios with the message texts')
        convert_cancel = gr.Button('Cancel', visible=False)
        convert_confirm = gr.Button('Confirm (cannot be undone)', variant="stop", visible=False)

    # Convert history with confirmation
    convert_arr = [convert_confirm, convert, convert_cancel]
    convert.click(
        lambda: [gr.update(visible=True), gr.update(visible=False),
                 gr.update(visible=True)], None, convert_arr
    )
    convert_confirm.click(
        lambda: [gr.update(visible=False), gr.update(visible=True),
                 gr.update(visible=False)], None, convert_arr
    )
    convert_confirm.click(
        remove_tts_from_history, [shared.gradio[k] for k in ['name1', 'name2', 'mode', 'chat_style']], shared.gradio['display']
    )
    convert_confirm.click(chat.save_history, shared.gradio['mode'], [], show_progress=False)
    convert_cancel.click(
        lambda: [gr.update(visible=False), gr.update(visible=True),
                 gr.update(visible=False)], None, convert_arr
    )

    # Event functions to update the parameters in the backend
    activate.change(lambda x: params.update({'activate': x}), activate, None)
    voice.change(lambda x: params.update({'selected_voice': x}), voice, None)
    api_key.change(lambda x: params.update({'api_key': x}), api_key, None)
    # connect.click(check_valid_api, [], connection_status)
    refresh.click(refresh_voices_dd, [], voice)
    # Toggle message text in history
    show_text.change(lambda x: params.update({"show_text": x}), show_text, None)
    show_text.change(
        toggle_text_in_history, [shared.gradio[k] for k in ['name1', 'name2', 'mode', 'chat_style']], shared.gradio['display']
    )
    show_text.change(chat.save_history, shared.gradio['mode'], [], show_progress=False)
    # Event functions to update the parameters in the backend
    autoplay.change(lambda x: params.update({"autoplay": x}), autoplay, None)
