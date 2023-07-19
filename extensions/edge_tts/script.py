import time
from pathlib import Path

import gradio as gr
import torch
import edge_tts
import asyncio

from modules import chat, shared
from modules.utils import gradio

torch._C._jit_set_profiling_mode(False)


params = {
    'activate': True,
    'speaker': 'en-US-MichelleNeural',
    'show_text': False,
    'autoplay': True
}

current_params = params.copy()
voices = []

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


def history_modifier(history):
    # Remove autoplay from the last reply
    if len(history['internal']) > 0:
        history['visible'][-1] = [
            history['visible'][-1][0],
            history['visible'][-1][1].replace('controls autoplay>', 'controls>')
        ]

    return history

def output_modifier(string, state):
    if not params['activate']:
        return string

    original_string = string

    if string == '':
        string = '*Empty reply, try regenerating*'
    else:
        output_file = Path(f'extensions/edge_tts/outputs/{int(time.time())}.mp3')

        print(f'Outputting audio to {str(output_file)}')
        # print(f'{string}')

        communicate = edge_tts.Communicate(string, params['speaker'])
        asyncio.run(communicate.save(output_file))
        
        autoplay = 'autoplay' if params['autoplay'] else ''
        string = f'<audio src="file/{output_file.as_posix()}" controls {autoplay}></audio>'
        if params['show_text']:
            string += f'\n\n{original_string}'

    shared.processing_message = "*Is typing...*"
    return string


def get_voices():
    voices = asyncio.run(edge_tts.list_voices())
    names = [x['ShortName'] for x in voices]
    return names

def setup():
    global voices, current_params
    for i in params:
        if params[i] != current_params[i]:
            current_params = params.copy()
            break
    voices = get_voices()

def ui():
    # Gradio elements
    with gr.Accordion("Edge TTS"):
        with gr.Row():
            activate = gr.Checkbox(value=params['activate'], label='Activate TTS')
            autoplay = gr.Checkbox(value=params['autoplay'], label='Play TTS automatically')

        show_text = gr.Checkbox(value=params['show_text'], label='Show message text under audio player')
        voice = gr.Dropdown(value=params['speaker'], choices=voices, label='TTS voice')

    if shared.is_chat():
        # Toggle message text in history
        show_text.change(
            lambda x: params.update({"show_text": x}), show_text, None).then(
            toggle_text_in_history, gradio('history'), gradio('history')).then(
            chat.save_persistent_history, gradio('history', 'character_menu', 'mode'), None).then(
            chat.redraw_html, shared.reload_inputs, gradio('display'))
        
    # Event functions to update the parameters in the backend
    activate.change(lambda x: params.update({"activate": x}), activate, None)
    autoplay.change(lambda x: params.update({"autoplay": x}), autoplay, None)
    voice.change(lambda x: params.update({"speaker": x}), voice, None)