import torch
import torchaudio

# needs to be before the tortoise stuff to properly import
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'tortoise'))

from tortoise import api
from tortoise.utils.audio import load_voices
from tortoise.utils.text import split_and_recombine_text

from pathlib import Path
import time

from modules import chat, shared, tts_preprocessor
from modules.html_generator import chat_html_wrapper

import gradio as gr

params = {
    'activate': True,
    'voice': 'emma',
    'preset': 'standard',
    'device': 'cuda',
    'show_text': True,
    'autoplay': True,
}

voices = [
    'angie',
    'applejack',
    'cond_latent_example',
    'daniel',
    'deniro',
    'emma',
    'freeman',
    'geralt',
    'halle',
    'jlaw',
    'lj',
    'mol',
    'myself',
    'pat',
    'pat2',
    'rainbow',
    'snakes',
    'tim_reynolds',
    'tom',
    'train_atkins',
    'train_daws',
    'train_dotrice',
    'train_dreams',
    'train_empire',
    'train_grace',
    'train_kennard',
    'train_lescault',
    'train_mouse',
    'weaver',
    'william'
]

presets = ['ultra_fast', 'fast', 'standard', 'high_quality']


def load_model():
    # Init TTS
    tts = api.TextToSpeech()
    samples, latents = load_voices(voices=[params['voice']])

    return tts, samples, latents


model, voice_samples, conditioning_latents = load_model()
current_params = params.copy()
streaming_state = shared.args.no_stream  # remember if chat streaming was enabled


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


def input_modifier(string):
    """
    This function is applied to your text inputs before
    they are fed into the model.
    """

    # Remove autoplay from the last reply
    if shared.is_chat() and len(shared.history['internal']) > 0:
        shared.history['visible'][-1] = [shared.history['visible'][-1][0], shared.history['visible'][-1][1].replace('controls autoplay>', 'controls>')]

    shared.processing_message = "*Is recording a voice message...*"
    shared.args.no_stream = True  # Disable streaming cause otherwise the audio output will stutter and begin anew every time the message is being updated
    return string


def output_modifier(string):
    """
    This function is applied to the model outputs.
    """

    global model, voice_samples, conditioning_latents, params

    original_string = string
    # we don't need to handle numbers. The text normalizer in coqui does it better
    string = tts_preprocessor.replace_invalid_chars(string)
    string = tts_preprocessor.replace_abbreviations(string)
    string = tts_preprocessor.clean_whitespace(string)
    processed_string = string

    if string == '':
        string = '*Empty reply, try regenerating*'
    else:
        output_dir = Path(f'extensions/tortoise_tts_mrq/outputs/parts')
        if not output_dir.is_dir():
            output_dir.mkdir(parents=True, exist_ok=True)

        output_file = Path(f'extensions/tortoise_tts_mrq/outputs/test_{int(time.time())}.wav')

        if '|' in string:
            texts = string.split('|')
        else:
            texts = split_and_recombine_text(string, desired_length=10, max_length=400)

        all_parts = []
        for j, text in enumerate(texts):
            gen = model.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                                        preset=params['preset'], k=1, use_deterministic_seed=int(time.time()))
            gen = gen.squeeze(0).cpu()
            torchaudio.save(output_dir.joinpath(f'{j}_{int(time.time())}.wav'), gen, 24000)
            all_parts.append(gen)

        full_audio = torch.cat(all_parts, dim=-1)
        torchaudio.save(str(output_file), full_audio, 24000)

        autoplay = 'autoplay' if params['autoplay'] else ''
        string = f'<audio src="file/{output_file.as_posix()}" controls {autoplay}></audio>'
        if params['show_text']:
            string += f'\n\n{original_string}'

    shared.processing_message = "*Is typing...*"
    shared.args.no_stream = streaming_state  # restore the streaming option to the previous value
    return string


def bot_prefix_modifier(string):
    """
    This function is only applied in chat mode. It modifies
    the prefix text for the Bot and can be used to bias its
    behavior.
    """

    return string


def setup():
    global model, voice_samples, conditioning_latents
    model, voice_samples, conditioning_latents = load_model()


def ui():
    # Gradio elements
    with gr.Accordion("Tortoise TTS"):
        with gr.Row():
            activate = gr.Checkbox(value=params['activate'], label='Activate TTS')
            autoplay = gr.Checkbox(value=params['autoplay'], label='Play TTS automatically')

        show_text = gr.Checkbox(value=params['show_text'], label='Show message text under audio player')
        voice_dropdown = gr.Dropdown(value=params['voice'], choices=voices, label='Voice')
        preset_dropdown = gr.Textbox(value=params['preset'], choices=voices, label='Preset')
        device_textbox = gr.Textbox(value=params['device'], label='Device')

        with gr.Row():
            convert = gr.Button('Permanently replace audios with the message texts')
            convert_cancel = gr.Button('Cancel', visible=False)
            convert_confirm = gr.Button('Confirm (cannot be undone)', variant="stop", visible=False)

    # Convert history with confirmation
    convert_arr = [convert_confirm, convert, convert_cancel]
    convert.click(lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)], None, convert_arr)
    convert_confirm.click(lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)], None, convert_arr)
    convert_confirm.click(remove_tts_from_history, [shared.gradio[k] for k in ['name1', 'name2', 'mode']], shared.gradio['display'])
    convert_confirm.click(lambda: chat.save_history(mode='chat', timestamp=False), [], [], show_progress=False)
    convert_cancel.click(lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)], None, convert_arr)

    # Toggle message text in history
    show_text.change(lambda x: params.update({"show_text": x}), show_text, None)
    show_text.change(toggle_text_in_history, [shared.gradio[k] for k in ['name1', 'name2', 'mode']], shared.gradio['display'])
    show_text.change(lambda: chat.save_history(mode='chat', timestamp=False), [], [], show_progress=False)

    # Event functions to update the parameters in the backend
    activate.change(lambda x: params.update({"activate": x}), activate, None)
    autoplay.change(lambda x: params.update({"autoplay": x}), autoplay, None)
    voice_dropdown.change(lambda x: update_model(x), voice_dropdown, None)
    preset_dropdown.change(lambda x: params.update({"preset": x}), preset_dropdown, None)
    device_textbox.change(lambda x: params.update({"device": x}), device_textbox, None)


def update_model(x):
    params.update({"voice": x})
    global model, voice_samples, conditioning_latents
    model, voice_samples, conditioning_latents = load_model()
