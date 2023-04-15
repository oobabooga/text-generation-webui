import os
import sys

import torch
import torchaudio

sys.path.append(os.path.join(os.path.dirname(__file__), 'tortoise'))
from .tortoise.tortoise import api
from .tortoise.tortoise.utils.audio import load_voices
from .tortoise.tortoise.utils.text import split_and_recombine_text

from pathlib import Path
import time

from modules import chat, shared, tts_preprocessor
from modules.html_generator import chat_html_wrapper

import gradio as gr

params = {
    'activate': True,
    'voice_dir': None,
    'voice': 'emma',
    'preset': 'standard',
    'seed': None,
    'device': 'cuda',
    'sentence_length': 10,
    'show_text': True,
    'autoplay': True,
    'tuning_settings': {
        'verbose': False,
        'k': 1,
        'num_autoregressive_samples': None,
        'temperature': None,
        'length_penalty': None,
        'repetition_penalty': None,
        'top_p': None,
        'max_mel_tokens': None,
        'cvvp_amount': None,
        'diffusion_iterations': None,
        'cond_free': None,
        'cond_free_k': None,
        'diffusion_temperature': None
    }
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
    models_dir = shared.args.model_dir if hasattr(shared.args, 'model_dir') and shared.args.model_dir is not None else \
        api.MODELS_DIR
    tts = api.TextToSpeech(models_dir=os.path.join(models_dir, 'tortoise'), device=params['device'])
    samples, latents = load_voices(voices=[params['voice']],
                                   extra_voice_dirs=[params['voice_dir']] if params['voice_dir'] is not None else [])

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
        output_dir = Path(f'extensions/tortoise_tts/outputs/parts')
        if not output_dir.is_dir():
            output_dir.mkdir(parents=True, exist_ok=True)

        output_file = Path(f'extensions/tortoise_tts/outputs/test_{int(time.time())}.wav')

        if '|' in string:
            texts = string.split('|')
        else:
            texts = split_and_recombine_text(string, desired_length=params['sentence_length'], max_length=1000)

        gen_kwargs = {
            'use_deterministic_seed': int(time.time()) if params['seed'] is None else params['seed']
        }

        for option in params['tuning_settings'].keys():
            if params['tuning_settings'][option] is not None:
                gen_kwargs[option] = params['tuning_settings'][option]

        all_parts = []
        # only cat if it's needed
        if len(texts) > 1:
            for j, text in enumerate(texts):
                gen = model.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                                            **gen_kwargs)
                gen = gen.squeeze(0).cpu()
                torchaudio.save(output_dir.joinpath(f'{j}_{int(time.time())}.wav'), gen, 24000)
                all_parts.append(gen)

            full_audio = torch.cat(all_parts, dim=-1)
            torchaudio.save(str(output_file), full_audio, 24000)
        elif len(texts) == 1:
            text = texts[1]
            gen = model.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                                        **gen_kwargs)
            gen = gen.squeeze(0).cpu()
            torchaudio.save(str(output_file), gen, 24000)

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
        preset_dropdown = gr.Dropdown(value=params['preset'], choices=presets, label='Preset')
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
