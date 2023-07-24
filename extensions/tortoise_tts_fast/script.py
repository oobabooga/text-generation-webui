import gc
import os
import sys
import traceback
from typing import Union

import torch
import torchaudio

from pathlib import Path
import time

from modules import chat, shared, tts_preprocessor
from modules.models import reload_model as load_llm, unload_model as unload_llm
from modules.utils import gradio

import gradio as gr

sys.path.append(os.path.join(os.path.dirname(__file__), 'tortoise'))
from .tortoise.tortoise import api
from .tortoise.tortoise.models import utils
from .tortoise.tortoise.utils import audio
from .tortoise.tortoise.utils.text import split_and_recombine_text


params = {
    'activate': True,
    'voice_dir': None,
    'output_dir': None,
    'voice': 'emma',
    'preset': 'single_sample',
    'low_vram': True,
    'model_swap': True,
    'seed': 0,
    'sentence_length': 10,
    'show_text': True,
    'autoplay': True,
    'tuning_settings': {
        'k': 1,
        'num_autoregressive_samples': 0,
        'temperature': 0,
        'length_penalty': 0,
        'repetition_penalty': 0,
        'top_p': 0,
        'max_mel_tokens': 0,
        'cvvp_amount': 0,
        'diffusion_iterations': 0,
        'cond_free': None,
        'cond_free_k': 0,
        'diffusion_temperature': 0,
        'sampler': None
    }
}

# Presets are defined here.
preset_options = {
    "single_sample": {"num_autoregressive_samples": 8, "diffusion_iterations": 10, "sampler": "ddim"},
    "ultra_fast": {"num_autoregressive_samples": 16, "diffusion_iterations": 10, "sampler": "ddim"},
    "ultra_fast_old": {"num_autoregressive_samples": 16, "diffusion_iterations": 30, "cond_free": False},
    "very_fast": {"num_autoregressive_samples": 32, "diffusion_iterations": 30, "sampler": "dpm++2m"},
    "fast": {"num_autoregressive_samples": 96, "diffusion_iterations": 20, "sampler": "dpm++2m"},
    "fast_old": {"num_autoregressive_samples": 96, "diffusion_iterations": 80},
    "standard": {"num_autoregressive_samples": 256, "diffusion_iterations": 200},
    "high_quality": {"num_autoregressive_samples": 256, "diffusion_iterations": 400},
}

presets = list(preset_options.keys())
model = voice_samples = conditioning_latents = voices = current_params = None
streaming_state = shared.args.no_stream  # remember if chat streaming was enabled
controls = {}


def set_preset(preset):
    global params
    settings = get_preset_settings(preset)
    for opt in settings.keys():
        option = params['tuning_settings'][opt]
        if option == settings[opt]:
            continue

        if isinstance(option, bool) and option is not None:
            continue

        if isinstance(option, str) and option is not None and option != '':
            continue

        if isinstance(option, (int, float)) and option > 0:
            continue

        params['tuning_settings'][opt] = settings[opt]


def get_preset_settings(preset):
    settings = {
        "temperature": 0.2, "length_penalty": 1.0, "repetition_penalty": 2.0, "top_p": 0.8, "cond_free_k": 2.0,
        "diffusion_temperature": 1.0, 'num_autoregressive_samples': 512, 'max_mel_tokens': 500, 'cvvp_amount': 0,
        'diffusion_iterations': 100, 'cond_free': True, 'sampler': 'ddim'
    }

    settings.update(preset_options[preset])
    return settings


def get_gen_kwargs(par):
    gen_kwargs = {
        'preset': par['preset'],
        'use_deterministic_seed': int(time.time()) if par['seed'] is None or par['seed'] == 0 else par['seed'],
        'k': 1
    }

    preset_options = get_preset_settings(par['preset'])

    for option in par['tuning_settings'].keys():
        opt: [Union[float, int, str, bool, None]] = par['tuning_settings'][option]
        if opt is None:
            continue

        if isinstance(opt, (int, float)) and opt <= 0:
            continue

        if isinstance(opt, str) and opt == '':
            continue

        if option in preset_options.keys() and preset_options[option] == opt:
            continue

        gen_kwargs[option] = opt

    return gen_kwargs


def get_voices():
    extra_voice_dirs = [params['voice_dir']] if params['voice_dir'] is not None and Path(params['voice_dir']).is_dir() else []
    detected_voices = audio.get_voices(extra_voice_dirs=extra_voice_dirs)
    detected_voices = sorted(detected_voices.keys()) if len(detected_voices) > 0 else []
    return detected_voices


def load_model():
    # Init TTS
    try:
        global params
        extra_voice_dirs = [params['voice_dir']] if params['voice_dir'] is not None else []
        models_dir = shared.args.model_dir if hasattr(shared.args, 'model_dir') and shared.args.model_dir is not None else utils.MODELS_DIR
        if not Path(models_dir).is_dir():
            Path(models_dir).mkdir(parents=True, exist_ok=True)

        utils.MODELS_DIR = os.path.join(models_dir, 'tortoise')
        tts = api.TextToSpeech(high_vram=not params['low_vram'], models_dir=utils.MODELS_DIR)
        samples, latents = audio.load_voice(voice=params['voice'], extra_voice_dirs=extra_voice_dirs)
    except Exception as e:
        print(e)
        return None, None, None

    return tts, samples, latents


def unload_model():
    try:
        global model, voice_samples, conditioning_latents
        model = voice_samples = conditioning_latents = None
        gc.collect()
        torch.cuda.empty_cache()
    except:
        pass


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
    """
    This function is applied to the model outputs.
    """

    try:
        global model, voice_samples, conditioning_latents, params, current_params

        refresh_model = False

        if params['voice'] != current_params['voice'] or params['low_vram'] != current_params['low_vram']:
            refresh_model = True

        for i in params:
            if params[i] != current_params[i]:
                current_params = params.copy()
                break

        if not current_params['activate']:
            return string

        if model is None:
            refresh_model = True

        if params['model_swap']:
            unload_llm()
            refresh_model = True

        if refresh_model:
            model, voice_samples, conditioning_latents = load_model()

        if model is None:
            return string

        original_string = string
        # we don't need to handle numbers. The text normalizer in tortoise does it better
        string = tts_preprocessor.replace_invalid_chars(string)
        string = tts_preprocessor.replace_abbreviations(string)
        string = tts_preprocessor.clean_whitespace(string)

        if string == '':
            string = '*Empty reply, try regenerating*'
            if params['model_swap']:
                unload_model()
                load_llm()

            return string

        out_dir_root = params['output_dir'] if params['output_dir'] is not None and Path(params['output_dir']).is_dir() \
            else 'extensions/tortoise_tts_fast/outputs'

        output_dir = Path(out_dir_root).joinpath('parts')
        if not output_dir.is_dir():
            output_dir.mkdir(parents=True, exist_ok=True)

        output_file = Path(out_dir_root).joinpath(f'test_{int(time.time())}.wav')

        if '|' in string:
            texts = string.split('|')
        else:
            texts = split_and_recombine_text(string, desired_length=params['sentence_length'], max_length=1000)

        gen_kwargs = get_gen_kwargs(params)

        generate_audio(model, voice_samples, conditioning_latents, output_dir, output_file, gen_kwargs, texts)

        autoplay = 'autoplay' if params['autoplay'] else ''
        string = f'<audio src="file/{output_file.as_posix()}" controls {autoplay}></audio>'
        if params['show_text']:
            string += f'\n\n{original_string}'

        shared.processing_message = "*Is typing...*"
        if params['model_swap']:
            unload_model()
            load_llm()

        return string
    except Exception as e:
        shared.processing_message = "*Is typing...*"
        if params['model_swap']:
            unload_model()
            load_llm()
        return traceback.format_exc()


def generate_audio(tts, samples, latents, output_dir, output_file, gen_kwargs, texts):
    # only cat if it's needed
    if len(texts) <= 0:
        return

    if len(texts) == 1:
        text = texts[0]
        gen = tts.tts_with_preset(text, voice_samples=samples, conditioning_latents=latents, **gen_kwargs)
        gen = gen.squeeze(0).cpu()
        torchaudio.save(str(output_file), gen, 24000)
        return

    all_parts = []
    for j, text in enumerate(texts):
        gen = tts.tts_with_preset(text, voice_samples=samples, conditioning_latents=latents, **gen_kwargs)
        gen = gen.squeeze(0).cpu()
        torchaudio.save(str(output_dir.joinpath(f'{j}_{int(time.time())}.wav')), gen, 24000)
        all_parts.append(gen)

    full_audio = torch.cat(all_parts, dim=-1)
    torchaudio.save(str(output_file), full_audio, 24000)


def setup():
    global voices, model, voice_samples, conditioning_latents, current_params
    current_params = params.copy()
    voices = get_voices()
    set_preset(params['preset'])
    if not params['model_swap']:
        model, voice_samples, conditioning_latents = load_model()


def ui():
    global controls, params
    # Gradio elements
    with gr.Accordion("Tortoise TTS Fast"):
        with gr.Row():
            controls['activate'] = gr.Checkbox(value=params['activate'], label='Activate TTS')
            controls['autoplay'] = gr.Checkbox(value=params['autoplay'], label='Play TTS automatically')

        controls['show_text'] = gr.Checkbox(value=params['show_text'], label='Show message text under audio player')
        controls['voice_dropdown'] = gr.Dropdown(value=params['voice'], choices=voices, label='Voice')
        controls['voice_dir_textbox'] = gr.Textbox(value=params['voice_dir'], label='Custom Voices Directory')
        controls['output_dir_textbox'] = gr.Textbox(value=params['output_dir'], label='Custom Output Directory')
        controls['vram_checkbox'] = gr.Checkbox(value=params['low_vram'], label='Low VRAM')
        controls['model_swap'] = gr.Checkbox(value=params['model_swap'], label='Unload LLM Model to save VRAM')
        controls['seed_picker'] = gr.Number(value=params['seed'], precision=0, label='Seed', interactive=True)
        controls['sentence_picker'] = gr.Number(value=params['sentence_length'], precision=0, label='Optimal Sentence Length', interactive=True)
        controls['preset_dropdown'] = gr.Dropdown(value=params['preset'], choices=presets, label='Preset')
        with gr.Accordion(label='Tuning Settings', open=False):
            tune_settings: dict[str, (Union[float, int, str, bool])] = params['tuning_settings']
            controls['num_autoregressive_samples'] = gr.Number(value=tune_settings['num_autoregressive_samples'], label='num_autoregressive_samples', precision=0)
            controls['temperature'] = gr.Number(value=tune_settings['temperature'], label='temperature')
            controls['length_penalty'] = gr.Number(value=tune_settings['length_penalty'], label='length_penalty')
            controls['repetition_penalty'] = gr.Number(value=tune_settings['repetition_penalty'], label='repetition_penalty')
            controls['top_p'] = gr.Number(value=tune_settings['top_p'], label='top_p')
            controls['max_mel_tokens'] = gr.Number(value=tune_settings['max_mel_tokens'], label='max_mel_tokens', precision=0)
            controls['cvvp_amount'] = gr.Number(value=tune_settings['cvvp_amount'], label='cvvp_amount')
            controls['diffusion_iterations'] = gr.Number(value=tune_settings['diffusion_iterations'], label='diffusion_iterations', precision=0)
            controls['cond_free_k'] = gr.Number(value=tune_settings['cond_free_k'], label='cond_free_k')
            controls['diffusion_temperature'] = gr.Number(value=tune_settings['diffusion_temperature'], label='diffusion_temperature')
            controls['num_autoregressive_samples'].change(lambda x: params['tuning_settings'].update({'num_autoregressive_samples': x}), controls['num_autoregressive_samples'], outputs=None)
            controls['temperature'].change(lambda x: params['tuning_settings'].update({'temperature': x}), controls['temperature'], outputs=None)
            controls['length_penalty'].change(lambda x: params['tuning_settings'].update({'length_penalty': x}), controls['length_penalty'], outputs=None)
            controls['repetition_penalty'].change(lambda x: params['tuning_settings'].update({'repetition_penalty': x}), controls['repetition_penalty'], outputs=None)
            controls['top_p'].change(lambda x: params['tuning_settings'].update({'top_p': x}), controls['top_p'], outputs=None)
            controls['max_mel_tokens'].change(lambda x: params['tuning_settings'].update({'max_mel_tokens': x}), controls['max_mel_tokens'], outputs=None)
            controls['cvvp_amount'].change(lambda x: params['tuning_settings'].update({'cvvp_amount': x}), controls['cvvp_amount'], outputs=None)
            controls['diffusion_iterations'].change(lambda x: params['tuning_settings'].update({'diffusion_iterations': x}), controls['diffusion_iterations'], outputs=None)
            controls['cond_free_k'].change(lambda x: params['tuning_settings'].update({'cond_free_k': x}), controls['cond_free_k'], outputs=None)
            controls['diffusion_temperature'].change(lambda x: params['tuning_settings'].update({'diffusion_temperature': x}), controls['diffusion_temperature'], outputs=None)

        with gr.Row():
            controls['convert'] = gr.Button('Permanently replace audios with the message texts')
            controls['convert_cancel'] = gr.Button('Cancel', visible=False)
            controls['convert_confirm'] = gr.Button('Confirm (cannot be undone)', variant="stop", visible=False)

    # Convert history with confirmation
    controls['convert_arr'] = [controls['convert_confirm'], controls['convert'], controls['convert_cancel']]
    controls['convert'].click(lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)],
                              None, controls['convert_arr'])
    controls['convert_confirm'].click(
        lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)], None,
        controls['convert_arr']).then(
        remove_tts_from_history, gradio('history'), gradio('history')).then(
        chat.save_persistent_history, gradio('history', 'character_menu', 'mode'), None).then(
        chat.redraw_html, shared.reload_inputs, gradio('display'))

    controls['convert_cancel'].click(
        lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)], None,
        controls['convert_arr'])

    # Toggle message text in history
    controls['show_text'].change(
        lambda x: params.update({"show_text": x}), controls['show_text'], None).then(
        toggle_text_in_history, gradio('history'), gradio('history')).then(
        chat.save_persistent_history, gradio('history', 'character_menu', 'mode'), None).then(
        chat.redraw_html, shared.reload_inputs, gradio('display'))

    # Event functions to update the parameters in the backend
    controls['activate'].change(lambda x: params.update({"activate": x}), controls['activate'], None)
    controls['autoplay'].change(lambda x: params.update({"autoplay": x}), controls['autoplay'], None)
    controls['voice_dropdown'].change(lambda x: params.update({"voice": x}), controls['voice_dropdown'], None)
    controls['voice_dir_textbox'].change(update_voice_dir, [controls['voice_dir_textbox'], controls['voice_dropdown']], controls['voice_dropdown'])
    controls['output_dir_textbox'].change(lambda x: params.update({'output_dir': x}), controls['output_dir_textbox'], None)
    controls['preset_dropdown'].change(update_preset, controls['preset_dropdown'], outputs=[
        controls['num_autoregressive_samples'],
        controls['temperature'],
        controls['length_penalty'],
        controls['repetition_penalty'],
        controls['top_p'],
        controls['max_mel_tokens'],
        controls['cvvp_amount'],
        controls['diffusion_iterations'],
        controls['cond_free_k'],
        controls['diffusion_temperature']
    ])
    controls['vram_checkbox'].change(lambda x: params.update({'low_vram': x}), controls['vram_checkbox'], None)
    controls['model_swap'].change(lambda x: params.update({'model_swap': x}), controls['model_swap'], None)
    controls['seed_picker'].change(lambda x: params.update({'seed': x}), controls['seed_picker'], None)
    controls['sentence_picker'].change(lambda x: params.update({'sentence_length': x}), controls['sentence_picker'], None)


def update_voice_dir(x, voice):
    global controls, params, voices
    params.update({"voice_dir": x})
    voices = get_voices()
    controls['voice_dropdown'].choices = voices
    value = voice if voice in voices else voices[0] if len(voices) > 0 else None
    return gr.update(choices=voices, value=value, visible=True)


def update_preset(preset):
    global params
    params.update({'preset': preset})
    set_preset(preset)
    tune: dict[str, Union[float, int, str, bool]] = params['tuning_settings']
    return [
        # num_autoregressive_samples
        gr.update(value=tune['num_autoregressive_samples'], visible=True),
        # temperature
        gr.update(value=tune['temperature'], visible=True),
        # length_penalty
        gr.update(value=tune['length_penalty'], visible=True),
        # repetition_penalty
        gr.update(value=tune['repetition_penalty'], visible=True),
        # top_p
        gr.update(value=tune['top_p'], visible=True),
        # max_mel_tokens
        gr.update(value=tune['max_mel_tokens'], visible=True),
        # cvvp_amount
        gr.update(value=tune['cvvp_amount'], visible=True),
        # diffusion_iterations
        gr.update(value=tune['diffusion_iterations'], visible=True),
        # cond_free_k
        gr.update(value=tune['cond_free_k'], visible=True),
        # diffusion_temperature
        gr.update(value=tune['diffusion_temperature'], visible=True)
    ]
