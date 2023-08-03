import time
from pathlib import Path

import gradio as gr
import torch

from extensions.silero_tts import tts_preprocessor
from modules import chat, shared
from modules.utils import gradio

torch._C._jit_set_profiling_mode(False)


params = {
    'activate': True,
    'speaker': 'en_56',
    'language': 'en',
    'model_id': 'v3_en',
    'sample_rate': 48000,
    'device': 'cpu',
    'show_text': False,
    'autoplay': True,
    'voice_pitch': 'medium',
    'voice_speed': 'medium',
    'local_cache_path': ''  # User can override the default cache path to something other via settings.json
}

current_params = params.copy()
voices_by_gender = ['en_99', 'en_45', 'en_18', 'en_117', 'en_49', 'en_51', 'en_68', 'en_0', 'en_26', 'en_56', 'en_74', 'en_5', 'en_38', 'en_53', 'en_21', 'en_37', 'en_107', 'en_10', 'en_82', 'en_16', 'en_41', 'en_12', 'en_67', 'en_61', 'en_14', 'en_11', 'en_39', 'en_52', 'en_24', 'en_97', 'en_28', 'en_72', 'en_94', 'en_36', 'en_4', 'en_43', 'en_88', 'en_25', 'en_65', 'en_6', 'en_44', 'en_75', 'en_91', 'en_60', 'en_109', 'en_85', 'en_101', 'en_108', 'en_50', 'en_96', 'en_64', 'en_92', 'en_76', 'en_33', 'en_116', 'en_48', 'en_98', 'en_86', 'en_62', 'en_54', 'en_95', 'en_55', 'en_111', 'en_3', 'en_83', 'en_8', 'en_47', 'en_59', 'en_1', 'en_2', 'en_7', 'en_9', 'en_13', 'en_15', 'en_17', 'en_19', 'en_20', 'en_22', 'en_23', 'en_27', 'en_29', 'en_30', 'en_31', 'en_32', 'en_34', 'en_35', 'en_40', 'en_42', 'en_46', 'en_57', 'en_58', 'en_63', 'en_66', 'en_69', 'en_70', 'en_71', 'en_73', 'en_77', 'en_78', 'en_79', 'en_80', 'en_81', 'en_84', 'en_87', 'en_89', 'en_90', 'en_93', 'en_100', 'en_102', 'en_103', 'en_104', 'en_105', 'en_106', 'en_110', 'en_112', 'en_113', 'en_114', 'en_115']
voice_pitches = ['x-low', 'low', 'medium', 'high', 'x-high']
voice_speeds = ['x-slow', 'slow', 'medium', 'fast', 'x-fast']

# Used for making text xml compatible, needed for voice pitch and speed control
table = str.maketrans({
    "<": "&lt;",
    ">": "&gt;",
    "&": "&amp;",
    "'": "&apos;",
    '"': "&quot;",
})


def xmlesc(txt):
    return txt.translate(table)


def load_model():
    torch_cache_path = torch.hub.get_dir() if params['local_cache_path'] == '' else params['local_cache_path']
    model_path = torch_cache_path + "/snakers4_silero-models_master/src/silero/model/" + params['model_id'] + ".pt"
    if Path(model_path).is_file():
        print(f'\nUsing Silero TTS cached checkpoint found at {torch_cache_path}')
        model, example_text = torch.hub.load(repo_or_dir=torch_cache_path + '/snakers4_silero-models_master/', model='silero_tts', language=params['language'], speaker=params['model_id'], source='local', path=model_path, force_reload=True)
    else:
        print(f'\nSilero TTS cache not found at {torch_cache_path}. Attempting to download...')
        model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_tts', language=params['language'], speaker=params['model_id'])
    model.to(params['device'])
    return model


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
    global model, current_params, streaming_state
    for i in params:
        if params[i] != current_params[i]:
            model = load_model()
            current_params = params.copy()
            break

    if not params['activate']:
        return string

    original_string = string
    string = tts_preprocessor.preprocess(string)

    if string == '':
        string = '*Empty reply, try regenerating*'
    else:
        output_file = Path(f'extensions/silero_tts/outputs/{state["character_menu"]}_{int(time.time())}.wav')
        prosody = '<prosody rate="{}" pitch="{}">'.format(params['voice_speed'], params['voice_pitch'])
        silero_input = f'<speak>{prosody}{xmlesc(string)}</prosody></speak>'
        model.save_wav(ssml_text=silero_input, speaker=params['speaker'], sample_rate=int(params['sample_rate']), audio_path=str(output_file))

        autoplay = 'autoplay' if params['autoplay'] else ''
        string = f'<audio src="file/{output_file.as_posix()}" controls {autoplay}></audio>'
        if params['show_text']:
            string += f'\n\n{original_string}'

    shared.processing_message = "*Is typing...*"
    return string


def setup():
    global model
    model = load_model()


def ui():
    # Gradio elements
    with gr.Accordion("Silero TTS"):
        with gr.Row():
            activate = gr.Checkbox(value=params['activate'], label='Activate TTS')
            autoplay = gr.Checkbox(value=params['autoplay'], label='Play TTS automatically')

        show_text = gr.Checkbox(value=params['show_text'], label='Show message text under audio player')
        voice = gr.Dropdown(value=params['speaker'], choices=voices_by_gender, label='TTS voice')
        with gr.Row():
            v_pitch = gr.Dropdown(value=params['voice_pitch'], choices=voice_pitches, label='Voice pitch')
            v_speed = gr.Dropdown(value=params['voice_speed'], choices=voice_speeds, label='Voice speed')

        with gr.Row():
            convert = gr.Button('Permanently replace audios with the message texts')
            convert_cancel = gr.Button('Cancel', visible=False)
            convert_confirm = gr.Button('Confirm (cannot be undone)', variant="stop", visible=False)

        gr.Markdown('[Click here for Silero audio samples](https://oobabooga.github.io/silero-samples/index.html)')

    if shared.is_chat():
        # Convert history with confirmation
        convert_arr = [convert_confirm, convert, convert_cancel]
        convert.click(lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)], None, convert_arr)
        convert_confirm.click(
            lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)], None, convert_arr).then(
            remove_tts_from_history, gradio('history'), gradio('history')).then(
            chat.save_persistent_history, gradio('history', 'character_menu', 'mode'), None).then(
            chat.redraw_html, shared.reload_inputs, gradio('display'))

        convert_cancel.click(lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)], None, convert_arr)

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
    v_pitch.change(lambda x: params.update({"voice_pitch": x}), v_pitch, None)
    v_speed.change(lambda x: params.update({"voice_speed": x}), v_speed, None)
