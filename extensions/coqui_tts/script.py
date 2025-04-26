import html
import json
import os
import random
import time
from pathlib import Path

import gradio as gr
import torch
from TTS.api import TTS
from TTS.utils.synthesizer import Synthesizer

from modules import chat, shared, ui_chat
from modules.ui import create_refresh_button
from modules.utils import gradio

os.environ["COQUI_TOS_AGREED"] = "1"

params = {
    "activate": True,
    "autoplay": True,
    "show_text": False,
    "remove_trailing_dots": False,
    "voice": "female_01.wav",
    "language": "English",
    "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

this_dir = str(Path(__file__).parent.resolve())
model = None
with open(Path(f"{this_dir}/languages.json"), encoding='utf8') as f:
    languages = json.load(f)


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


def load_model():
    model = TTS(params["model_name"]).to(params["device"])
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


def random_sentence():
    with open(Path("extensions/coqui_tts/harvard_sentences.txt")) as f:
        return random.choice(list(f))


def voice_preview(string):
    string = html.unescape(string) or random_sentence()

    output_file = Path('extensions/coqui_tts/outputs/voice_preview.wav')
    model.tts_to_file(
        text=string,
        file_path=output_file,
        speaker_wav=[f"{this_dir}/voices/{params['voice']}"],
        language=languages[params["language"]]
    )

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


def output_modifier(string, state):
    if not params['activate']:
        return string

    original_string = string
    string = preprocess(html.unescape(string))
    if string == '':
        string = '*Empty reply, try regenerating*'
    else:
        output_file = Path(f'extensions/coqui_tts/outputs/{state["character_menu"]}_{int(time.time())}.wav')
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
    return string


def custom_css():
    path_to_css = Path(f"{this_dir}/style.css")
    return open(path_to_css, 'r').read()


def setup():
    global model
    print("[XTTS] Loading XTTS...")
    model = load_model()
    print("[XTTS] Done!")
    Path(f"{this_dir}/outputs").mkdir(parents=True, exist_ok=True)


def ui():
    with gr.Accordion("Coqui TTS (XTTSv2)"):
        with gr.Row():
            activate = gr.Checkbox(value=params['activate'], label='Activate TTS')
            autoplay = gr.Checkbox(value=params['autoplay'], label='Play TTS automatically')

        with gr.Row():
            show_text = gr.Checkbox(value=params['show_text'], label='Show message text under audio player')
            remove_trailing_dots = gr.Checkbox(value=params['remove_trailing_dots'], label='Remove trailing "." from text segments before converting to audio')

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
    remove_trailing_dots.change(lambda x: params.update({"remove_trailing_dots": x}), remove_trailing_dots, None)
    voice.change(lambda x: params.update({"voice": x}), voice, None)
    language.change(lambda x: params.update({"language": x}), language, None)

    # Play preview
    preview_text.submit(voice_preview, preview_text, preview_audio)
    preview_play.click(voice_preview, preview_text, preview_audio)
