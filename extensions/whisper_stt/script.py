import base64
import gc
import io
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import whisper
from pydub import AudioSegment

from modules import shared

input_hijack = {
    'state': False,
    'value': ["", ""]
}

# parameters which can be customized in settings.yaml of webui
params = {
    'whipser_language': 'english',
    'whipser_model': 'small.en',
    'auto_submit': True
}

startup_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WHISPERMODEL = whisper.load_model(params['whipser_model'], device=startup_device)


def chat_input_modifier(text, visible_text, state):
    global input_hijack
    if input_hijack['state']:
        input_hijack['state'] = False
        return input_hijack['value']
    else:
        return text, visible_text


def do_stt(audio, whipser_language):
    # use pydub to convert sample_rate and sample_width for whisper input
    dubaudio = AudioSegment.from_file(io.BytesIO(audio))
    dubaudio = dubaudio.set_channels(1)
    dubaudio = dubaudio.set_frame_rate(16000)
    dubaudio = dubaudio.set_sample_width(2)

    # same method to get the array as openai whisper repo used from wav file
    audio_np = np.frombuffer(dubaudio.raw_data, np.int16).flatten().astype(np.float32) / 32768.0

    if len(whipser_language) == 0:
        result = WHISPERMODEL.transcribe(audio=audio_np)
    else:
        result = WHISPERMODEL.transcribe(audio=audio_np, language=whipser_language)
    return result["text"]


def auto_transcribe(audio, auto_submit, whipser_language):
    if audio is None or audio == "":
        print("Whisper received no audio data")
        return "", ""
    audio_bytes = base64.b64decode(audio.split(',')[1])

    transcription = do_stt(audio_bytes, whipser_language)
    if auto_submit:
        input_hijack.update({"state": True, "value": [transcription, transcription]})
    return transcription


def reload_whispermodel(whisper_model_name: str, whisper_language: str, device: str):
    if len(whisper_model_name) > 0:
        global WHISPERMODEL
        WHISPERMODEL = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if device != "none":
            if device == "cuda":
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            WHISPERMODEL = whisper.load_model(whisper_model_name, device=device)
            params.update({"whipser_model": whisper_model_name})
            if ".en" in whisper_model_name:
                whisper_language = "english"
            audio_update = gr.Audio.update(interactive=True)
        else:
            audio_update = gr.Audio.update(interactive=False)
        return [whisper_model_name, whisper_language, str(device), audio_update]


def ui():
    with gr.Accordion("Whisper STT", open=True):
        with gr.Row():
            audio = gr.Textbox(elem_id="audio-base64", visible=False)
            record_button = gr.Button("Rec.", elem_id="record-button", elem_classes="custom-button")
        with gr.Row():
            with gr.Accordion("Settings", open=False):
                auto_submit = gr.Checkbox(label='Submit the transcribed audio automatically', value=params['auto_submit'])
                device_dropd = gr.Dropdown(label='Device', value=str(startup_device), choices=["cuda", "cpu", "none"])
                whisper_model_dropd = gr.Dropdown(label='Whisper Model', value=params['whipser_model'], choices=["tiny.en", "base.en", "small.en", "medium.en", "tiny", "base", "small", "medium", "large"])
                whisper_language = gr.Dropdown(label='Whisper Language', value=params['whipser_language'], choices=["english", "chinese", "german", "spanish", "russian", "korean", "french", "japanese", "portuguese", "turkish", "polish", "catalan", "dutch", "arabic", "swedish", "italian", "indonesian", "hindi", "finnish", "vietnamese", "hebrew", "ukrainian", "greek", "malay", "czech", "romanian", "danish", "hungarian", "tamil", "norwegian", "thai", "urdu", "croatian", "bulgarian", "lithuanian", "latin", "maori", "malayalam", "welsh", "slovak", "telugu", "persian", "latvian", "bengali", "serbian", "azerbaijani", "slovenian", "kannada", "estonian", "macedonian", "breton", "basque", "icelandic", "armenian", "nepali", "mongolian", "bosnian", "kazakh", "albanian", "swahili", "galician", "marathi", "punjabi", "sinhala", "khmer", "shona", "yoruba", "somali", "afrikaans", "occitan", "georgian", "belarusian", "tajik", "sindhi", "gujarati", "amharic", "yiddish", "lao", "uzbek", "faroese", "haitian creole", "pashto", "turkmen", "nynorsk", "maltese", "sanskrit", "luxembourgish", "myanmar", "tibetan", "tagalog", "malagasy", "assamese", "tatar", "hawaiian", "lingala", "hausa", "bashkir", "javanese", "sundanese"])

    audio.change(
        auto_transcribe, [audio, auto_submit, whisper_language], [shared.gradio['textbox']]).then(
        None, auto_submit, None, _js="(check) => {if (check) { document.getElementById('Generate').click() }}")

    device_dropd.input(reload_whispermodel, [whisper_model_dropd, whisper_language, device_dropd], [whisper_model_dropd, whisper_language, device_dropd, audio])
    whisper_model_dropd.change(reload_whispermodel, [whisper_model_dropd, whisper_language, device_dropd], [whisper_model_dropd, whisper_language, device_dropd, audio])
    whisper_language.change(lambda x: params.update({"whipser_language": x}), whisper_language, None)
    auto_submit.change(lambda x: params.update({"auto_submit": x}), auto_submit, None)


def custom_js():
    """
    Returns custom javascript as a string. It is applied whenever the web UI is
    loaded.
    :return:
    """
    with open(Path(__file__).parent.resolve() / "script.js", "r") as f:
        return f.read()
