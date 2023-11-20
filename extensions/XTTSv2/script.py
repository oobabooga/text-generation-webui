from TTS.api import TTS
import os
import shutil
import json
import time
from pathlib import Path
import gradio as gr
import soundfile as sf
import numpy as np
from modules import shared

streaming_state = shared.args.no_stream

tts = None
this_dir = os.path.dirname(os.path.abspath(__file__))
params = json.load(open(f"{this_dir}/config.json"))
languages = params["available_languages"]
voice_presets = sorted(os.listdir(f"{this_dir}/voices"))
narrator_presets = ["None", "Skip"] + voice_presets


def preprocess(raw_input):
    raw_input = raw_input.replace("&amp;", "&")
    raw_input = raw_input.replace("&lt;", "<")
    raw_input = raw_input.replace("&gt;", ">")
    raw_input = raw_input.replace("&quot;", '"')
    raw_input = raw_input.replace("&#x27;", "'")
    raw_input = raw_input.strip("\"")
    return raw_input


def delete_old():
    shutil.rmtree(f"{this_dir}/generated")


def preprocess_narrator(raw_input):
    raw_input = preprocess(raw_input)
    raw_input = raw_input.replace("***", "*")
    raw_input = raw_input.replace("**", "*")
    narrated_text = raw_input.split("*")
    return raw_input, narrated_text


def combine(audiofiles):
    audio = np.array([])
    for audiofile in audiofiles:
        audio = np.concatenate((audio, sf.read(audiofile)[0]))
    return audio


def history_modifier(history):
    if len(history["internal"]) > 0:
        history["visible"][-1] = [
            history["visible"][-1][0],
            history["visible"][-1][1].replace(
                'controls autoplay style="height: 30px;">', 'controls style="height: 30px;">')
        ]
    return history


def format_html(audiofiles):
    if params["combine"]:
        autoplay = "autoplay" if params["autoplay"] else ""
        combined = combine(audiofiles)
        time_label = audiofiles[0].split("/")[-1].split("_")[0]
        sf.write(f"{this_dir}/generated/{time_label}_combined.wav",
                 combined, 24000)
        return f'<audio src="file/{this_dir}/generated/{time_label}_combined.wav" controls {autoplay} style="height: 30px;"></audio>'
    else:
        string = ""
        for audiofile in audiofiles:
            string += f'<audio src="file/{audiofile}" controls style="height: 30px;"></audio>'
    return string


def input_modifier(string):
    if not params["activate"]:
        shared.processing_message = "*Is typing...*"
        return string
    shared.processing_message = "*Is recording a voice message...*"
    shared.args.no_stream = True
    return string


def tts_char(string):
    global tts
    string = string
    if not params["activate"]:
        return string

    if tts is None:
        print("[XTTS] Loading XTTS...")
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

    ttstext = preprocess(string)
    time_label = int(time.time())
    tts.tts_to_file(text=ttstext,
                    file_path=f"{this_dir}/generated/{time_label}.wav",
                    speaker_wav=[f"{this_dir}/voices/{params['voice']}"],
                    language=languages[params["language"]])

    autoplay = "autoplay" if params["autoplay"] else ""

    string = f'<audio src="file/{this_dir}/generated/{time_label}.wav" controls {autoplay} style="height: 30px;"></audio><br>{ttstext}'
    if params["show_text"]:
        string += f"<br>{ttstext}"

    shared.args.no_stream = streaming_state
    return string


def tts_narrator(string):
    global tts
    string = string
    if not params["activate"]:
        return string

    if tts is None:
        print("[XTTS] Loading XTTS...")
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

    ttstext, turns = preprocess_narrator(string)
    voices = (params["voice"], params["narrator"])
    audiofiles = []
    time_label = int(time.time())
    for i, turn in enumerate(turns):
        if turn.strip() == "":
            continue
        voice = voices[i % 2]
        if voice == "Skip":
            continue
        tts.tts_to_file(text=turn,
                        file_path=f"{this_dir}/generated/{time_label}_{i:03d}.wav",
                        speaker_wav=[f"{this_dir}/voices/{voice}"],
                        language=languages[params["language"]])
        audiofiles.append(
            f"{this_dir}/generated/{time_label}_{i:03d}.wav")

    string = format_html(audiofiles)
    if params["show_text"]:
        string += f"<br>{ttstext}"
    shared.args.no_stream = streaming_state
    return string


def output_modifier(string):
    if params["narrator"] == "None":
        return tts_char(string)
    else:
        return tts_narrator(string)


def setup():
    global tts
    print("[XTTS] Loading XTTS...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    print("[XTTS] Done!")
    if params["delete"]:
        print("[XTTS] Deleting old generated files...")
        delete_old()
        print("[XTTS] Done!")
    print("[XTTS] Creating directories (if they don't exist)...")
    if not Path(f"{this_dir}/generated").exists():
        Path(f"{this_dir}/generated").mkdir(parents=True)
    print("[XTTS] Done!")


def ui():
    with gr.Accordion("XTTS"):
        with gr.Row():
            activate = gr.Checkbox(
                value=params["activate"], label="Activate TTS")
            autoplay = gr.Checkbox(value=params["autoplay"], label="Autoplay")
            show_text = gr.Checkbox(
                value=params["show_text"], label="Show text")
            combine_audio = gr.Checkbox(
                value=params["combine"], label="Combine audio")
        with gr.Row():
            voice = gr.Dropdown(
                voice_presets, label="Voice Wav", value=params["voice"])
            narrator = gr.Dropdown(
                narrator_presets, label="Narrator Wav", value=params["narrator"])
            language = gr.Dropdown(
                languages.keys(), label="Language", value=params["language"])

    activate.change(lambda x: params.update({"activate": x}), activate, None)
    autoplay.change(lambda x: params.update({"autoplay": x}), autoplay, None)
    show_text.change(lambda x: params.update(
        {"show_text": x}), show_text, None)
    combine_audio.change(lambda x: params.update(
        {"combine": x}), combine_audio, None)

    voice.change(lambda x: params.update({"voice": x}), voice, None)
    narrator.change(lambda x: params.update({"narrator": x}), narrator, None)
    language.change(lambda x: params.update({"language": x}), language, None)
