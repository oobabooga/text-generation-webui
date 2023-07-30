import re
from pathlib import Path
from scipy.io import wavfile
import gradio as gr
from modules import chat, shared
from extensions.中文朗读.vits import (
    vits,
    vitsNoiseScale,
    vitsNoiseScaleW,
    vitsLengthScale,
    speakers,
)

params = {
    "activate": True,
    "selected_voice": "None",
    "autoplay": False,
    "show_text": True,
}

voices = None
wav_idx = 0


def refresh_voices():
    return speakers


def refresh_voices_dd():
    all_voices = refresh_voices()
    return gr.Dropdown.update(value=all_voices[0], choices=all_voices)


def toggle_text_in_history():
    for i, entry in enumerate(shared.history["visible"]):
        visible_reply = entry[1]
        if visible_reply.startswith("<audio"):
            if params["show_text"]:
                reply = shared.history["internal"][i][1]
                shared.history["visible"][i] = [
                    shared.history["visible"][i][0],
                    f"{visible_reply.split('</audio>')[0]}</audio>\n\n{reply}",
                ]
            else:
                shared.history["visible"][i] = [
                    shared.history["visible"][i][0],
                    f"{visible_reply.split('</audio>')[0]}</audio>",
                ]


def remove_surrounded_chars(string):
    # this expression matches to 'as few symbols as possible (0 upwards) between any asterisks' OR
    # 'as few symbols as possible (0 upwards) between an asterisk and the end of the string'
    return re.sub("\*[^\*]*?(\*|$)", "", string)


def state_modifier(state):
    if not params["activate"]:
        return state

    state["stream"] = False
    return state


def input_modifier(string):
    if not params["activate"]:
        return string

    shared.processing_message = "*Is recording a voice message...*"
    return string


def history_modifier(history):
    # Remove autoplay from the last reply
    if len(history["internal"]) > 0:
        history["visible"][-1] = [
            history["visible"][-1][0],
            history["visible"][-1][1].replace("controls autoplay>", "controls>"),
        ]

    return history


def output_modifier(input_str):
    global params, wav_idx

    if not params["activate"]:
        return input_str

    original_string = input_str
    input_str = remove_surrounded_chars(input_str)
    input_str = input_str.replace('"', "")
    input_str = input_str.replace("“", "")
    input_str = input_str.replace("\n", " ")
    input_str = input_str.strip()
    if input_str == "":
        input_str = "empty reply, try regenerating"

    output_file = Path(f"extensions/中文朗读/outputs/{wav_idx:06d}.wav".format(wav_idx))
    print(f"Outputting audio to {str(output_file)}")
    try:
        status, audios, time = vits(
            input_str,
            speakers.index(params["selected_voice"]),
            100,
            vitsNoiseScale,
            vitsNoiseScaleW,
            vitsLengthScale,
        )
        print("VITS : ", status, time)
        wavfile.write(output_file, audios[0], audios[1])

        autoplay = "autoplay" if params["autoplay"] else ""
        string = (
            f'<audio src="file/{output_file.as_posix()}" controls {autoplay}></audio>'
        )
        wav_idx += 1
    except Exception as err:
        string = f"🤖 ElevenLabs Error: {err}\n\n"

    if params["show_text"]:
        string += f"\n\n{original_string}"

    shared.processing_message = "*Is typing...*"
    return string


def ui():
    global voices
    if not voices:
        voices = refresh_voices()
        params["selected_voice"] = voices[0]

    # Gradio elements
    with gr.Row():
        activate = gr.Checkbox(value=params["activate"], label="开启语音")
        autoplay = gr.Checkbox(value=params["autoplay"], label="自动播放")
        show_text = gr.Checkbox(value=params["show_text"], label="在音频下方显示文本")

    with gr.Row():
        voice = gr.Dropdown(
            value=params["selected_voice"], choices=voices, label="声音角色"
        )
        refresh = gr.Button(value="刷新")

    # Toggle message text in history
    show_text.change(lambda x: params.update({"show_text": x}), show_text, None).then(
        toggle_text_in_history, None, None
    ).then(chat.save_history, shared.gradio["mode"], None, show_progress=False).then(
        chat.redraw_html, shared.reload_inputs, shared.gradio["display"]
    )

    # Event functions to update the parameters in the backend
    activate.change(lambda x: params.update({"activate": x}), activate, None)
    voice.change(lambda x: params.update({"selected_voice": x}), voice, None)

    # connect.click(check_valid_api, [], connection_status)
    refresh.click(refresh_voices_dd, [], voice)
    # Event functions to update the parameters in the backend
    autoplay.change(lambda x: params.update({"autoplay": x}), autoplay, None)
