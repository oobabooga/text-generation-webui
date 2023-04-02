import gradio as gr
import speech_recognition as sr

input_hijack = {
    'state': False,
    'value': ["", ""]
}


def do_stt(audio, text_state=""):
    transcription = ""
    r = sr.Recognizer()

    # Convert to AudioData
    audio_data = sr.AudioData(sample_rate=audio[0], frame_data=audio[1], sample_width=4)

    try:
        transcription = r.recognize_whisper(audio_data, language="english", model="base.en")
    except sr.UnknownValueError:
        print("Whisper could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Whisper", e)

    input_hijack.update({"state": True, "value": [transcription, transcription]})

    text_state += transcription + " "
    return text_state, text_state


def update_hijack(val):
    input_hijack.update({"state": True, "value": [val, val]})
    return val


def auto_transcribe(audio, audio_auto, text_state=""):
    if audio is None:
        return "", ""
    if audio_auto:
        return do_stt(audio, text_state)
    return "", ""


def ui():
    tr_state = gr.State(value="")
    output_transcription = gr.Textbox(label="STT-Input",
                                      placeholder="Speech Preview. Click \"Generate\" to send",
                                      interactive=True)
    output_transcription.change(fn=update_hijack, inputs=[output_transcription], outputs=[tr_state])
    audio_auto = gr.Checkbox(label="Auto-Transcribe", value=True)
    with gr.Row():
        audio = gr.Audio(source="microphone")
        audio.change(fn=auto_transcribe, inputs=[audio, audio_auto, tr_state], outputs=[output_transcription, tr_state])
        transcribe_button = gr.Button(value="Transcribe")
        transcribe_button.click(do_stt, inputs=[audio, tr_state], outputs=[output_transcription, tr_state])
