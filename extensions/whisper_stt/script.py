import gradio as gr
import speech_recognition as sr

input_hijack = {
    'state': False,
    'value': ["", ""]
}


def do_stt():
    transcription = ""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)

    try:
        transcription = r.recognize_whisper(audio, language="english", model="tiny.en")
    except sr.UnknownValueError:
        print("Whisper could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Whisper", e)

    input_hijack.update({"state": True, "value": [transcription, transcription]})
    return transcription


def update_hijack(val):
    input_hijack.update({"state": True, "value": [val, val]})
    return val


def ui():
    speech_button = gr.Button(value="🎙️")
    output_transcription = gr.Textbox(label="STT-Input", placeholder="Speech Preview. Click \"Generate\" to send", interactive=True)
    output_transcription.change(fn=update_hijack, inputs=[output_transcription])
    speech_button.click(do_stt, outputs=[output_transcription])
