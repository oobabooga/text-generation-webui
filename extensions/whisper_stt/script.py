import gradio as gr
import speech_recognition as sr
import modules.shared as shared

input_hijack = {
    'state': False,
    'value': ["", ""]
}


def input_modifier(string):
    return string


def do_stt():
    transcription = ""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something!")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)

    # recognize speech using whisper
    try:
        transcription = r.recognize_whisper(audio, language="english", model="tiny.en")
        print("Whisper thinks you said " + transcription)
    except sr.UnknownValueError:
        print("Whisper could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Whisper")

    # input_modifier(transcription)
    input_hijack.update({"state": True, "value": [transcription, transcription]})
    return transcription


def ui():
    speech_button = gr.Button(value="STT")
    output_transcription = gr.Textbox(label="Speech Preview")
    speech_button.click(do_stt, outputs=[output_transcription])
