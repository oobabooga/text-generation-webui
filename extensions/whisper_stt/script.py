import gradio as gr
import speech_recognition as sr

from modules import shared

input_hijack = {
    'state': False,
    'value': ["", ""]
}

# parameters which can be customized in settings.json of webui
params = {
    'whipser_language': 'english',
    'whipser_model': 'small.en',
    'auto_submit': True
}


def chat_input_modifier(text, visible_text, state):
    global input_hijack
    if input_hijack['state']:
        input_hijack['state'] = False
        return input_hijack['value']
    else:
        return text, visible_text


def do_stt(audio, whipser_model, whipser_language):
    transcription = ""
    r = sr.Recognizer()

    # Convert to AudioData
    audio_data = sr.AudioData(sample_rate=audio[0], frame_data=audio[1], sample_width=4)

    try:
        transcription = r.recognize_whisper(audio_data, language=whipser_language, model=whipser_model)
    except sr.UnknownValueError:
        print("Whisper could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Whisper", e)

    return transcription


def auto_transcribe(audio, auto_submit, whipser_model, whipser_language):
    if audio is None:
        return "", ""
    transcription = do_stt(audio, whipser_model, whipser_language)
    if auto_submit:
        input_hijack.update({"state": True, "value": [transcription, transcription]})

    return transcription, None


def ui():
    with gr.Accordion("Whisper STT", open=True):
        with gr.Row():
            audio = gr.Audio(source="microphone")
        with gr.Row():
            with gr.Accordion("Settings", open=False):
                auto_submit = gr.Checkbox(label='Submit the transcribed audio automatically', value=params['auto_submit'])
                whipser_model = gr.Dropdown(label='Whisper Model', value=params['whipser_model'], choices=["tiny.en", "base.en", "small.en", "medium.en", "tiny", "base", "small", "medium", "large"])
                whipser_language = gr.Dropdown(label='Whisper Language', value=params['whipser_language'], choices=["chinese", "german", "spanish", "russian", "korean", "french", "japanese", "portuguese", "turkish", "polish", "catalan", "dutch", "arabic", "swedish", "italian", "indonesian", "hindi", "finnish", "vietnamese", "hebrew", "ukrainian", "greek", "malay", "czech", "romanian", "danish", "hungarian", "tamil", "norwegian", "thai", "urdu", "croatian", "bulgarian", "lithuanian", "latin", "maori", "malayalam", "welsh", "slovak", "telugu", "persian", "latvian", "bengali", "serbian", "azerbaijani", "slovenian", "kannada", "estonian", "macedonian", "breton", "basque", "icelandic", "armenian", "nepali", "mongolian", "bosnian", "kazakh", "albanian", "swahili", "galician", "marathi", "punjabi", "sinhala", "khmer", "shona", "yoruba", "somali", "afrikaans", "occitan", "georgian", "belarusian", "tajik", "sindhi", "gujarati", "amharic", "yiddish", "lao", "uzbek", "faroese", "haitian creole", "pashto", "turkmen", "nynorsk", "maltese", "sanskrit", "luxembourgish", "myanmar", "tibetan", "tagalog", "malagasy", "assamese", "tatar", "hawaiian", "lingala", "hausa", "bashkir", "javanese", "sundanese"])

    audio.change(
        auto_transcribe, [audio, auto_submit, whipser_model, whipser_language], [shared.gradio['textbox'], audio]).then(
        None, auto_submit, None, _js="(check) => {if (check) { document.getElementById('Generate').click() }}")

    whipser_model.change(lambda x: params.update({"whipser_model": x}), whipser_model, None)
    whipser_language.change(lambda x: params.update({"whipser_language": x}), whipser_language, None)
    auto_submit.change(lambda x: params.update({"auto_submit": x}), auto_submit, None)
