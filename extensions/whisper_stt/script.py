import gradio as gr
import speech_recognition as sr
import numpy as np
import base64
import os
import whisper


from modules import shared

input_hijack = {
    'state': False,
    'value': ["", ""]
}

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

def do_stt(audio, whisper_model, whisper_language):
    print(f"Attempting to transcribe with model {whisper_model} and language {whisper_language}")
    
    try:
        # Load Whisper model
        print("Loading Whisper model...")
        model = whisper.load_model(whisper_model)
        print("Whisper model loaded successfully")

        # Convert audio data to the format Whisper expects
        audio_np = audio[1].astype(np.float32) / 32768.0
        
        print(f"Audio data shape: {audio_np.shape}, Sample rate: {audio[0]}")
        
        # Transcribe
        print("Starting Whisper transcription...")
        result = model.transcribe(audio_np, language=whisper_language, fp16=False)
        transcription = result["text"]
        print("Whisper transcription completed")
        
        return transcription
    except Exception as e:
        print(f"Error in do_stt: {str(e)}")
        import traceback
        traceback.print_exc()
        return ""

def auto_transcribe(audio_base64, auto_submit, whipser_model, whipser_language):
    print("auto_transcribe called")
    print(f"auto_submit: {auto_submit}")
    print(f"whipser_model: {whipser_model}")
    print(f"whipser_language: {whipser_language}")
    
    if audio_base64 is None or audio_base64 == "":
        print("No audio data received")
        return "", ""
    
    try:
        # Decode base64 audio data
        audio_bytes = base64.b64decode(audio_base64.split(',')[1])
        print(f"Decoded audio bytes length: {len(audio_bytes)}")
        
        print("Processing audio...")
        # Convert WebM to PCM using ffmpeg
        import subprocess
        
        command = ['ffmpeg', '-i', 'pipe:0', '-ar', '16000', '-ac', '1', '-f', 's16le', '-loglevel', 'error', '-']
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate(input=audio_bytes)
        
        if error:
            print("FFmpeg error:", error.decode())
        
        # Convert to numpy array
        audio_np = np.frombuffer(output, dtype=np.int16)
        
        # Use 16kHz sample rate for Whisper
        sample_rate = 16000
        audio = (sample_rate, audio_np)
        
        transcription = do_stt(audio, whipser_model, whipser_language)
        print(f"Transcription: {transcription}")
        
        if auto_submit:
            input_hijack.update({"state": True, "value": [transcription, transcription]})

        return transcription, None
    except Exception as e:
        print(f"Error in auto_transcribe: {str(e)}")
        import traceback
        traceback.print_exc()
        return "", None
    

def ui():
    with gr.Accordion("Whisper STT", open=True):
        audio_base64 = gr.Textbox(elem_id="audio-base64", visible=False)
        with gr.Row():
            with gr.Accordion("Settings", open=False):
                auto_submit = gr.Checkbox(label='Submit the transcribed audio automatically', value=params['auto_submit'])
                whipser_model = gr.Dropdown(
                    label='Whisper Model', 
                    value=params['whipser_model'], 
                    choices=["tiny.en", "base.en", "small.en", "medium.en", "tiny", "base", "small", "medium", "large"]
                )
                whipser_language = gr.Dropdown(
                    label='Whisper Language', 
                    value=params['whipser_language'], 
                    choices=["english", "german", "french", "japanese", "spanish", "russian", "chinese", "korean",  "portuguese", "turkish", "polish", "catalan", "dutch", "arabic", "swedish", "italian", "indonesian", "hindi", "finnish", "vietnamese", "hebrew", "ukrainian", "greek", "malay", "czech", "romanian", "danish", "hungarian", "tamil", "norwegian", "thai", "urdu", "croatian", "bulgarian", "lithuanian", "latin", "maori", "malayalam", "welsh", "slovak", "telugu", "persian", "latvian", "bengali", "serbian", "azerbaijani", "slovenian", "kannada", "estonian", "macedonian", "breton", "basque", "icelandic", "armenian", "nepali", "mongolian", "bosnian", "kazakh", "albanian", "swahili", "galician", "marathi", "punjabi", "sinhala", "khmer", "shona", "yoruba", "somali", "afrikaans", "occitan", "georgian", "belarusian", "tajik", "sindhi", "gujarati", "amharic", "yiddish", "lao", "uzbek", "faroese", "haitian creole", "pashto", "turkmen", "nynorsk", "maltese", "sanskrit", "luxembourgish", "myanmar", "tibetan", "tagalog", "malagasy", "assamese", "tatar", "hawaiian", "lingala", "hausa", "bashkir", "javanese", "sundanese"]  # Add more languages as needed
                )

    audio_base64.change(
        auto_transcribe, 
        inputs=[audio_base64, auto_submit, whipser_model, whipser_language],
        outputs=[shared.gradio['textbox'], audio_base64]
    ).then(
        None, auto_submit, None, 
        _js="(check) => {if (check) { document.getElementById('Generate').click() }}"
    )

    whipser_model.change(lambda x: params.update({"whipser_model": x}), whipser_model, None)
    whipser_language.change(lambda x: params.update({"whipser_language": x}), whipser_language, None)
    auto_submit.change(lambda x: params.update({"auto_submit": x}), auto_submit, None)

def custom_js():
    js_file_path = os.path.join(os.path.dirname(__file__), "script.js")
    with open(js_file_path, "r") as js_file:
        return js_file.read()
