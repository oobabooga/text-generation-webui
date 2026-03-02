import html
import json
import os
import time
from pathlib import Path

import gradio as gr

from modules import chat, shared, ui_chat
from modules.utils import gradio

# CAMB AI SDK imports (deferred to setup() for graceful error handling)
CambAI = None
save_stream_to_file = None

params = {
    # Shared
    'api_key': '',

    # TTS
    'tts_activate': True,
    'tts_autoplay': True,
    'tts_show_text': False,
    'tts_language': 'English',
    'tts_voice_id': 147320,
    'tts_speech_model': 'mars-pro',
    'tts_output_format': 'wav',

    # Translation
    'translate_activate': False,
    'translate_source_language': 'English',
    'translate_target_language': 'English',
}

client = None

with open(Path(__file__).parent.resolve() / "camb_languages.json", encoding='utf8') as f:
    languages = json.load(f)

SPEECH_MODELS = ['mars-pro', 'mars-flash', 'mars-instruct']
OUTPUT_FORMATS = ['wav', 'mp3', 'flac']
OUTPUTS_DIR = Path('extensions/camb_ai/outputs')


def _get_api_key():
    """Resolve API key from params, env var, or settings (3-tier)."""
    key = params.get('api_key', '').strip()
    if key:
        return key
    return os.environ.get('CAMB_API_KEY', '').strip() or None


def _init_client():
    """Initialize or reinitialize the CAMB AI client."""
    global client
    api_key = _get_api_key()
    if not api_key:
        client = None
        return False

    try:
        client = CambAI(api_key=api_key)
        return True
    except Exception as e:
        print(f"CAMB AI: Failed to initialize client: {e}")
        client = None
        return False


def _get_lang_id(name):
    """Get CAMB language integer ID from language name."""
    entry = languages.get(name)
    return entry['id'] if entry else 1


def _get_tts_locale(name):
    """Get TTS locale string from language name."""
    entry = languages.get(name)
    return entry['tts_locale'] if entry else 'en-us'


def _extract_translation_text(result):
    """Extract translated text from translation_stream() result."""
    if isinstance(result, str):
        return result
    for attr in ('text', 'translation', 'translated_text'):
        if hasattr(result, attr):
            return str(getattr(result, attr))
    return str(result)


# ---------------------------------------------------------------------------
# Extension hooks
# ---------------------------------------------------------------------------

def setup():
    global CambAI, save_stream_to_file

    from camb.client import CambAI as _CambAI
    from camb.client import save_stream_to_file as _save

    CambAI = _CambAI
    save_stream_to_file = _save

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    _init_client()

    if client:
        print("CAMB AI: Extension loaded successfully.")
    else:
        print("CAMB AI: No API key configured. Set it in the UI, settings.yaml (camb_ai-api_key), or CAMB_API_KEY env var.")


def state_modifier(state):
    if not params['tts_activate']:
        return state

    state['stream'] = False
    return state


def input_modifier(string, state, is_chat=False):
    if params['tts_activate']:
        shared.processing_message = "*Is generating a voice message...*"

    if params['translate_activate'] and client and string.strip():
        src_id = _get_lang_id(params['translate_source_language'])
        if src_id != 1:  # Only translate if source is not English
            try:
                result = client.translation.translation_stream(
                    source_language=src_id,
                    target_language=1,  # English
                    text=string,
                )
                if result:
                    string = _extract_translation_text(result)
            except Exception as e:
                print(f"CAMB AI Translation (input): {e}")

    return string


def history_modifier(history):
    # Remove autoplay from the last reply so only the newest plays
    if len(history['internal']) > 0:
        history['visible'][-1] = [
            history['visible'][-1][0],
            history['visible'][-1][1].replace('controls autoplay>', 'controls>')
        ]
    return history


def output_modifier(string, state, is_chat=False):
    if not params['tts_activate'] and not params['translate_activate']:
        shared.processing_message = "*Is typing...*"
        return string

    original_string = string

    # Translation: LLM output (English) -> target language
    if params['translate_activate'] and client and string.strip():
        tgt_id = _get_lang_id(params['translate_target_language'])
        if tgt_id != 1:  # Only translate if target is not English
            try:
                result = client.translation.translation_stream(
                    source_language=1,  # English
                    target_language=tgt_id,
                    text=html.unescape(string),
                )
                if result:
                    string = _extract_translation_text(result)
                    original_string = string
            except Exception as e:
                print(f"CAMB AI Translation (output): {e}")

    # TTS: synthesize speech from text
    if params['tts_activate'] and client:
        text_for_tts = html.unescape(string) if string.strip() else ''
        if text_for_tts == '':
            string = '*Empty reply, try regenerating*'
        else:
            try:
                locale = _get_tts_locale(params['tts_language'])
                output_file = OUTPUTS_DIR / f"{state.get('character_menu', 'char')}_{int(time.time())}.{params['tts_output_format']}"

                tts_kwargs = dict(
                    text=text_for_tts,
                    language=locale,
                    voice_id=int(params['tts_voice_id']),
                    speech_model=params['tts_speech_model'],
                )

                stream = client.text_to_speech.tts(**tts_kwargs)
                save_stream_to_file(stream, str(output_file))

                autoplay = 'autoplay' if params['tts_autoplay'] else ''
                string = f'<audio src="file/{output_file.as_posix()}" controls {autoplay}></audio>'
                if params['tts_show_text']:
                    string += f'\n\n{original_string}'
            except Exception as e:
                print(f"CAMB AI TTS error: {e}")
                string = original_string

    shared.processing_message = "*Is typing...*"
    return string


def custom_css():
    with open(Path(__file__).parent.resolve() / "style.css", "r") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def remove_tts_from_history(history):
    for i, entry in enumerate(history['internal']):
        history['visible'][i] = [history['visible'][i][0], entry[1]]
    return history


def toggle_text_in_history(history):
    for i, entry in enumerate(history['visible']):
        visible_reply = entry[1]
        if visible_reply.startswith('<audio'):
            if params['tts_show_text']:
                reply = history['internal'][i][1]
                history['visible'][i] = [
                    history['visible'][i][0],
                    f"{visible_reply.split('</audio>')[0]}</audio>\n\n{reply}"
                ]
            else:
                history['visible'][i] = [
                    history['visible'][i][0],
                    f"{visible_reply.split('</audio>')[0]}</audio>"
                ]
    return history


def voice_preview(text):
    if not client:
        return "CAMB AI: No API key configured."
    if not text or not text.strip():
        text = "Hello! This is a voice preview from CAMB AI."

    try:
        locale = _get_tts_locale(params['tts_language'])
        output_file = OUTPUTS_DIR / f"voice_preview.{params['tts_output_format']}"

        stream = client.text_to_speech.tts(
            text=text,
            language=locale,
            voice_id=int(params['tts_voice_id']),
            speech_model=params['tts_speech_model'],
        )
        save_stream_to_file(stream, str(output_file))
        return f'<audio src="file/{output_file.as_posix()}?{int(time.time())}" controls autoplay></audio>'
    except Exception as e:
        return f"CAMB AI TTS preview error: {e}"


def clone_voice(voice_name, gender, audio_file):
    if not client:
        return "No API key configured."
    if not voice_name or not voice_name.strip():
        return "Please enter a voice name."
    if audio_file is None:
        return "Please upload an audio file."

    try:
        gender_int = 1 if gender == "Male" else 2
        with open(audio_file, "rb") as f:
            result = client.voice_cloning.create_custom_voice(
                voice_name=voice_name.strip(),
                gender=gender_int,
                file=f,
            )
        return f"Voice cloned successfully! Voice ID: {result.voice_id}"
    except Exception as e:
        return f"Voice cloning failed: {e}"


def refresh_voices():
    if not client:
        return gr.update(choices=[])
    try:
        voices = client.voice_cloning.list_voices()
        choices = []
        for v in voices:
            if isinstance(v, dict):
                vid = v.get('id', '')
                name = v.get('voice_name', str(vid))
                choices.append((f"{name} ({vid})", str(vid)))
            else:
                choices.append((f"{v.voice_name} ({v.id})", str(v.id)))
        return gr.update(choices=choices)
    except Exception as e:
        print(f"CAMB AI: Failed to list voices: {e}")
        return gr.update(choices=[])


def on_api_key_change(key):
    params.update({"api_key": key})
    _init_client()
    if client:
        return "API key set successfully."
    elif key.strip():
        return "Failed to initialize client with this key."
    else:
        return ""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def ui():
    lang_names = sorted(languages.keys())

    with gr.Accordion("CAMB AI", open=True):
        # --- API Key ---
        with gr.Row():
            api_key_input = gr.Textbox(
                value=params['api_key'],
                label='CAMB AI API Key',
                type='password',
                placeholder='sk-... (or set CAMB_API_KEY env var)',
            )
            api_key_status = gr.Textbox(value='', label='Status', interactive=False)

        # --- TTS ---
        with gr.Accordion("Text-to-Speech", open=True):
            with gr.Row():
                tts_activate = gr.Checkbox(value=params['tts_activate'], label='Activate TTS')
                tts_autoplay = gr.Checkbox(value=params['tts_autoplay'], label='Autoplay')

            tts_show_text = gr.Checkbox(value=params['tts_show_text'], label='Show text under audio player')

            with gr.Row():
                tts_language = gr.Dropdown(value=params['tts_language'], choices=lang_names, label='Language')
                tts_speech_model = gr.Dropdown(value=params['tts_speech_model'], choices=SPEECH_MODELS, label='Speech Model')

            with gr.Row():
                tts_voice_id = gr.Number(value=params['tts_voice_id'], label='Voice ID', precision=0)
                tts_voice_dropdown = gr.Dropdown(choices=[], label='Voice List (click Refresh)', allow_custom_value=True)
                tts_output_format = gr.Dropdown(value=params['tts_output_format'], choices=OUTPUT_FORMATS, label='Output Format')

            with gr.Row():
                preview_text = gr.Textbox(show_label=False, placeholder="Preview text", elem_id="camb_preview_text")
                preview_button = gr.Button("Preview")
            preview_audio = gr.HTML(visible=True, elem_classes="camb-ai-preview")

            with gr.Row():
                convert = gr.Button('Permanently replace audios with the message texts')
                convert_cancel = gr.Button('Cancel', visible=False)
                convert_confirm = gr.Button('Confirm (cannot be undone)', variant="stop", visible=False)

        # --- Translation ---
        with gr.Accordion("Translation", open=False):
            translate_activate = gr.Checkbox(value=params['translate_activate'], label='Activate Translation')
            with gr.Row():
                translate_source = gr.Dropdown(value=params['translate_source_language'], choices=lang_names, label='Source Language (your input)')
                translate_target = gr.Dropdown(value=params['translate_target_language'], choices=lang_names, label='Target Language (LLM output)')

        # --- Voice Cloning ---
        with gr.Accordion("Voice Cloning", open=False):
            with gr.Row():
                clone_name = gr.Textbox(label='Voice Name', placeholder='My Custom Voice')
                clone_gender = gr.Dropdown(value='Male', choices=['Male', 'Female'], label='Gender')
            clone_audio = gr.Audio(label='Upload audio file (2s+ recommended)', type='filepath')
            clone_button = gr.Button("Clone Voice")
            clone_status = gr.Textbox(label='Status', interactive=False)
            refresh_button = gr.Button("Refresh Voice List")

    # --- Event handlers ---

    # API Key
    api_key_input.change(on_api_key_change, api_key_input, api_key_status)

    # TTS
    tts_activate.change(lambda x: params.update({"tts_activate": x}), tts_activate, None)
    tts_autoplay.change(lambda x: params.update({"tts_autoplay": x}), tts_autoplay, None)
    tts_language.change(lambda x: params.update({"tts_language": x}), tts_language, None)
    tts_speech_model.change(lambda x: params.update({"tts_speech_model": x}), tts_speech_model, None)
    tts_voice_id.change(lambda x: params.update({"tts_voice_id": int(x)}) if x is not None else None, tts_voice_id, None)
    tts_output_format.change(lambda x: params.update({"tts_output_format": x}), tts_output_format, None)

    tts_show_text.change(
        lambda x: params.update({"tts_show_text": x}), tts_show_text, None).then(
        toggle_text_in_history, gradio('history'), gradio('history')).then(
        chat.save_history, gradio('history', 'unique_id', 'character_menu', 'mode'), None).then(
        chat.redraw_html, gradio(ui_chat.reload_arr), gradio('display'))

    # Convert history with confirmation
    convert_arr = [convert_confirm, convert, convert_cancel]
    convert.click(
        lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)],
        None, convert_arr)
    convert_confirm.click(
        lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)],
        None, convert_arr).then(
        remove_tts_from_history, gradio('history'), gradio('history')).then(
        chat.save_history, gradio('history', 'unique_id', 'character_menu', 'mode'), None).then(
        chat.redraw_html, gradio(ui_chat.reload_arr), gradio('display'))
    convert_cancel.click(
        lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)],
        None, convert_arr)

    # Preview
    preview_text.submit(voice_preview, preview_text, preview_audio)
    preview_button.click(voice_preview, preview_text, preview_audio)

    # Translation
    translate_activate.change(lambda x: params.update({"translate_activate": x}), translate_activate, None)
    translate_source.change(lambda x: params.update({"translate_source_language": x}), translate_source, None)
    translate_target.change(lambda x: params.update({"translate_target_language": x}), translate_target, None)

    # Voice dropdown -> voice ID sync
    tts_voice_dropdown.change(
        lambda x: (params.update({"tts_voice_id": int(x)}), int(x))[1] if x else params['tts_voice_id'],
        tts_voice_dropdown, tts_voice_id)

    # Voice Cloning
    clone_button.click(clone_voice, [clone_name, clone_gender, clone_audio], clone_status)
    refresh_button.click(refresh_voices, None, tts_voice_dropdown)
