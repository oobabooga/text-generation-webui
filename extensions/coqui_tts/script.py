import time
import traceback
from pathlib import Path

import gradio as gr

from modules import chat, shared, tts_preprocessor
from modules.utils import gradio

from TTS.api import TTS
import TTS.utils.synthesizer


# Running a multi-speaker and multilingual model
params = {
    'activate': True,
    'speaker': None,
    'language': None,
    'model_id': 'tts_models/en/ljspeech/tacotron2-DDC',
    'voice_clone_reference_path': None,
    'Cuda': True,
    'show_text': True,
    'autoplay': True,
}

current_params = params.copy()
speakers = []
languages = []
models = [
    {'name': 'multi/your_tts', 'value': 'tts_models/multilingual/multi-dataset/your_tts'},
    {'name': 'ek1/tacotron2', 'value': 'tts_models/en/ek1/tacotron2'},
    {'name': 'ljspeech/tacotron2-DDC', 'value': 'tts_models/en/ljspeech/tacotron2-DDC'},
    {'name': 'ljspeech/tacotron2-DDC_ph', 'value': 'tts_models/en/ljspeech/tacotron2-DDC_ph'},
    {'name': 'ljspeech/glow-tts', 'value': 'tts_models/en/ljspeech/glow-tts'},
    {'name': 'ljspeech/speedy-speech', 'value': 'tts_models/en/ljspeech/speedy-speech'},
    {'name': 'ljspeech/tacotron2-DCA', 'value': 'tts_models/en/ljspeech/tacotron2-DCA'},
    {'name': 'ljspeech/vits (espeak)', 'value': 'tts_models/en/ljspeech/vits'},
    {'name': 'ljspeech/vits--neon (espeak)', 'value': 'tts_models/en/ljspeech/vits--neon'},
    {'name': 'ljspeech/fast_pitch', 'value': 'tts_models/en/ljspeech/fast_pitch'},
    {'name': 'ljspeech/overflow (espeak)', 'value': 'tts_models/en/ljspeech/overflow'},
    {'name': 'ljspeech/neural_hmm (espeak)', 'value': 'tts_models/en/ljspeech/neural_hmm'},
    {'name': 'vctk/vits (espeak)', 'value': 'tts_models/en/vctk/vits'},
    {'name': 'vctk/fast_pitch', 'value': 'tts_models/en/vctk/fast_pitch'},
    {'name': 'sam/tacotron-DDC (espeak)', 'value': 'tts_models/en/sam/tacotron-DDC'},
    {'name': 'blizzard2013/capacitron-t2-c50 (espeak)', 'value': 'tts_models/en/blizzard2013/capacitron-t2-c50'},
    {'name': 'blizzard2013/capacitron-t2-c150_v2 (espeak)', 'value': 'tts_models/en/blizzard2013/capacitron-t2-c150_v2'}
]

model_values = [i['value'] for i in models]
model_choices = [i['name'] for i in models]


def load_model():
    # Init TTS
    global speakers, languages
    tts = TTS(params['model_id'], gpu=params['Cuda'])
    if tts is not None and tts.synthesizer is not None and tts.synthesizer.tts_config is not None and hasattr(
            tts.synthesizer.tts_config, 'num_chars'):
        tts.synthesizer.tts_config.num_chars = 250

    speakers = tts.speakers if tts.speakers is not None else []
    temp_speaker = params['speaker'] if params['speaker'] in speakers else speakers[0] if len(speakers) > 0 else None

    languages = tts.languages if tts.languages is not None else []
    temp_language = params['language'] if params['language'] in languages else languages[0] if len(
        languages) > 0 else None

    return tts, temp_speaker, temp_language


model, speaker, language = load_model()
streaming_state = shared.args.no_stream  # remember if chat streaming was enabled


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


def history_modifier(history):
    # Remove autoplay from the last reply
    if len(history['internal']) > 0:
        history['visible'][-1] = [
            history['visible'][-1][0],
            history['visible'][-1][1].replace('controls autoplay>', 'controls>')
        ]

    return history


def output_modifier(string, state):
    global model, speaker, language, current_params

    for i in params:
        if params[i] != current_params[i]:
            model, speaker, language = load_model()
            current_params = params.copy()
            break

    if not current_params['activate']:
        return string

    original_string = string
    # we don't need to handle numbers. The text normalizer in coqui does it better
    string = tts_preprocessor.replace_invalid_chars(string)
    string = tts_preprocessor.replace_abbreviations(string)
    string = tts_preprocessor.clean_whitespace(string)

    if string == '':
        string = '*Empty reply, try regenerating*'
    else:
        output_file = Path(f'extensions/coqui_tts/outputs/{state["character_menu"]}_{int(time.time())}.wav')
        if params['voice_clone_reference_path'] is not None:
            model.tts_with_vc_to_file(text=string, language=language, speaker_wav=params['voice_clone_reference_path'],
                                      file_path=str(output_file))
        else:
            model.tts_to_file(text=string, speaker=speaker, language=language, file_path=str(output_file))

        autoplay = 'autoplay' if current_params['autoplay'] else ''
        string = f'<audio src="file/{output_file.as_posix()}" controls {autoplay}></audio>'
        if params['show_text']:
            string += f'\n\n{original_string}'

    shared.processing_message = "*Is typing...*"
    return string


def setup():
    global model, speaker, language
    model, speaker, language = load_model()


def ui():
    # Gradio elements
    with gr.Accordion("Coqui AI TTS"):
        with gr.Row():
            activate = gr.Checkbox(value=params['activate'], label='Activate TTS')
            autoplay = gr.Checkbox(value=params['autoplay'], label='Play TTS automatically')

        show_text = gr.Checkbox(value=params['show_text'], label='Show message text under audio player')
        model_dropdown = gr.Dropdown(
            value=model_choices[model_values.index(params['model_id'])] if params['model_id'] in model_values else None,
            choices=model_choices, type='index', label='Model')
        speaker_dropdown = gr.Dropdown(value=params['speaker'],
                                       choices=model.speakers if model.speakers is not None else [], label='Speaker')
        language_dropdown = gr.Dropdown(value=params['language'],
                                        choices=model.languages if model.languages is not None else [],
                                        label='Language')
        vc_textbox = gr.Textbox(value=params['voice_clone_reference_path'], label='Voice Clone Speaker Path')

        with gr.Row():
            convert = gr.Button('Permanently replace audios with the message texts')
            convert_cancel = gr.Button('Cancel', visible=False)
            convert_confirm = gr.Button('Confirm (cannot be undone)', variant="stop", visible=False)

    # Convert history with confirmation
    convert_arr = [convert_confirm, convert, convert_cancel]
    convert.click(lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)], None,
                  convert_arr)
    convert_confirm.click(lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)], None,
                          convert_arr)
    convert_confirm.click(remove_tts_from_history, [shared.gradio[k] for k in ['name1', 'name2', 'mode']],
                          shared.gradio['display'])
    convert_confirm.click(lambda: chat.save_history(mode='chat', timestamp=False), [], [], show_progress=False)
    convert_cancel.click(lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)], None,
                         convert_arr)

    # Toggle message text in history
    show_text.change(lambda x: params.update({"show_text": x}), show_text, None)
    show_text.change(toggle_text_in_history, [shared.gradio[k] for k in ['name1', 'name2', 'mode']],
                     shared.gradio['display'])
    show_text.change(lambda: chat.save_history(mode='chat', timestamp=False), [], [], show_progress=False)

    # Event functions to update the parameters in the backend
    activate.change(lambda x: params.update({"activate": x}), activate, None)
    autoplay.change(lambda x: params.update({"autoplay": x}), autoplay, None)
    model_dropdown.change(lambda x: update_model(x), model_dropdown, [speaker_dropdown, language_dropdown])
    speaker_dropdown.change(lambda x: params.update({"speaker": x}), speaker_dropdown, None)
    language_dropdown.change(lambda x: params.update({"language": x}), language_dropdown, None)
    vc_textbox.change(lambda x: params.update({"voice_clone_reference_path": x}), vc_textbox, None)


def update_model(x):
    model_id = model_values[x]
    params.update({"model_id": model_id})
    global model, speaker, language, speakers, languages
    try:
        model, speaker, language = load_model()
    except:
        print(traceback.format_exc())
    return [gr.update(value=speaker, choices=speakers), gr.update(value=language, choices=languages)]
