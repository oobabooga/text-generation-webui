from pathlib import Path

import gradio as gr
import torch

import modules.shared as shared
import simpleaudio as sa

torch._C._jit_set_profiling_mode(False)

params = {
    'activate': True,
    'speaker': 'en_5',
    'language': 'en',
    'model_id': 'v3_en',
    'sample_rate': 48000,
    'device': 'cpu',
    'max_wavs': 20,
    'play_audio': True,
    'show_text': True,
}
current_params = params.copy()
voices_by_gender = ['en_99', 'en_45', 'en_18', 'en_117', 'en_49', 'en_51', 'en_68', 'en_0', 'en_26', 'en_56', 'en_74', 'en_5', 'en_38', 'en_53', 'en_21', 'en_37', 'en_107', 'en_10', 'en_82', 'en_16', 'en_41', 'en_12', 'en_67', 'en_61', 'en_14', 'en_11', 'en_39', 'en_52', 'en_24', 'en_97', 'en_28', 'en_72', 'en_94', 'en_36', 'en_4', 'en_43', 'en_88', 'en_25', 'en_65', 'en_6', 'en_44', 'en_75', 'en_91', 'en_60', 'en_109', 'en_85', 'en_101', 'en_108', 'en_50', 'en_96', 'en_64', 'en_92', 'en_76', 'en_33', 'en_116', 'en_48', 'en_98', 'en_86', 'en_62', 'en_54', 'en_95', 'en_55', 'en_111', 'en_3', 'en_83', 'en_8', 'en_47', 'en_59', 'en_1', 'en_2', 'en_7', 'en_9', 'en_13', 'en_15', 'en_17', 'en_19', 'en_20', 'en_22', 'en_23', 'en_27', 'en_29', 'en_30', 'en_31', 'en_32', 'en_34', 'en_35', 'en_40', 'en_42', 'en_46', 'en_57', 'en_58', 'en_63', 'en_66', 'en_69', 'en_70', 'en_71', 'en_73', 'en_77', 'en_78', 'en_79', 'en_80', 'en_81', 'en_84', 'en_87', 'en_89', 'en_90', 'en_93', 'en_100', 'en_102', 'en_103', 'en_104', 'en_105', 'en_106', 'en_110', 'en_112', 'en_113', 'en_114', 'en_115']
wav_idx = 0

table = str.maketrans({
    "<": "&lt;",
    ">": "&gt;",
    "&": "&amp;",
    "'": "&apos;",
    '"': "&quot;",
})
def xmlesc(txt):
    return txt.translate(table)

def load_model():
    model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_tts', language=params['language'], speaker=params['model_id'])
    model.to(params['device'])
    return model
model = load_model()

def remove_surrounded_chars(string):
    new_string = ""
    in_star = False
    for char in string:
        if char == '*':
            in_star = not in_star
        elif not in_star:
            new_string += char
    return new_string

def input_modifier(string):
    """
    This function is applied to your text inputs before
    they are fed into the model.
    """

    return string

def output_modifier(string):
    """
    This function is applied to the model outputs.
    """

    global wav_idx, model, current_params

    for i in params:
        if params[i] != current_params[i]:
            model = load_model()
            current_params = params.copy()
            break

    if params['activate'] == False:
        return string

    orig_string = string
    string = remove_surrounded_chars(string)
    string = string.replace('"', '')
    string = string.replace('“', '')
    string = string.replace('\n', ' ')
    string = string.strip()

    auto_playable=True
    if string == '':
            string = 'empty reply, try regenerating'
            auto_playable=False
            

    #x-slow, slow, medium, fast, x-fast
    #x-low, low, medium, high, x-high
    #prosody='<prosody rate="fast" pitch="medium">'
    prosody='<prosody rate="fast">'
    string ='<speak>'+prosody+xmlesc(string)+'</prosody></speak>'
        
    output_file = Path(f'extensions/silero_tts/outputs/{wav_idx:06d}.wav')
    model.save_wav(ssml_text=string, speaker=params['speaker'], sample_rate=int(params['sample_rate']), audio_path=str(output_file))
    string = f'<audio src="file/{output_file.as_posix()}" controls></audio>'
    
    #reset if too many wavs. set max to -1 for unlimited.
    if wav_idx < params['max_wavs'] or params['max_wavs'] < 0:
        #only increment if starting a new stream, else replace during streaming. Does not update duration on webui sometimes?
        if not shared.still_streaming:
            wav_idx += 1
    else:
        wav_idx = 0
        
    if params['show_text']:
        string+='\n\n'+orig_string
    
    #if params['play_audio'] == True and auto_playable and shared.stop_everything:
    if params['play_audio'] == True and auto_playable and not shared.still_streaming:
        stop_autoplay()
        wave_obj = sa.WaveObject.from_wave_file(output_file.as_posix())
        wave_obj.play()

    return string

def bot_prefix_modifier(string):
    """
    This function is only applied in chat mode. It modifies
    the prefix text for the Bot and can be used to bias its
    behavior.
    """

    return string

def stop_autoplay():
    sa.stop_all()

def ui():
    # Gradio elements
    activate = gr.Checkbox(value=params['activate'], label='Activate TTS')
    show_text = gr.Checkbox(value=params['show_text'], label='Show message text under audio player')
    play_audio = gr.Checkbox(value=params['play_audio'], label='Play TTS automatically')
    stop_audio = gr.Button("Stop Auto-Play")
    voice = gr.Dropdown(value=params['speaker'], choices=voices_by_gender, label='TTS voice')

    # Event functions to update the parameters in the backend
    activate.change(lambda x: params.update({"activate": x}), activate, None)
    play_audio.change(lambda x: params.update({"play_audio": x}), play_audio, None)
    show_text.change(lambda x: params.update({"show_text": x}), show_text, None)
    stop_audio.click(stop_autoplay)
    voice.change(lambda x: params.update({"speaker": x}), voice, None)
