import asyncio
from pathlib import Path

import torch

torch._C._jit_set_profiling_mode(False)

params = {
    'activate': True,
    'speaker': 'en_56',
    'language': 'en',
    'model_id': 'v3_en',
    'sample_rate': 48000,
    'device': 'cpu',
}
current_params = params.copy()
wav_idx = 0

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

    string = remove_surrounded_chars(string)
    string = string.replace('"', '')
    string = string.replace('“', '')
    string = string.replace('\n', ' ')
    string = string.strip()

    if string == '':
        string = 'empty reply, try regenerating'

    output_file = Path(f'extensions/silero_tts/outputs/{wav_idx:06d}.wav')
    audio = model.save_wav(text=string, speaker=params['speaker'], sample_rate=int(params['sample_rate']), audio_path=str(output_file))

    string = f'<audio src="file/{output_file.as_posix()}" controls></audio>'
    wav_idx += 1

    return string

def bot_prefix_modifier(string):
    """
    This function is only applied in chat mode. It modifies
    the prefix text for the Bot and can be used to bias its
    behavior.
    """

    return string
