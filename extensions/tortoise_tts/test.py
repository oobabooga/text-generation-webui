import torch
import torchaudio

# needs to be before the tortoise stuff to properly import
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'tortoise'))

from tortoise import api
from tortoise.utils.audio import load_voices
from tortoise.utils.text import split_and_recombine_text

from pathlib import Path
import time

from modules import tts_preprocessor

params = {
    'activate': True,
    'voice': 'emma',
    'preset': 'ultra_fast',
    'device': 'cuda',
    'show_text': True,
    'autoplay': True,
}

voices = [
    'angie',
    'applejack',
    'cond_latent_example',
    'daniel',
    'deniro',
    'emma',
    'freeman',
    'geralt',
    'halle',
    'jlaw',
    'lj',
    'mol',
    'myself',
    'pat',
    'pat2',
    'rainbow',
    'snakes',
    'tim_reynolds',
    'tom',
    'train_atkins',
    'train_daws',
    'train_dotrice',
    'train_dreams',
    'train_empire',
    'train_grace',
    'train_kennard',
    'train_lescault',
    'train_mouse',
    'weaver',
    'william'
]


def load_model():
    # Init TTS
    tts = api.TextToSpeech()
    samples, latents = load_voices(voices=[params['voice']])

    return tts, samples, latents


model, voice_samples, conditioning_latents = load_model()


def output_modifier(string):
    """
    This function is applied to the model outputs.
    """

    global model, voice_samples, conditioning_latents, params

    original_string = string
    # we don't need to handle numbers. The text normalizer in coqui does it better
    string = tts_preprocessor.replace_invalid_chars(string)
    string = tts_preprocessor.replace_abbreviations(string)
    string = tts_preprocessor.clean_whitespace(string)
    processed_string = string

    if string == '':
        string = '*Empty reply, try regenerating*'
    else:
        output_dir = Path(f'extensions/tortoise_tts/outputs/parts')
        if not output_dir.is_dir():
            output_dir.mkdir(parents=True, exist_ok=True)

        output_file = Path(f'extensions/tortoise_tts/outputs/test_{int(time.time())}.wav')

        if '|' in string:
            texts = string.split('|')
        else:
            texts = split_and_recombine_text(string, desired_length=10, max_length=400)

        all_parts = []
        for j, text in enumerate(texts):
            gen = model.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                                        preset=params['preset'], k=1, use_deterministic_seed=int(time.time()))
            gen = gen.squeeze(0).cpu()
            torchaudio.save(output_dir.joinpath(f'{j}_{int(time.time())}.wav'), gen, 24000)
            all_parts.append(gen)

        full_audio = torch.cat(all_parts, dim=-1)
        torchaudio.save(str(output_file), full_audio, 24000)

        autoplay = 'autoplay' if params['autoplay'] else ''
        string = f'<audio src="file/{output_file.as_posix()}" controls {autoplay}></audio>'

        if params['show_text']:
            string += f'\n\n{original_string}\n\nProcessed:\n{processed_string}'

    print(string)


if __name__ == '__main__':
    import sys
    output_modifier(sys.argv[1])
