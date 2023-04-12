from TTS.api import TTS
from pathlib import Path
import time

from modules import tts_preprocessor

# Running a multi-speaker and multilingual model
params = {
    'activate': True,
    'speaker': None,
    'language': None,
    'model_id': 'tts_models/en/ek1/tacotron2',
    'Cuda': True,
    'show_text': True,
    'autoplay': True,
}

voices_by_gender = [
    'tts_models/en/ek1/tacotron2',
    'tts_models/en/ljspeech/tacotron2-DDC',
    'tts_models/en/ljspeech/tacotron2-DDC_ph',
    'tts_models/en/ljspeech/glow-tts',
    'tts_models/en/ljspeech/speedy-speech',
    'tts_models/en/ljspeech/tacotron2-DCA',
    'tts_models/en/ljspeech/vits',
    'tts_models/en/ljspeech/vits--neon',
    'tts_models/en/ljspeech/fast_pitch',
    'tts_models/en/ljspeech/overflow',
    'tts_models/en/ljspeech/neural_hmm',
    'tts_models/en/vctk/vits',
    'tts_models/en/vctk/fast_pitch',
    'tts_models/en/sam/tacotron-DDC',
    'tts_models/en/blizzard2013/capacitron-t2-c50',
    'tts_models/en/blizzard2013/capacitron-t2-c150_v2'
]


def load_model():
    # Init TTS
    tts = TTS(params['model_id'], gpu=params['Cuda'])
    if tts is not None and tts.synthesizer is not None and tts.synthesizer.tts_config is not None and hasattr(tts.synthesizer.tts_config, 'num_chars'):
        tts.synthesizer.tts_config.num_chars = 1000

    temp_speaker = tts.speakers if tts.speakers is not None else []
    temp_speaker = params['speaker'] if params['speaker'] in temp_speaker else temp_speaker[0] if len(temp_speaker) > 0 else None

    temp_language = tts.languages if tts.languages is not None else []
    temp_language = params['language'] if params['language'] in temp_language else temp_language[0] if len(temp_language) > 0 else None

    return tts, temp_speaker, temp_language


model, speaker, language = load_model()


def output_modifier(string):
    """
    This function is applied to the model outputs.
    """

    global model, speaker, language, params

    original_string = string
    # we don't need to handle numbers. The text normalizer in coqui does it better
    string = tts_preprocessor.replace_invalid_chars(string)
    string = tts_preprocessor.replace_abbreviations(string)
    string = tts_preprocessor.clean_whitespace(string)
    processed_string = string

    if string == '':
        string = '*Empty reply, try regenerating*'
    else:
        output_file = Path(f'extensions/coqui_tts/outputs/test_{int(time.time())}.wav')
        # ❗ Since this model is multi-speaker and multilingual, we must set the target speaker and the language
        model.tts_to_file(text=string, speaker=speaker, language=language, file_path=str(output_file))

        autoplay = 'autoplay' if params['autoplay'] else ''
        string = f'<audio src="file/{output_file.as_posix()}" controls {autoplay}></audio>'

        if params['show_text']:
            string += f'\n\n{original_string}\n\nProcessed:\n{processed_string}'

    print(string)


if __name__ == '__main__':
    import sys
    output_modifier(sys.argv[1])
