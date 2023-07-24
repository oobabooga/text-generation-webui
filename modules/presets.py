import functools
from pathlib import Path

import yaml


def load_preset(name):
    generate_params = {
        'do_sample': True,
        'temperature': 1,
        'top_p': 1,
        'typical_p': 1,
        'epsilon_cutoff': 0,
        'eta_cutoff': 0,
        'tfs': 1,
        'top_a': 0,
        'repetition_penalty': 1,
        'repetition_penalty_range': 0,
        'encoder_repetition_penalty': 1,
        'top_k': 0,
        'num_beams': 1,
        'penalty_alpha': 0,
        'min_length': 0,
        'length_penalty': 1,
        'no_repeat_ngram_size': 0,
        'early_stopping': False,
        'mirostat_mode': 0,
        'mirostat_tau': 5.0,
        'mirostat_eta': 0.1,
    }

    if name not in ['None', None, '']:
        with open(Path(f'presets/{name}.yaml'), 'r') as infile:
            preset = yaml.safe_load(infile)

        for k in preset:
            generate_params[k] = preset[k]

    generate_params['temperature'] = min(1.99, generate_params['temperature'])
    return generate_params


@functools.cache
def load_preset_memoized(name):
    return load_preset(name)


def load_preset_for_ui(name, state):
    generate_params = load_preset(name)
    state.update(generate_params)
    return state, *[generate_params[k] for k in ['do_sample', 'temperature', 'top_p', 'typical_p', 'epsilon_cutoff', 'eta_cutoff', 'repetition_penalty', 'repetition_penalty_range', 'encoder_repetition_penalty', 'top_k', 'min_length', 'no_repeat_ngram_size', 'num_beams', 'penalty_alpha', 'length_penalty', 'early_stopping', 'mirostat_mode', 'mirostat_tau', 'mirostat_eta', 'tfs', 'top_a']]


def generate_preset_yaml(state):
    data = {k: state[k] for k in ['do_sample', 'temperature', 'top_p', 'typical_p', 'epsilon_cutoff', 'eta_cutoff', 'repetition_penalty', 'repetition_penalty_range', 'encoder_repetition_penalty', 'top_k', 'min_length', 'no_repeat_ngram_size', 'num_beams', 'penalty_alpha', 'length_penalty', 'early_stopping', 'mirostat_mode', 'mirostat_tau', 'mirostat_eta', 'tfs', 'top_a']}
    return yaml.dump(data, sort_keys=False)
