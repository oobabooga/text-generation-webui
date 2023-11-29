import functools
import random
from pathlib import Path

import yaml

from modules import shared
from modules.loaders import loaders_samplers


def default_preset():
    return {
        'temperature': 1,
        'temperature_last': False,
        'top_p': 1,
        'min_p': 0,
        'top_k': 0,
        'repetition_penalty': 1,
        'presence_penalty': 0,
        'frequency_penalty': 0,
        'repetition_penalty_range': 0,
        'typical_p': 1,
        'tfs': 1,
        'top_a': 0,
        'epsilon_cutoff': 0,
        'eta_cutoff': 0,
        'guidance_scale': 1,
        'penalty_alpha': 0,
        'mirostat_mode': 0,
        'mirostat_tau': 5,
        'mirostat_eta': 0.1,
        'do_sample': True,
        'encoder_repetition_penalty': 1,
        'no_repeat_ngram_size': 0,
        'min_length': 0,
        'num_beams': 1,
        'length_penalty': 1,
        'early_stopping': False,
    }


def presets_params():
    return [k for k in default_preset()]


def load_preset(name):
    generate_params = default_preset()
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
    return state, *[generate_params[k] for k in presets_params()]


def random_preset(state):
    params_and_values = {
        'remove_tail_tokens': {
            'top_p': [0.5, 0.8, 0.9, 0.95, 0.99],
            'min_p': [0.5, 0.2, 0.1, 0.05, 0.01],
            'top_k': [3, 5, 10, 20, 30, 40],
            'typical_p': [0.2, 0.575, 0.95],
            'tfs': [0.5, 0.8, 0.9, 0.95, 0.99],
            'top_a': [0.5, 0.2, 0.1, 0.05, 0.01],
            'epsilon_cutoff': [1, 3, 5, 7, 9],
            'eta_cutoff': [3, 6, 9, 12, 15, 18],
        },
        'flatten_distribution': {
            'temperature': [0.5, 0.7, 0.8, 1, 1.2, 1.5, 2.0],
        },
        'repetition': {
            'repetition_penalty': [1, 1.05, 1.1, 1.15, 1.20, 1.25],
            'presence_penalty': [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0],
            'frequency_penalty': [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0],
        },
        'other': {
            'temperature_last': [True, False],
        }
    }

    generate_params = default_preset()
    for cat in params_and_values:
        choices = list(params_and_values[cat].keys())
        if shared.args.loader is not None:
            choices = [x for x in choices if x in loaders_samplers[shared.args.loader]]

        if len(choices) > 0:
            choice = random.choice(choices)
            generate_params[choice] = random.choice(params_and_values[cat][choice])

    state.update(generate_params)
    return state, *[generate_params[k] for k in presets_params()]


def generate_preset_yaml(state):
    defaults = default_preset()
    data = {k: state[k] for k in presets_params()}

    # Remove entries that are identical to the defaults
    for k in list(data.keys()):
        if data[k] == defaults[k]:
            del data[k]

    return yaml.dump(data, sort_keys=False)
