import functools
import pprint
import random
from pathlib import Path

import yaml

from modules import shared
from modules.loaders import loaders_samplers
from modules.logging_colors import logger


def default_preset():
    return {
        'temperature': 1,
        'dynatemp_low': 1,
        'dynatemp_high': 1,
        'dynatemp_exponent': 1,
        'smoothing_factor': 0,
        'smoothing_curve': 1,
        'min_p': 0,
        'top_p': 1,
        'top_k': 0,
        'typical_p': 1,
        'xtc_threshold': 0.1,
        'xtc_probability': 0,
        'epsilon_cutoff': 0,
        'eta_cutoff': 0,
        'tfs': 1,
        'top_a': 0,
        'top_n_sigma': 0,
        'dry_multiplier': 0,
        'dry_allowed_length': 2,
        'dry_base': 1.75,
        'repetition_penalty': 1,
        'frequency_penalty': 0,
        'presence_penalty': 0,
        'encoder_repetition_penalty': 1,
        'no_repeat_ngram_size': 0,
        'repetition_penalty_range': 1024,
        'penalty_alpha': 0,
        'guidance_scale': 1,
        'mirostat_mode': 0,
        'mirostat_tau': 5,
        'mirostat_eta': 0.1,
        'do_sample': True,
        'dynamic_temperature': False,
        'temperature_last': False,
        'sampler_priority': 'repetition_penalty\npresence_penalty\nfrequency_penalty\ndry\ntemperature\ndynamic_temperature\nquadratic_sampling\ntop_n_sigma\ntop_k\ntop_p\ntypical_p\nepsilon_cutoff\neta_cutoff\ntfs\ntop_a\nmin_p\nmirostat\nxtc\nencoder_repetition_penalty\nno_repeat_ngram',
        'dry_sequence_breakers': '"\\n", ":", "\\"", "*"',
    }


def presets_params():
    return [k for k in default_preset()]


def load_preset(name, verbose=False):
    generate_params = default_preset()
    if name not in ['None', None, '']:
        path = Path(f'user_data/presets/{name}.yaml')
        if path.exists():
            with open(path, 'r') as infile:
                preset = yaml.safe_load(infile)

            for k in preset:
                generate_params[k] = preset[k]
        else:
            logger.error(f"The preset \"{name}\" does not exist under \"{path}\". Using the default parameters.")

    if verbose:
        logger.info(f"\"{name}\" preset:")
        pprint.PrettyPrinter(indent=4, width=1, sort_dicts=False).pprint(remove_defaults(generate_params))

    return generate_params


@functools.cache
def load_preset_memoized(name):
    return load_preset(name)


def load_preset_for_ui(name, state):
    generate_params = load_preset(name, verbose=True)
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
            'temperature': [0.1, 0.5, 0.7, 0.8, 1, 1.2, 1.5, 2.0, 5.0],
            'dynamic_temperature': [
                [0.1, 1],
                [0.1, 1.5],
                [0.1, 2],
                [0.1, 5],
                [0.5, 1],
                [0.5, 1.5],
                [0.5, 2],
                [0.5, 5],
                [0.8, 1],
                [0.8, 1.5],
                [0.8, 2],
                [0.8, 5],
                [1, 1.5],
                [1, 2],
                [1, 5]
            ],
            'smoothing_factor': [0.2, 0.3, 0.6, 1.2],
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
            choices = [x for x in choices if loader_contains(x)]

        if len(choices) > 0:
            choice = random.choice(choices)
            value = random.choice(params_and_values[cat][choice])
            if choice == 'dynamic_temperature':
                generate_params['dynamic_temperature'] = True
                generate_params['dynatemp_low'] = value[0]
                generate_params['dynatemp_high'] = value[1]
            else:
                generate_params[choice] = value

    state.update(generate_params)
    logger.info("GENERATED_PRESET=")
    pprint.PrettyPrinter(indent=4, width=1, sort_dicts=False).pprint(remove_defaults(state))
    return state, *[generate_params[k] for k in presets_params()]


def loader_contains(sampler):
    if sampler == 'dynamic_temperature' and 'dynatemp_low' in loaders_samplers[shared.args.loader]:
        return True
    else:
        return sampler in loaders_samplers[shared.args.loader]


def remove_defaults(state):
    defaults = default_preset()
    data = {k: state[k] for k in presets_params()}

    for k in list(data.keys()):
        if data[k] == defaults[k]:
            del data[k]

    return data


def generate_preset_yaml(state):
    data = remove_defaults(state)
    return yaml.dump(data, sort_keys=False)
