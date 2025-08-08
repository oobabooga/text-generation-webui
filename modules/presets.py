import functools
import pprint
from pathlib import Path

import yaml

from modules import shared
from modules.loaders import loaders_samplers
from modules.logging_colors import logger


def default_preset():
    result = {
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
        'sampler_priority': 'repetition_penalty\npresence_penalty\nfrequency_penalty\ndry\ntop_n_sigma\ntemperature\ndynamic_temperature\nquadratic_sampling\ntop_k\ntop_p\ntypical_p\nepsilon_cutoff\neta_cutoff\ntfs\ntop_a\nmin_p\nmirostat\nxtc\nencoder_repetition_penalty\nno_repeat_ngram',
        'dry_sequence_breakers': '"\\n", ":", "\\"", "*"',
    }

    if shared.args.portable:
        samplers = result['sampler_priority'].split('\n')
        samplers = [sampler for sampler in samplers if sampler in ["dry", "top_k", "top_p", "top_n_sigma", "min_p", "temperature", "xtc", "typical_p", "repetition_penalty"]]
        result['sampler_priority'] = '\n'.join(samplers)

    return result


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


def reset_preset_for_ui(name, state):
    """Reset current preset to its saved values from file"""
    generate_params = load_preset(name, verbose=True)
    state.update(generate_params)
    return state, *[generate_params[k] for k in presets_params()]


def neutralize_samplers_for_ui(state):
    """Set all samplers to their default/neutral values"""
    generate_params = default_preset()
    state.update(generate_params)
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
