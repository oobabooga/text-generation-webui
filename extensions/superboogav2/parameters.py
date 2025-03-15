"""
This module provides a singleton class `Parameters` that is used to manage all hyperparameters for the embedding application.
It expects a JSON file in `extensions/superboogav2/config.json`.

Each element in the JSON must have a `default` value which will be used for the current run. Elements can have `categories`.
These categories define the range in which the optimizer will search. If the element is tagged with `"should_optimize": false`,
then the optimizer will only ever use the default value.
"""
import json
from pathlib import Path

from modules.logging_colors import logger

NUM_TO_WORD_METHOD = 'Number to Word'
NUM_TO_CHAR_METHOD = 'Number to Char'
NUM_TO_CHAR_LONG_METHOD = 'Number to Multi-Char'


DIST_MIN_STRATEGY = 'Min of Two'
DIST_HARMONIC_STRATEGY = 'Harmonic Mean'
DIST_GEOMETRIC_STRATEGY = 'Geometric Mean'
DIST_ARITHMETIC_STRATEGY = 'Arithmetic Mean'


PREPEND_TO_LAST = 'Prepend to Last Message'
APPEND_TO_LAST = 'Append to Last Message'
HIJACK_LAST_IN_CONTEXT = 'Hijack Last Message in Context ⚠️ WIP ⚠️ (Works Partially)'


SORT_DISTANCE = 'distance'
SORT_ID = 'id'


class Parameters:
    _instance = None

    variable_mapping = {
        'NUM_TO_WORD_METHOD': NUM_TO_WORD_METHOD,
        'NUM_TO_CHAR_METHOD': NUM_TO_CHAR_METHOD,
        'NUM_TO_CHAR_LONG_METHOD': NUM_TO_CHAR_LONG_METHOD,
        'DIST_MIN_STRATEGY': DIST_MIN_STRATEGY,
        'DIST_HARMONIC_STRATEGY': DIST_HARMONIC_STRATEGY,
        'DIST_GEOMETRIC_STRATEGY': DIST_GEOMETRIC_STRATEGY,
        'DIST_ARITHMETIC_STRATEGY': DIST_ARITHMETIC_STRATEGY,
        'PREPEND_TO_LAST': PREPEND_TO_LAST,
        'APPEND_TO_LAST': APPEND_TO_LAST,
        'HIJACK_LAST_IN_CONTEXT': HIJACK_LAST_IN_CONTEXT,
    }

    @staticmethod
    def getInstance():
        if Parameters._instance is None:
            Parameters()
        return Parameters._instance

    def __init__(self):
        if Parameters._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            Parameters._instance = self
            self.hyperparameters = self._load_from_json(Path("extensions/superboogav2/config.json"))

    def _load_from_json(self, file_path):
        logger.debug('Loading hyperparameters...')

        with open(file_path, 'r') as file:
            data = json.load(file)

        # Replace variable names in the dict and create Categorical objects
        for key in data:
            if "default" in data[key] and data[key]["default"] in self.variable_mapping:
                data[key]["default"] = self.variable_mapping[data[key]["default"]]
            if "categories" in data[key]:
                data[key]["categories"] = [self.variable_mapping.get(cat, cat) for cat in data[key]["categories"]]

        return data


def should_to_lower() -> bool:
    return bool(Parameters.getInstance().hyperparameters['to_lower']['default'])


def get_num_conversion_strategy() -> str:
    return Parameters.getInstance().hyperparameters['num_conversion']['default']


def should_merge_spaces() -> bool:
    return bool(Parameters.getInstance().hyperparameters['merge_spaces']['default'])


def should_strip() -> bool:
    return bool(Parameters.getInstance().hyperparameters['strip']['default'])


def should_remove_punctuation() -> bool:
    return bool(Parameters.getInstance().hyperparameters['remove_punctuation']['default'])


def should_remove_stopwords() -> bool:
    return bool(Parameters.getInstance().hyperparameters['remove_stopwords']['default'])


def should_remove_specific_pos() -> bool:
    return bool(Parameters.getInstance().hyperparameters['remove_specific_pos']['default'])


def should_lemmatize() -> bool:
    return bool(Parameters.getInstance().hyperparameters['lemmatize']['default'])


def get_min_num_sentences() -> int:
    return int(Parameters.getInstance().hyperparameters['min_num_sent']['default'])


def get_delta_start() -> int:
    return int(Parameters.getInstance().hyperparameters['delta_start']['default'])


def set_to_lower(value: bool):
    Parameters.getInstance().hyperparameters['to_lower']['default'] = value


def set_num_conversion_strategy(value: str):
    Parameters.getInstance().hyperparameters['num_conversion']['default'] = value


def set_merge_spaces(value: bool):
    Parameters.getInstance().hyperparameters['merge_spaces']['default'] = value


def set_strip(value: bool):
    Parameters.getInstance().hyperparameters['strip']['default'] = value


def set_remove_punctuation(value: bool):
    Parameters.getInstance().hyperparameters['remove_punctuation']['default'] = value


def set_remove_stopwords(value: bool):
    Parameters.getInstance().hyperparameters['remove_stopwords']['default'] = value


def set_remove_specific_pos(value: bool):
    Parameters.getInstance().hyperparameters['remove_specific_pos']['default'] = value


def set_lemmatize(value: bool):
    Parameters.getInstance().hyperparameters['lemmatize']['default'] = value


def set_min_num_sentences(value: int):
    Parameters.getInstance().hyperparameters['min_num_sent']['default'] = value


def set_delta_start(value: int):
    Parameters.getInstance().hyperparameters['delta_start']['default'] = value


def get_chunk_len() -> str:
    lens = []
    mask = Parameters.getInstance().hyperparameters['chunk_len_mask']['default']

    lens.append(Parameters.getInstance().hyperparameters['chunk_len1']['default'] if mask & (1 << 0) else None)
    lens.append(Parameters.getInstance().hyperparameters['chunk_len2']['default'] if mask & (1 << 1) else None)
    lens.append(Parameters.getInstance().hyperparameters['chunk_len3']['default'] if mask & (1 << 2) else None)
    lens.append(Parameters.getInstance().hyperparameters['chunk_len4']['default'] if mask & (1 << 3) else None)

    return ','.join([str(len) for len in lens if len])


def set_chunk_len(val: str):
    chunk_lens = sorted([int(len.strip()) for len in val.split(',')])

    # Reset the mask to zero
    Parameters.getInstance().hyperparameters['chunk_len_mask']['default'] = 0

    if len(chunk_lens) > 0:
        Parameters.getInstance().hyperparameters['chunk_len1']['default'] = chunk_lens[0]
        Parameters.getInstance().hyperparameters['chunk_len_mask']['default'] |= (1 << 0)
    if len(chunk_lens) > 1:
        Parameters.getInstance().hyperparameters['chunk_len2']['default'] = chunk_lens[1]
        Parameters.getInstance().hyperparameters['chunk_len_mask']['default'] |= (1 << 1)
    if len(chunk_lens) > 2:
        Parameters.getInstance().hyperparameters['chunk_len3']['default'] = chunk_lens[2]
        Parameters.getInstance().hyperparameters['chunk_len_mask']['default'] |= (1 << 2)
    if len(chunk_lens) > 3:
        Parameters.getInstance().hyperparameters['chunk_len4']['default'] = chunk_lens[3]
        Parameters.getInstance().hyperparameters['chunk_len_mask']['default'] |= (1 << 3)

    if len(chunk_lens) > 4:
        logger.warning(f'Only up to four chunk lengths are supported. Skipping {chunk_lens[4:]}')


def get_context_len() -> str:
    context_len = str(Parameters.getInstance().hyperparameters['context_len_left']['default']) + ',' + str(Parameters.getInstance().hyperparameters['context_len_right']['default'])
    return context_len


def set_context_len(val: str):
    context_lens = [int(len.strip()) for len in val.split(',') if len.isdigit()]
    if len(context_lens) == 1:
        Parameters.getInstance().hyperparameters['context_len_left']['default'] = Parameters.getInstance().hyperparameters['context_len_right']['default'] = context_lens[0]
    elif len(context_lens) == 2:
        Parameters.getInstance().hyperparameters['context_len_left']['default'] = context_lens[0]
        Parameters.getInstance().hyperparameters['context_len_right']['default'] = context_lens[1]
    else:
        logger.warning(f'Incorrect context length received {val}. Skipping.')


def get_new_dist_strategy() -> str:
    return Parameters.getInstance().hyperparameters['new_dist_strategy']['default']


def get_chunk_count() -> int:
    return int(Parameters.getInstance().hyperparameters['chunk_count']['default'])


def get_min_num_length() -> int:
    return int(Parameters.getInstance().hyperparameters['min_num_length']['default'])


def get_significant_level() -> float:
    return float(Parameters.getInstance().hyperparameters['significant_level']['default'])


def get_time_steepness() -> float:
    return float(Parameters.getInstance().hyperparameters['time_steepness']['default'])


def get_time_power() -> float:
    return float(Parameters.getInstance().hyperparameters['time_power']['default'])


def get_chunk_separator() -> str:
    return Parameters.getInstance().hyperparameters['chunk_separator']['default']


def get_prefix() -> str:
    return Parameters.getInstance().hyperparameters['prefix']['default']


def get_data_separator() -> str:
    return Parameters.getInstance().hyperparameters['data_separator']['default']


def get_postfix() -> str:
    return Parameters.getInstance().hyperparameters['postfix']['default']


def get_is_manual() -> bool:
    return bool(Parameters.getInstance().hyperparameters['manual']['default'])


def get_add_date_time() -> bool:
    return bool(Parameters.getInstance().hyperparameters['add_date_time']['default'])


def get_add_chat_to_data() -> bool:
    return bool(Parameters.getInstance().hyperparameters['add_chat_to_data']['default'])


def get_injection_strategy() -> str:
    return Parameters.getInstance().hyperparameters['injection_strategy']['default']


def get_chunk_regex() -> str:
    return Parameters.getInstance().hyperparameters['chunk_regex']['default']


def get_is_strong_cleanup() -> bool:
    return bool(Parameters.getInstance().hyperparameters['strong_cleanup']['default'])


def get_max_token_count() -> int:
    return int(Parameters.getInstance().hyperparameters['max_token_count']['default'])


def get_num_threads() -> int:
    return int(Parameters.getInstance().hyperparameters['threads']['default'])


def get_optimization_steps() -> int:
    return int(Parameters.getInstance().hyperparameters['optimization_steps']['default'])


def get_api_port() -> int:
    return int(Parameters.getInstance().hyperparameters['api_port']['default'])


def get_api_on() -> bool:
    return bool(Parameters.getInstance().hyperparameters['api_on']['default'])


def set_new_dist_strategy(value: str):
    Parameters.getInstance().hyperparameters['new_dist_strategy']['default'] = value


def set_chunk_count(value: int):
    Parameters.getInstance().hyperparameters['chunk_count']['default'] = value


def set_min_num_length(value: int):
    Parameters.getInstance().hyperparameters['min_num_length']['default'] = value


def set_significant_level(value: float):
    Parameters.getInstance().hyperparameters['significant_level']['default'] = value


def set_time_steepness(value: float):
    Parameters.getInstance().hyperparameters['time_steepness']['default'] = value


def set_time_power(value: float):
    Parameters.getInstance().hyperparameters['time_power']['default'] = value


def set_chunk_separator(value: str):
    Parameters.getInstance().hyperparameters['chunk_separator']['default'] = value


def set_prefix(value: str):
    Parameters.getInstance().hyperparameters['prefix']['default'] = value


def set_data_separator(value: str):
    Parameters.getInstance().hyperparameters['data_separator']['default'] = value


def set_postfix(value: str):
    Parameters.getInstance().hyperparameters['postfix']['default'] = value


def set_manual(value: bool):
    Parameters.getInstance().hyperparameters['manual']['default'] = value


def set_add_date_time(value: bool):
    Parameters.getInstance().hyperparameters['add_date_time']['default'] = value


def set_add_chat_to_data(value: bool):
    Parameters.getInstance().hyperparameters['add_chat_to_data']['default'] = value


def set_injection_strategy(value: str):
    Parameters.getInstance().hyperparameters['injection_strategy']['default'] = value


def set_chunk_regex(value: str):
    Parameters.getInstance().hyperparameters['chunk_regex']['default'] = value


def set_strong_cleanup(value: bool):
    Parameters.getInstance().hyperparameters['strong_cleanup']['default'] = value


def set_max_token_count(value: int):
    Parameters.getInstance().hyperparameters['max_token_count']['default'] = value


def set_num_threads(value: int):
    Parameters.getInstance().hyperparameters['threads']['default'] = value


def set_optimization_steps(value: int):
    Parameters.getInstance().hyperparameters['optimization_steps']['default'] = value


def set_api_port(value: int):
    Parameters.getInstance().hyperparameters['api_port']['default'] = value


def set_api_on(value: bool):
    Parameters.getInstance().hyperparameters['api_on']['default'] = value
