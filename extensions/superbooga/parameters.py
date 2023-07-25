"""
This file contains all the hyperparameters used in the embedding pipeline.
The first parameter in the list is the current value.
The second parameter in the list is the scope.
"""
from skopt.space import Categorical, Integer, Real
from modules.logging_colors import logger


NUM_TO_WORD_METHOD = 'Number to Word'
NUM_TO_CHAR_METHOD = 'Number to Char'
NUM_TO_CHAR_LONG_METHOD = 'Number to Multi-Char'


DIST_MIN_STRATEGY = 'Min of Two'
DIST_HARMONIC_STRATEGY = 'Harmonic Mean'
DIST_GEOMETRIC_STRATEGY = 'Geometric Mean'
DIST_ARITHMETIC_STRATEGY = 'Arithmetic Mean'


hyperparameters = {
    # === PREPROCESS HYPERPARAMS ===
    'to_lower': [True, Categorical([True, False])],
    'num_conversion': [NUM_TO_WORD_METHOD, Categorical([NUM_TO_WORD_METHOD, NUM_TO_CHAR_METHOD, NUM_TO_CHAR_LONG_METHOD, None])],
    'merge_spaces': [True, Categorical([True, False])],
    'strip': [True, Categorical([True, False])],
    'remove_punctuation': [True, Categorical([True, False])],
    'remove_stopwords': [False, Categorical([True, False])],
    'remove_specific_pos': [True, Categorical([True, False])],
    'lemmatize': [False, Categorical([True, False])],

    'min_num_sent': [1, Categorical([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 999999])],


    # === PROCESS HYPERPARAMS ===
    'delta_start': [30, Integer(0, 100)],

    'chunk_len1': [40, Integer(30, 100)],
    'include_len1': [True, Categorical([True, False])],
    'chunk_len2': [50, Integer(30, 100)],
    'include_len2': [True, Categorical([True, False])],
    'chunk_len3': [100, Integer(30, 700)],
    'include_len3': [False, Categorical([True, False])],
    'chunk_len4': [300, Integer(30, 700)],
    'include_len4': [False, Categorical([True, False])],

    'context_len_left': [300, Integer(10, 1000)],
    'context_len_right': [850, Integer(100, 1500)],


    # === POSTPROCESS HYPERPARAMS ===
    'new_dist_strategy': [DIST_MIN_STRATEGY, Categorical([DIST_MIN_STRATEGY, DIST_HARMONIC_STRATEGY, DIST_GEOMETRIC_STRATEGY, DIST_ARITHMETIC_STRATEGY])],
    'chunk_count': [180, Integer(50, 400)],
    'min_num_length': [1, Integer(1, 10)],
    'confidence_interval': [0.6, Real(0.1, 1.0, prior="uniform")],
    'time_weight': [0, Real(0.0, 1.0, prior="uniform")],

    
    # === NON-OPTIMIZABLE HYPERPARAMS ===
    'chunk_separator': '',
    'data_separator': '\n\n<<document chunk>>\n\n',
    'chunk_regex': '(?<==== ).*?(?= ===)|User story: \d+',
    'strong_cleanup': False,
    'max_token_count': 3072,
    'threads': 4
}


def should_to_lower() -> bool:
    return hyperparameters['to_lower'][0]


def get_num_conversion_strategy() -> str:
    return hyperparameters['num_conversion'][0]


def should_merge_spaces() -> bool:
    return hyperparameters['merge_spaces'][0]


def should_strip() -> bool:
    return hyperparameters['strip'][0]


def should_remove_punctuation() -> bool:
    return hyperparameters['remove_punctuation'][0]


def should_remove_stopwords() -> bool:
    return hyperparameters['remove_stopwords'][0]


def should_remove_specific_pos() -> bool:
    return hyperparameters['remove_specific_pos'][0]


def should_lemmatize() -> bool:
    return hyperparameters['lemmatize'][0]


def get_min_num_sentences() -> int:
    return hyperparameters['min_num_sent'][0]


def get_delta_start() -> int:
    return hyperparameters['delta_start'][0]


def set_to_lower(value: bool):
    hyperparameters['to_lower'][0] = value


def set_num_conversion_strategy(value: str):
    hyperparameters['num_conversion'][0] = value


def set_merge_spaces(value: bool):
    hyperparameters['merge_spaces'][0] = value


def set_strip(value: bool):
    hyperparameters['strip'][0] = value


def set_remove_punctuation(value: bool):
    hyperparameters['remove_punctuation'][0] = value


def set_remove_stopwords(value: bool):
    hyperparameters['remove_stopwords'][0] = value


def set_remove_specific_pos(value: bool):
    hyperparameters['remove_specific_pos'][0] = value


def set_lemmatize(value: bool):
    hyperparameters['lemmatize'][0] = value


def set_min_num_sentences(value: int):
    hyperparameters['min_num_sent'][0] = value


def set_delta_start(value: int):
    hyperparameters['delta_start'][0] = value


def get_chunk_len() -> str:
    lens = []
    if hyperparameters['include_len1'][0]:
        lens.append(hyperparameters['chunk_len1'][0])

    if hyperparameters['include_len2'][0]:
        lens.append(hyperparameters['chunk_len2'][0])

    if hyperparameters['include_len3'][0]:
        lens.append(hyperparameters['chunk_len3'][0])

    if hyperparameters['include_len4'][0]:
        lens.append(hyperparameters['chunk_len4'][0])

    return ','.join([str(len) for len in lens])


def set_chunk_len(val: str):
    chunk_lens = sorted([int(len.strip()) for len in val.split(',')])
    if len(chunk_lens) > 0:
        if 30 <= chunk_lens[0] <= 100:
            hyperparameters['chunk_len1'][0] = chunk_lens[0]
    if len(chunk_lens) > 1:
        if 30 <= chunk_lens[1] <= 100:
            hyperparameters['chunk_len2'][0] = chunk_lens[1]
    if len(chunk_lens) > 2:
        if 30 <= chunk_lens[2] <= 700:
            hyperparameters['chunk_len3'][0] = chunk_lens[2]
    if len(chunk_lens) > 3:
        if 30 <= chunk_lens[3] <= 700:
            hyperparameters['chunk_len4'][0] = chunk_lens[3]

    if len(chunk_lens) > 4:
        logger.warning(f'Only up to four chunk lengths are supported. Skipping {chunk_lens[4:]}')


def get_context_len() -> str:
    context_len = str(hyperparameters['context_len_left'][0]) + ',' + str(hyperparameters['context_len_right'][0])
    return context_len


def set_context_len(val: str):
    context_lens = [int(len.strip()) for len in val.split(',') if len.isdigit()]
    if len(context_lens) == 1:
        hyperparameters['context_len_left'] = hyperparameters['context_len_right'] = context_lens[0]
    elif len(context_lens) == 2:
        hyperparameters['context_len_left'] = context_lens[0]
        hyperparameters['context_len_right'] = context_lens[1]
    else:
        logger.warning(f'Incorrect context length received {val}. Skipping.')


def get_new_dist_strategy() -> str:
    return hyperparameters['new_dist_strategy'][0]


def get_chunk_count() -> int:
    return hyperparameters['chunk_count'][0]


def get_min_num_length() -> int:
    return hyperparameters['min_num_length'][0]


def get_confidence_interval() -> float:
    return hyperparameters['confidence_interval'][0]


def get_time_weight() -> int:
    return hyperparameters['time_weight'][0]


def get_chunk_separator() -> str:
    return hyperparameters['chunk_separator']


def get_data_separator() -> str:
    return hyperparameters['data_separator']


def get_chunk_regex() -> str:
    return hyperparameters['chunk_regex']


def get_is_strong_cleanup() -> bool:
    return hyperparameters['strong_cleanup']


def get_max_token_count() -> int:
    return hyperparameters['max_token_count']


def get_num_threads() -> int:
    return hyperparameters['threads']

def set_new_dist_strategy(value: str):
    hyperparameters['new_dist_strategy'][0] = value


def set_chunk_count(value: int):
    hyperparameters['chunk_count'][0] = value


def set_min_num_length(value: int):
    hyperparameters['min_num_length'][0] = value


def set_confidence_interval(value: float):
    hyperparameters['confidence_interval'][0] = value


def set_time_weight(value: int):
    hyperparameters['time_weight'][0] = value


def set_chunk_separator(value: str):
    hyperparameters['chunk_separator'] = value


def set_data_separator(value: str):
    hyperparameters['data_separator'] = value


def set_chunk_regex(value: str):
    hyperparameters['chunk_regex'] = value


def set_strong_cleanup(value: bool):
    hyperparameters['strong_cleanup'] = value


def set_max_token_count(value: int):
    hyperparameters['max_token_count'] = value


def set_num_threads(value: int):
    hyperparameters['threads'] = value
