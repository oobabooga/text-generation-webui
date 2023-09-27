"""
This module implements a hyperparameter optimization routine for the embedding application. It utilizes TPE optimization from Optuna.

Each run, the optimizer will set the default values inside the hyperparameters. At the end, it will output the best ones it has found.
"""
import re
import json
import optuna
import gradio as gr
import numpy as np
import logging
import hashlib
logging.getLogger('optuna').setLevel(logging.WARNING)

import extensions.superboogav2.parameters as parameters

from pathlib import Path

from .benchmark import benchmark
from .parameters import Parameters
from modules.logging_colors import logger


# Format the parameters into markdown format.
def _markdown_hyperparams():
    res = []
    for param_name, param_value in Parameters.getInstance().hyperparameters.items():
        # Escape any markdown syntax
        param_name = re.sub(r"([_*\[\]()~`>#+-.!])", r"\\\1", param_name)
        param_value_default = re.sub(r"([_*\[\]()~`>#+-.!])", r"\\\1", str(param_value['default'])) if param_value['default'] else ' '
        
        res.append('* {}: **{}**'.format(param_name, param_value_default))

    return '\n'.join(res)


# Convert numpy types to python types.
def _convert_np_types(params):
    for key in params:
        if type(params[key]) == np.bool_:
            params[key] = bool(params[key])
        elif type(params[key]) == np.int64:
            params[key] = int(params[key])
        elif type(params[key]) == np.float64:
            params[key] = float(params[key])
    return params


# Set the default values for the hyperparameters.
def _set_hyperparameters(params):
    for param_name, param_value in params.items():
        if param_name in Parameters.getInstance().hyperparameters: 
            Parameters.getInstance().hyperparameters[param_name]['default'] = param_value


# Check if the parameter is for optimization.
def _is_optimization_param(val):
    is_opt = val.get('should_optimize', False) # Either does not exist or is false
    return is_opt


# Create a hashable representation of the parameters
def _get_params_hash(params):
    params_str = json.dumps(params, sort_keys=True)
    return hashlib.sha256(params_str.encode()).hexdigest()


def optimize(collector, progress=gr.Progress()):
    # Inform the user that something is happening.
    progress(0, desc=f'Setting Up...')

    # Track the current step
    current_step = 0

    # Track the best score
    best_score = 0

    # Dictionary for caching scores
    scores_cache = {}

    def objective_function(trial):
        nonlocal current_step
        nonlocal best_score
        nonlocal scores_cache

        params = {}
        for key, val in Parameters.getInstance().hyperparameters.items():
            if _is_optimization_param(val):
                params[key] = trial.suggest_categorical(key, val['categories'])

        _set_hyperparameters(params)

        params_hash = _get_params_hash(params)

        # If the score for these parameters is in the cache, return it
        if params_hash in scores_cache:
            return scores_cache[params_hash]

        # Benchmark the current set of parameters.
        score, max_score = benchmark(Path("extensions/superboogav2/benchmark_texts/questions.json"), collector)

        # Cache the score
        scores_cache[params_hash] = score

        result = json.dumps(_convert_np_types(params), indent=4)
        result += f'\nScore: {score}/{max_score}'

        logger.debug(result)

        # Increment the current step
        current_step += 1

        # Update the best score
        best_score = max(best_score, score)

        # Update the progress
        progress(current_step / parameters.get_optimization_steps(), desc=f'Optimizing... {current_step}/{parameters.get_optimization_steps()}')

        return -score

    # Run the optimization.
    study = optuna.create_study()
    study.optimize(objective_function, n_trials=int(parameters.get_optimization_steps()))

    best_params = study.best_params
    _set_hyperparameters(best_params)

    # Convert results to a markdown string.
    str_result = f"## Best parameters:\n\n{_markdown_hyperparams()}\n\n## Score:\n\n{best_score}"

    # Save to JSON file
    with open('best_params.json', 'w') as fp:
        json.dump(_convert_np_types(best_params), fp, indent=4)

    return str_result