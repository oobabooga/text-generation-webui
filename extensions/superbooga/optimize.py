"""
This module implements a hyperparameter optimization routine for the embedding application. It utilizes Gaussian Process optimization from Scikit-Optimize.

Each run, the optimizer will set the default values inside the hyperparameters. At the end, it will output the best ones it has found.
"""
import json
import gradio as gr
import numpy as np

import extensions.superbooga.parameters as parameters

from skopt.utils import use_named_args
from skopt import gp_minimize
from pathlib import Path

from .benchmark import benchmark
from .parameters import Parameters
from modules.logging_colors import logger


# Format the parameters into markdown format.
def _markdown_hyperparams():
    res = []
    for param_name, param_value in Parameters.getInstance().hyperparameters.items():
        res.append('* {}: **{}**'.format(param_name, param_value['default']))

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


def optimize(collector, progress=gr.Progress()):
    # Create the optimization space.
    optimization_space = [val['categories'] for val in Parameters.getInstance().hyperparameters.values() if _is_optimization_param(val)]

    # Inform the user that something is happening.
    progress(0, desc=f'Setting Up...')

    # Track the current step
    current_step = 0

    # Track the best score
    best_score = 0

    @use_named_args(optimization_space)
    def objective_function(**params):
        nonlocal current_step
        nonlocal best_score

        _set_hyperparameters(params)

        # Benchmark the current set of parameters.
        score, max_score = benchmark(Path("extensions/superbooga/benchmark_texts/questions.json"), collector)

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

    # Retrieve initial parameter values.
    initial_params = [val['default'] for val in Parameters.getInstance().hyperparameters.values() if _is_optimization_param(val)]

    # Run the optimization.
    result = gp_minimize(objective_function, optimization_space, x0=initial_params, n_calls=int(parameters.get_optimization_steps()), random_state=0)

    best_params = dict(zip([key for key, val in Parameters.getInstance().hyperparameters.items() if _is_optimization_param(val)], result.x))
    _set_hyperparameters(best_params)

    # Convert results to a markdown string.
    str_result = f"## Best parameters:\n\n{_markdown_hyperparams()}\n\n## Score:\n\n{best_score}"

    # Save to JSON file
    with open('best_params.json', 'w') as fp:
        json.dump(_convert_np_types(best_params), fp, indent=4)

    return str_result