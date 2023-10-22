"""
This module is responsible for handling and modifying the notebook text.
"""
import re

import extensions.superboogav2.parameters as parameters

from modules import shared
from modules.logging_colors import logger
from extensions.superboogav2.utils import create_context_text

from .data_processor import preprocess_text

def _remove_special_tokens(string):
    pattern = r'(<\|begin-user-input\|>|<\|end-user-input\|>|<\|injection-point\|>)'
    return re.sub(pattern, '', string)


def input_modifier_internal(string, collector):
    # Sanity check.
    if shared.is_chat():
        return string

    # Find the user input
    pattern = re.compile(r"<\|begin-user-input\|>(.*?)<\|end-user-input\|>", re.DOTALL)
    match = re.search(pattern, string)
    if match:
        # Preprocess the user prompt.
        user_input = match.group(1).strip()
        user_input = preprocess_text(user_input)

        logger.debug(f"Preprocessed User Input: {user_input}")

        # Get the most similar chunks
        results = collector.get_sorted_by_dist(user_input, n_results=parameters.get_chunk_count(), max_token_count=int(parameters.get_max_token_count()))

        # Make the injection
        string = string.replace('<|injection-point|>', create_context_text(results))

    return _remove_special_tokens(string)