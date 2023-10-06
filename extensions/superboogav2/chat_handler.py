"""
This module is responsible for modifying the chat prompt and history.
"""
import json
import re

import extensions.superboogav2.parameters as parameters

from modules import chat
from modules.text_generation import get_encoded_length
from modules.logging_colors import logger
from extensions.superboogav2.utils import create_context_text, create_metadata_source

from .data_processor import process_and_add_to_collector
from .chromadb import ChromaCollector


CHAT_METADATA = create_metadata_source('automatic-chat-insert')

INSTRUCT_MODE = 'instruct'
CHAT_INSTRUCT_MODE = 'chat-instruct'


def _is_instruct_mode(state: dict):
    mode = state.get('mode')
    return mode == INSTRUCT_MODE or mode == CHAT_INSTRUCT_MODE


def _remove_tag_if_necessary(user_input: str):
    if not parameters.get_is_manual():
        return user_input
    
    return re.sub(r'^\s*!c\s*|\s*!c\s*$', '', user_input)


def _should_query(input: str):
    if not parameters.get_is_manual():
        return True
    
    if re.search(r'^\s*!c|!c\s*$', input, re.MULTILINE):
        return True
    
    return False


def _format_single_exchange(name, text):
    if re.search(r':\s*$', name):
        return '{} {}\n'.format(name, text)
    else:
        return '{}: {}\n'.format(name, text)


def _get_names(state: dict):
    if _is_instruct_mode(state):
        user_name = state['name1_instruct']
        bot_name = state['name2_instruct']
    else:
        user_name = state['name1']
        bot_name = state['name2']

    if not user_name:
        user_name = 'User'
    if not bot_name:
        bot_name = 'Assistant'

    return user_name, bot_name


def _concatinate_history(history: dict, state: dict):
    full_history_text = ''
    user_name, bot_name = _get_names(state)

    # Grab the internal history.
    internal_history = history['internal']
    assert isinstance(internal_history, list)

    # Iterate through the history.
    for exchange in internal_history:
        assert isinstance(exchange, list)

        if len(exchange) >= 1:
            full_history_text += _format_single_exchange(user_name, exchange[0])
        if len(exchange) >= 2:
            full_history_text += _format_single_exchange(bot_name, exchange[1])

    return full_history_text[:-1] # Remove the last new line.


def _hijack_last(context_text: str, history: dict, max_len: int, state: dict):
    num_context_tokens = get_encoded_length(context_text)

    names = _get_names(state)[::-1]

    history_tokens = 0
    replace_position = None
    for i, messages in enumerate(reversed(history['internal'])):
        for j, message in enumerate(reversed(messages)):
            num_message_tokens = get_encoded_length(_format_single_exchange(names[j], message))
            
            # TODO: This is an extremely naive solution. A more robust implementation must be made.
            if history_tokens + num_context_tokens <= max_len:
                # This message can be replaced
                replace_position = (i, j)
            
            history_tokens += num_message_tokens
    
    if replace_position is None:
        logger.warn("The provided context_text is too long to replace any message in the history.")
    else:
        # replace the message at replace_position with context_text
        i, j = replace_position
        history['internal'][-i-1][-j-1] = context_text


def custom_generate_chat_prompt_internal(user_input: str, state: dict, collector: ChromaCollector, **kwargs):
    if parameters.get_add_chat_to_data():
        # Get the whole history as one string
        history_as_text = _concatinate_history(kwargs['history'], state)

        if history_as_text:
            # Delete all documents that were auto-inserted
            collector.delete(ids_to_delete=None, where=CHAT_METADATA)
            # Insert the processed history
            process_and_add_to_collector(history_as_text, collector, False, CHAT_METADATA)

    if _should_query(user_input):
        user_input = _remove_tag_if_necessary(user_input)
        results = collector.get_sorted_by_dist(user_input, n_results=parameters.get_chunk_count(), max_token_count=int(parameters.get_max_token_count()))

        # Check if the strategy is to modify the last message. If so, prepend or append to the user query.
        if parameters.get_injection_strategy() == parameters.APPEND_TO_LAST:
            user_input = user_input + create_context_text(results)
        elif parameters.get_injection_strategy() == parameters.PREPEND_TO_LAST:
            user_input = create_context_text(results) + user_input
        elif parameters.get_injection_strategy() == parameters.HIJACK_LAST_IN_CONTEXT:
            _hijack_last(create_context_text(results), kwargs['history'], state['truncation_length'], state)
    
    return chat.generate_chat_prompt(user_input, state, **kwargs)
