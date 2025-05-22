from typing import Dict, List, Optional, Iterable
from functools import reduce
from operator import getitem
from modules import shared
from pathlib import Path
import json
import logging as logger

last_state = {
    'display_mode': 'html',  # Default display mode
}

# --- Functions to be called by integrated logic ---


def set_versioning_display_mode(new_dmode: str):
    """Sets the display mode ('html', 'overlay', 'off')."""
    global last_state
    last_state['display_mode'] = new_dmode


def get_versioning_display_mode():
    """Gets the current display mode from the loaded history."""
    current_mode = last_state.get('display_mode', 'html')
    return current_mode



# --- Core Logic ---

last_history_index = -1
last_msg_type = -1


# --- Helper Functions ---

def recursive_get(data: Dict | Iterable, keyList: List[int | str], default=None):
    """Iterate nested dictionary / iterable safely."""
    try:
        result = reduce(getitem, keyList, data)
        return result
    except (KeyError, IndexError, TypeError):
        return default
    except Exception as e:
        logger.error(f"Error in recursive_get: {e}", exc_info=True)
        return None


def length(data: Iterable):
    """Get length of iterable with try-catch."""
    try:
        result = len(data)
        return result
    except TypeError:
        return 0
    except Exception as e:
        logger.error(f"Error getting length: {e}", exc_info=True)
        return 0


def get_message_metadata(history: dict, history_index: int, msg_type: int) -> dict:
    msg_role = 'assistant' if msg_type == 1 else 'user'
    return recursive_get(history, ['metadata', f'{msg_role}_{history_index}'])

# --- Core Functions ---

def get_message_positions(history: dict, history_index: int, msg_type: int):
    """
    Get the message position and total message versions for a specific message from the currently loaded history data.
    Assumes the correct history data has already been loaded by the calling context.
    """
    msg_data = get_message_metadata(history, history_index, msg_type)

    if msg_data is None:
        return 0, 0

    pos = msg_data.get('current_message_index', 0)
    total = length(msg_data.get('versions', []))
    return pos, total


def navigate_message_version(history_index: float, msg_type: float, direction: str, history: Dict, character: str, unique_id: str, mode: str):
    """
    Handles navigation (left/right swipe) for a message version.
    Updates the history dictionary directly and returns the modified history.
    The calling function will be responsible for regenerating the HTML.
    """
    global last_history_index, last_msg_type
    try:
        i = int(history_index)
        m_type = int(msg_type)
        last_history_index = i  # Default to current index/type
        last_msg_type = m_type

        current_pos, total_pos = get_message_positions(history, i, m_type)

        if total_pos <= 1:
            return history

        # Calculate new position
        new_pos = current_pos
        if direction == "right":
            new_pos += 1
            if new_pos >= total_pos:
                return history
        elif direction == "left":
            new_pos -= 1
            if new_pos < 0:
                return history
        else:
            logger.warning(f"Invalid navigation direction: {direction}")
            return history

        
        history_metadata = history['metadata']
        msg_metadata = get_message_metadata(history, i, m_type)
        if msg_metadata is None:
            logger.error(f"Message metadata not found for index {i}, type {m_type}. Data: {history_metadata}")
            return history
        versions = msg_metadata.get('versions', [])
        new_ver = versions[new_pos]
        if new_ver is None:
            logger.error(f"Message version not found for metadata index {i}, type {m_type}, version index {new_pos}. Data: {history_metadata}")
            return history
        
        ver_content = new_ver['content']
        visible_ver_content = new_ver['visible_content']

        if ver_content is None or visible_ver_content is None:
            logger.error(f"Loaded version data structure invalid during navigation for index {i}, type {m_type}. Data: {history_metadata}")
            return history

        # Check bounds for the new position against the text lists
        if new_pos < 0 or new_pos >= len(msg_metadata['versions']):
            logger.error(f"Navigation error: new_pos {new_pos} out of bounds for history text list (len {len(versions)})")
            return history

        msg_metadata['current_message_index'] = new_pos  # Updates history object

        last_history_index = i
        last_msg_type = m_type

        return history

    except Exception as e:
        logger.error(f"Error during message version navigation: {e}", exc_info=True)
        return history


# --- Functions to be called during HTML Generation ---


def get_message_version_nav_elements(history: dict, history_index: int, msg_type: int):
    """
    Generates the HTML for the message version navigation arrows and position indicator.
    Should be called by the HTML generator.
    Returns an empty string if history storage mode is 'off' or navigation is not needed.
    """
    if get_versioning_display_mode() == 'off':
        return ""

    try:
        current_pos, total_pos = get_message_positions(history, history_index, msg_type)

        if total_pos <= 1:
            return ""

        left_activated_attr = f' activated="true"' if current_pos > 0 else ''
        right_activated_attr = f' activated="true"' if current_pos < total_pos - 1 else ''

        left_arrow_svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 18 9 12 15 6"></polyline></svg>'''
        right_arrow_svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"></polyline></svg>'''

        nav_left = f'<button class="message-versioning-nav-arrow message-versioning-nav-left footer-button"{left_activated_attr}>{left_arrow_svg}</button>'
        nav_pos = f'<div class="message-versioning-nav-pos">{current_pos + 1}/{total_pos}</div>'
        nav_right = f'<button class="message-versioning-nav-arrow message-versioning-nav-right footer-button"{right_activated_attr}>{right_arrow_svg}</button>'

        nav_container = (
            f'<div class="message-versioning-nav-container">'
            f'{nav_left}'
            f'{nav_pos}'
            f'{nav_right}'
            f'</div>'
        )

        versioning_container = (
            f'<div class="message-versioning-container">'
            f'{nav_container}'
            f'</div>'
        )
        return versioning_container

    except Exception as e:
        logger.error(f"Error generating message versioning nav elements for index {history_index}, type {msg_type}: {e}", exc_info=True)
        return ""


def is_message_selected(history_index: int, msg_type: int) -> bool:
    """
    Returns True if the message at the specified index and type is selected.
    """
    global last_history_index, last_msg_type
    return last_history_index == history_index and last_msg_type == msg_type


def handle_display_mode_change(display_mode: str):
    """
    Handles changes to the message versioning display mode radio button.
    Updates the history storage mode. The UI redraw should be triggered by the main logic.
    """
    set_versioning_display_mode(display_mode)


def handle_navigate_click(history_index: float, msg_type: float, direction: str, history: Dict, character: str, unique_id: str, mode: str, name1: str, name2: str, chat_style: str):
    """
    Handles clicks on the navigation buttons generated by get_message_version_nav_elements.
    Calls navigate_message_version and returns the updated history and potentially the regenerated HTML.
    """
    updated_history = navigate_message_version(history_index, msg_type, direction, history, character, unique_id, mode)

    import modules.chat as chat
    new_html = chat.redraw_html(updated_history, name1, name2, mode, chat_style, character)

    return updated_history, new_html
