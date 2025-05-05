from typing import Dict, List, Optional, Iterable
from functools import reduce
from operator import getitem
from modules import shared
import modules.chat as chat
from pathlib import Path
import json
import logging as logger

# --- History Storage Logic ---

# Global variable for the currently loaded history data
# Consider moving this to shared state if needed across modules more broadly
_loaded_history = {'visible': [], 'internal': [], '_character': None, '_unique_id': None, '_mode': None}
_mode = 'html'

def validate_list(lst: List, i: int):
    """Ensure list is properly extended to index i"""
    if len(lst) <= i:
        lst.extend([None] * (i + 1 - len(lst)))

def validate_history_structure(history_data: Dict, i: int):
    """Ensure history_data structure is properly extended to index i"""
    for history_type in ['visible', 'internal']:
        if history_type not in history_data:
             history_data[history_type] = []
        validate_list(history_data[history_type], i)


def initialize_history_entry(history_data: Dict, i: int):
    """Initialize history entry dicts at index i if nonexistent"""
    for history_type in ['visible', 'internal']:
        if history_data[history_type][i] is None or not isinstance(history_data[history_type][i], list):
            logger.debug(f"Initializing empty {history_type} history entry at index {i}")
            history_data[history_type][i] = [
                {'text': [], 'pos': 0},  # User message data
                {'text': [], 'pos': 0}   # Bot message data
            ]

def load_or_initialize_history_data(character: Optional[str], unique_id: Optional[str], mode: str) -> Dict:
    """
    Loads history data from file if character and unique_id are provided, the file exists,
    and the requested context (character, unique_id, mode) differs from the currently loaded context.
    Otherwise, returns the existing in-memory history or a new empty history structure.
    """
    global _loaded_history

    # --- Check if requested context matches already loaded context ---
    if (character == _loaded_history.get('_character') and
            unique_id == _loaded_history.get('_unique_id') and
            mode == _loaded_history.get('_mode') and
            _loaded_history.get('_character') is not None):
        logger.debug(f"Requested context matches loaded context ({character}/{unique_id}/{mode}). Returning cached history.")
        return _loaded_history

    # --- Context differs or history not loaded, proceed with load/initialization ---
    logger.debug(f"Context changed or history not loaded. Requested: {character}/{unique_id}/{mode}. Loaded: {_loaded_history.get('_character')}/{_loaded_history.get('_unique_id')}/{_loaded_history.get('_mode')}")

    # Function to create a standard empty history structure with context
    def _create_empty_history(char, uid, md):
        return {'visible': [], 'internal': [], '_character': char, '_unique_id': uid, '_mode': md}

    if not character or not unique_id:
         logger.warning("Cannot load history data: Character or ID is missing.")
         _loaded_history = _create_empty_history(character, unique_id, mode)
         return _loaded_history

    path = get_history_data_path(unique_id, character, mode)
    if not path.exists():
        logger.info(f"Initialized empty message versioning history data for {character}/{unique_id} (file not found)")
        _loaded_history = _create_empty_history(character, unique_id, mode)
        return _loaded_history
    try:
        with open(path, 'r', encoding='utf-8') as f:
            contents = f.read()
            if contents:
                loaded_data = json.loads(contents)
                loaded_data['_character'] = character
                loaded_data['_unique_id'] = unique_id
                loaded_data['_mode'] = mode
                logger.debug(f"Loaded message versioning history data from {path}")
                _loaded_history = loaded_data
                return _loaded_history
            else:
                logger.info(f"Initialized empty message versioning history data (file was empty) for {character}/{unique_id}")
                _loaded_history = _create_empty_history(character, unique_id, mode)
                return _loaded_history
    except json.JSONDecodeError:
         logger.error(f"Error decoding JSON from history data file: {path}. Initializing empty.", exc_info=True)
         _loaded_history = _create_empty_history(character, unique_id, mode)
         return _loaded_history
    except Exception as e:
        logger.error(f"Error loading message versioning history data from {path}: {e}", exc_info=True)
        _loaded_history = _create_empty_history(character, unique_id, mode)
        return _loaded_history


def append_to_history_data(history: Dict, state: Dict, is_bot=True) -> bool:
    """
    Append a message to the end of the currently loaded history data (_loaded_history).
    Requires state to potentially save the updated history.
    """
    global _loaded_history

    # NOTE: Assumes the correct history for the state's character/id is already loaded into _loaded_history

    msg_type = 1 if is_bot else 0
    if not history or not history.get('visible') or not history.get('internal'):
        logger.warning("Attempted to append to history data with invalid input history structure.")
        return False

    i = len(history['visible']) - 1
    if i < 0 or len(history['visible'][i]) <= msg_type or len(history['internal'][i]) <= msg_type:
         logger.warning(f"Input history index {i} or msg_type {msg_type} out of bounds for appending.")
         return False

    visible_text = history['visible'][i][msg_type]
    if not visible_text:
        return False
    internal_text = history['internal'][i][msg_type]

    try:
        validate_history_structure(_loaded_history, i)
        initialize_history_entry(_loaded_history, i)

        # Append the strings to the respective lists in _loaded_history
        if 'text' not in _loaded_history['visible'][i][msg_type]:
             _loaded_history['visible'][i][msg_type]['text'] = []
        if 'text' not in _loaded_history['internal'][i][msg_type]:
             _loaded_history['internal'][i][msg_type]['text'] = []

        vis_list = _loaded_history['visible'][i][msg_type]['text']
        int_list = _loaded_history['internal'][i][msg_type]['text']

        vis_list.append(visible_text)
        int_list.append(internal_text)

        # Update position to the new last index
        new_pos = len(vis_list) - 1
        _loaded_history['visible'][i][msg_type]['pos'] = new_pos
        _loaded_history['internal'][i][msg_type]['pos'] = new_pos

        logger.debug(f"Appended to history data: index={i}, type={msg_type}, new_pos={new_pos}")

        character = state['character_menu']
        unique_id = state['unique_id']
        mode = state['mode']
        if character and unique_id:
             save_history_data(character, unique_id, mode, _loaded_history)
        else:
             logger.warning("Could not save history data after append: character or unique_id missing in state.")

        return True

    except IndexError:
         logger.error(f"IndexError during history data append: index={i}, msg_type={msg_type}. History state: {_loaded_history}", exc_info=True)
    except Exception as e:
        logger.error(f"Error appending to history data: {e}", exc_info=True)
    return False

def save_history_data(character: str, unique_id: str, mode: str, history_data: Dict) -> bool:
    """Save the provided history data structure to disk"""

    if not unique_id or not character:
        logger.warning("Cannot save history data: Character or ID is not set.")
        return False

    current_mode = mode or history_data.get('_mode') or shared.persistent_interface_state.get('mode', 'chat')
    history_data['_mode'] = current_mode

    path = get_history_data_path(unique_id, character, current_mode)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            data_to_save = history_data.copy()
            json.dump(data_to_save, f, indent=4)
        logger.debug(f"Message versioning history data saved to {path}")
        return True
    except TypeError as e:
         logger.error(f"TypeError saving history data (possibly non-serializable data): {e}. Data: {history_data}", exc_info=True)
    except Exception as e:
        logger.error(f"Error saving message versioning history data to {path}: {e}", exc_info=True)
    return False

def get_history_data_path(unique_id: str, character: str, mode: str) -> Path:
    """Get the path to the history data file"""
    if not unique_id or not character or not mode:
        raise ValueError("unique_id, character, and mode must be provided to get history data path.")

    # Use the imported get_history_file_path to determine the base directory
    # This ensures consistency with how text-generation-webui handles history paths
    try:
        # We need the *directory* where the standard history file *would* be.
        base_history_path = chat.get_history_file_path(unique_id, character, mode)
        path = base_history_path.parent
    except Exception as e:
        # Fallback or error handling if get_history_file_path fails (shouldn't happen, but just in case)
        logger.error(f"Failed to get base history path via get_history_file_path: {e}. Using default log directory.", exc_info=True)
        # TODO: Define a fallback path, e.g., logs/boogaplus_storage/character/, and update when get_history_file_path is valid
        # path = Path("logs") / "boogaplus_storage" / character

    path.mkdir(parents=True, exist_ok=True)

    storage_filename = f'{unique_id}.boogaplus'
    final_path = path / storage_filename
    return final_path

# --- Functions to be called by integrated logic ---

def get_loaded_history():
    """Returns the current in-memory loaded history data."""
    return _loaded_history

def set_history_storage_mode(new_mode: str):
    """Sets the display mode ('html', 'overlay', 'off')."""
    global _mode, _loaded_history
    _mode = new_mode # Keep legacy global updated for now
    _loaded_history['_mode'] = new_mode
    logger.debug(f"Message versioning history storage mode set to: {new_mode}")

def get_history_storage_mode():
    """Gets the current display mode from the loaded history."""
    current_mode = _loaded_history.get('_mode', _mode)
    return current_mode

def clear_history_data(unique_id: str, character: str, mode: str):
    """
    Deletes the history data file associated with a history.
    NOTE: This function does NOT clear the in-memory _loaded_history.
    The caller should handle resetting/reloading the in-memory data if the deleted history was the active one.
    """
    try:
        history_data_path = get_history_data_path(unique_id, character, mode)
        if history_data_path.exists():
            history_data_path.unlink()
            logger.info(f"Deleted message versioning history data file: {history_data_path}")
        else:
            logger.warning(f"Message versioning history data file not found for deletion: {history_data_path}")
    except Exception as e:
        logger.error(f"Error deleting message versioning history data file {history_data_path}: {e}", exc_info=True)

def rename_history_data(old_id: str, new_id: str, character: str, mode: str):
    """
    Renames the history data file when a history is renamed.
    NOTE: This function does NOT update the in-memory _loaded_history if the renamed history was the active one.
    The caller should handle reloading the data under the new ID if necessary.
    """
    try:
        old_history_data_path = get_history_data_path(old_id, character, mode)
        new_history_data_path = get_history_data_path(new_id, character, mode)

        if new_history_data_path.exists():
            logger.warning(f"Cannot rename message versioning history data: Target path {new_history_data_path} already exists.")
            return

        if old_history_data_path.exists():
            old_history_data_path.rename(new_history_data_path)
            logger.info(f"Renamed message versioning history data file: {old_history_data_path} -> {new_history_data_path}")
        else:
            logger.warning(f"Message versioning history data file not found for renaming: {old_history_data_path}")

    except Exception as e:
        logger.error(f"Error renaming message versioning history data file from {old_id} to {new_id}: {e}", exc_info=True)


# --- Core Logic ---

_last_history_index = -1
_last_msg_type = -1

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

# --- Core Functions ---

def get_message_positions(history_index: int, msg_type: int):
    """
    Get the message position and total message versions for a specific message from the currently loaded history data.
    Assumes the correct history data has already been loaded by the calling context.
    """
    current_history_data = get_loaded_history()
    # Path: ['visible' or 'internal'][history_index][msg_type]['pos' or 'text']
    msg_data = recursive_get(current_history_data, ['visible', history_index, msg_type])

    if msg_data is None:
        logger.warning(f"History data not found for index {history_index}, type {msg_type}. Was history loaded correctly before calling this?")
        return 0, 0

    pos = recursive_get(msg_data, ['pos'], 0)
    total = length(recursive_get(msg_data, ['text'], []))
    return pos, total

def navigate_message_version(history_index: float, msg_type: float, direction: str, state: Dict):
    """
    Handles navigation (left/right swipe) for a message version.
    Updates the history dictionary directly and returns the modified history.
    The calling function will be responsible for regenerating the HTML.
    """
    global _last_history_index, _last_msg_type
    try:
        i = int(history_index)
        m_type = int(msg_type)
        _last_history_index = i  # Default to current index/type
        _last_msg_type = m_type

        load_or_initialize_history_data(state['character_menu'], state['unique_id'], state['mode'])
        current_pos, total_pos = get_message_positions(i, m_type)

        if total_pos <= 1:
            return state['history']

        # Calculate new position
        new_pos = current_pos
        if direction == "right":
            new_pos += 1
            if new_pos >= total_pos:
                return state['history']
        elif direction == "left":
            new_pos -= 1
            if new_pos < 0:
                return state['history']
        else:
            logger.warning(f"Invalid navigation direction: {direction}")
            return state['history']

        _loaded_history_data = get_loaded_history()

        visible_msg_data = recursive_get(_loaded_history_data, ['visible', i, m_type])
        internal_msg_data = recursive_get(_loaded_history_data, ['internal', i, m_type])

        if not visible_msg_data or not internal_msg_data or 'text' not in visible_msg_data or 'text' not in internal_msg_data:
            logger.error(f"Loaded history data structure invalid during navigation for index {i}, type {m_type}. Data: {_loaded_history_data}")
            return state['history']

        # Check bounds for the new position against the text lists
        if new_pos < 0 or new_pos >= len(visible_msg_data['text']):
            logger.error(f"Navigation error: new_pos {new_pos} out of bounds for visible history text list (len {len(visible_msg_data['text'])})")
            return state['history']
        if new_pos < 0 or new_pos >= len(internal_msg_data['text']):
            logger.error(f"Navigation error: new_pos {new_pos} out of bounds for internal history text list (len {len(internal_msg_data['text'])})")
            return state['history']

        new_visible = visible_msg_data['text'][new_pos]
        new_internal = internal_msg_data['text'][new_pos]

        # Update the position ('pos') within the loaded history data structure
        visible_msg_data['pos'] = new_pos
        internal_msg_data['pos'] = new_pos

        # Save the updated history data back to the file
        character = state['character_menu']
        unique_id = state['unique_id']
        mode = state['mode']
        if character and unique_id:
            save_history_data(character, unique_id, mode, _loaded_history_data)
        else:
            logger.warning("Could not save history data after navigation: character or unique_id missing in state.")

        # Update the main chat history dictionary in the state
        current_history = state['history']
        if i < len(current_history['visible']) and m_type < len(current_history['visible'][i]):
            current_history['visible'][i][m_type] = new_visible
        else:
             logger.error(f"History structure invalid (visible) for update at index {i}, type {m_type}")

        if i < len(current_history['internal']) and m_type < len(current_history['internal'][i]):
            current_history['internal'][i][m_type] = new_internal
        else:
             logger.error(f"History structure invalid (internal) for update at index {i}, type {m_type}")
            
        _last_history_index = i
        _last_msg_type = m_type

        logger.debug(f"Navigation successful: index={i}, type={m_type}, direction={direction}, old_pos={current_pos}/{total_pos-1}, new_pos={new_pos}/{total_pos-1}")
        return current_history

    except Exception as e:
        logger.error(f"Error during message version navigation: {e}", exc_info=True)
        return state.get('history', {'visible': [], 'internal': []})


# --- Functions to be called during HTML Generation ---

def get_message_version_nav_elements(history_index: int, msg_type: int):
    """
    Generates the HTML for the message version navigation arrows and position indicator.
    Should be called by the HTML generator.
    Returns an empty string if history storage mode is 'off' or navigation is not needed.
    """
    if get_history_storage_mode() == 'off':
        return ""

    try:
        current_pos, total_pos = get_message_positions(history_index, msg_type)

        if total_pos <= 1:
            return ""

        left_activated_attr = f' activated="true" data-history-index="{history_index}" data-msg-type="{msg_type}" data-direction="left"' if current_pos > 0 else ''
        right_activated_attr = f' activated="true" data-history-index="{history_index}" data-msg-type="{msg_type}" data-direction="right"' if current_pos < total_pos - 1 else ''

        # Similar size to copy/refresh icons
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
    global _last_history_index, _last_msg_type
    return _last_history_index == history_index and _last_msg_type == msg_type


# --- Functions to be called by Gradio Event Handlers ---

def handle_unique_id_select(unique_id: str, state: Dict):
    """
    Handles the selection of a unique_id from the dropdown.
    Loads the corresponding message versioning history data.
    """
    global _last_history_index, _last_msg_type
    # print(f"DEBUG: Entering handle_unique_id_select with unique_id: {unique_id}") # Keep print for debug?
    _last_history_index = -1
    _last_msg_type = -1
    
    logger.debug(f"Handling unique_id selection: {unique_id}")
    if not unique_id:
        logger.warning("handle_unique_id_select called with empty unique_id.")
        return

    # Load/initialize history data for the selected unique_id
    character = state['character_menu']
    mode = state['mode']

    if not character:
        logger.warning(f"Cannot load history for unique_id {unique_id}: character_menu not found in state.")
        # Avoid using stale data from a previous selection
        load_or_initialize_history_data(None, unique_id, mode)
    else:
        load_or_initialize_history_data(character, unique_id, mode)

def handle_message_versioning_change_display_mode(display_mode: str, state: Dict):
    """
    Handles changes to the message versioning display mode radio button.
    Updates the history storage mode. The UI redraw should be triggered by the main logic.
    """
    logger.debug(f"Handling message versioning display mode change to: {display_mode}")
    set_history_storage_mode(display_mode)

def handle_message_versioning_navigate_click(history_index: float, msg_type: float, direction: str, state: Dict):
    """
    Handles clicks on the navigation buttons generated by get_message_version_nav_elements.
    Calls navigate_message_version and returns the updated history and potentially the regenerated HTML.
    This function will be the target of the Gradio `message_versioning_navigate_hidden` button's click event.
    """
    logger.debug(f"Handling message versioning navigate click: index={history_index}, type={msg_type}, direction={direction}")

    # Modifies history in-place
    updated_history = navigate_message_version(history_index, msg_type, direction, state)

    import modules.chat as chat

    new_html = chat.redraw_html(
        updated_history,
        state['name1'],
        state['name2'],
        state['mode'],
        state['chat_style'],
        state['character_menu']
    )
    
    return updated_history, new_html
