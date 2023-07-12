import traceback
from functools import partial

import gradio as gr

import extensions
import modules.shared as shared
from modules.logging_colors import logger
from inspect import signature


state = {}
available_extensions = []
setup_called = set()


def apply_settings(extension, name):
    if not hasattr(extension, 'params'):
        return

    for param in extension.params:
        _id = f"{name}-{param}"
        if _id not in shared.settings:
            continue

        extension.params[param] = shared.settings[_id]


def load_extensions():
    global state, setup_called
    for i, name in enumerate(shared.args.extensions):
        if name in available_extensions:
            if name != 'api':
                logger.info(f'Loading the extension "{name}"...')
            try:
                exec(f"import extensions.{name}.script")
                extension = getattr(extensions, name).script
                apply_settings(extension, name)
                if extension not in setup_called and hasattr(extension, "setup"):
                    setup_called.add(extension)
                    extension.setup()

                state[name] = [True, i]
            except:
                logger.error(f'Failed to load the extension "{name}".')
                traceback.print_exc()


# This iterator returns the extensions in the order specified in the command-line
def iterator():
    for name in sorted(state, key=lambda x: state[x][1]):
        if state[name][0]:
            yield getattr(extensions, name).script, name


# Extension functions that map string -> string
def _apply_string_extensions(function_name, text, state):
    for extension, _ in iterator():
        if hasattr(extension, function_name):
            func = getattr(extension, function_name)
            if len(signature(func).parameters) == 2:
                text = func(text, state)
            else:
                text = func(text)

    return text


# Input hijack of extensions
def _apply_input_hijack(text, visible_text):
    for extension, _ in iterator():
        if hasattr(extension, 'input_hijack') and extension.input_hijack['state']:
            extension.input_hijack['state'] = False
            if callable(extension.input_hijack['value']):
                text, visible_text = extension.input_hijack['value'](text, visible_text)
            else:
                text, visible_text = extension.input_hijack['value']

    return text, visible_text


# custom_generate_chat_prompt handling - currently only the first one will work
def _apply_custom_generate_chat_prompt(text, state, **kwargs):
    for extension, _ in iterator():
        if hasattr(extension, 'custom_generate_chat_prompt'):
            return extension.custom_generate_chat_prompt(text, state, **kwargs)

    return None


# Extension that modifies the input parameters before they are used
def _apply_state_modifier_extensions(state):
    for extension, _ in iterator():
        if hasattr(extension, "state_modifier"):
            state = getattr(extension, "state_modifier")(state)

    return state


# Extension that modifies the chat history before it is used
def _apply_history_modifier_extensions(history):
    for extension, _ in iterator():
        if hasattr(extension, "history_modifier"):
            history = getattr(extension, "history_modifier")(history)

    return history


# Extension functions that override the default tokenizer output - currently only the first one will work
def _apply_tokenizer_extensions(function_name, state, prompt, input_ids, input_embeds):
    for extension, _ in iterator():
        if hasattr(extension, function_name):
            return getattr(extension, function_name)(state, prompt, input_ids, input_embeds)

    return prompt, input_ids, input_embeds


# Get prompt length in tokens after applying extension functions which override the default tokenizer output
# currently only the first one will work
def _apply_custom_tokenized_length(prompt):
    for extension, _ in iterator():
        if hasattr(extension, 'custom_tokenized_length'):
            return getattr(extension, 'custom_tokenized_length')(prompt)

    return None


# Custom generate reply handling - currently only the first one will work
def _apply_custom_generate_reply():
    for extension, _ in iterator():
        if hasattr(extension, 'custom_generate_reply'):
            return getattr(extension, 'custom_generate_reply')

    return None


def _apply_custom_css():
    all_css = ''
    for extension, _ in iterator():
        if hasattr(extension, 'custom_css'):
            all_css += getattr(extension, 'custom_css')()

    return all_css


def _apply_custom_js():
    all_js = ''
    for extension, _ in iterator():
        if hasattr(extension, 'custom_js'):
            all_js += getattr(extension, 'custom_js')()

    return all_js


def create_extensions_block():
    to_display = []
    for extension, name in iterator():
        if hasattr(extension, "ui") and not (hasattr(extension, 'params') and extension.params.get('is_tab', False)):
            to_display.append((extension, name))

    # Creating the extension ui elements
    if len(to_display) > 0:
        with gr.Column(elem_id="extensions"):
            for row in to_display:
                extension, name = row
                display_name = getattr(extension, 'params', {}).get('display_name', name)
                gr.Markdown(f"\n### {display_name}")
                extension.ui()


def create_extensions_tabs():
    for extension, name in iterator():
        if hasattr(extension, "ui") and (hasattr(extension, 'params') and extension.params.get('is_tab', False)):
            display_name = getattr(extension, 'params', {}).get('display_name', name)
            with gr.Tab(display_name, elem_classes="extension-tab"):
                extension.ui()


EXTENSION_MAP = {
    "input": partial(_apply_string_extensions, "input_modifier"),
    "output": partial(_apply_string_extensions, "output_modifier"),
    "state": _apply_state_modifier_extensions,
    "history": _apply_history_modifier_extensions,
    "bot_prefix": partial(_apply_string_extensions, "bot_prefix_modifier"),
    "tokenizer": partial(_apply_tokenizer_extensions, "tokenizer_modifier"),
    "input_hijack": _apply_input_hijack,
    "custom_generate_chat_prompt": _apply_custom_generate_chat_prompt,
    "custom_generate_reply": _apply_custom_generate_reply,
    "tokenized_length": _apply_custom_tokenized_length,
    "css": _apply_custom_css,
    "js": _apply_custom_js
}


def apply_extensions(typ, *args, **kwargs):
    if typ not in EXTENSION_MAP:
        raise ValueError(f"Invalid extension type {typ}")

    return EXTENSION_MAP[typ](*args, **kwargs)
