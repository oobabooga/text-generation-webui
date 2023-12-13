import base64
import copy
import functools
import html
import json
import re
from datetime import datetime
from functools import partial
from pathlib import Path

import gradio as gr
import yaml
from jinja2.sandbox import ImmutableSandboxedEnvironment
from PIL import Image

import modules.shared as shared
from modules.extensions import apply_extensions
from modules.html_generator import chat_html_wrapper, make_thumbnail
from modules.logging_colors import logger
from modules.text_generation import (
    generate_reply,
    get_encoded_length,
    get_max_prompt_length
)
from modules.utils import delete_file, get_available_characters, save_file

# Copied from the Transformers library
jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)


def str_presenter(dumper, data):
    """
    Copied from https://github.com/yaml/pyyaml/issues/240
    Makes pyyaml output prettier multiline strings.
    """

    if data.count('\n') > 0:
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')

    return dumper.represent_scalar('tag:yaml.org,2002:str', data)


yaml.add_representer(str, str_presenter)
yaml.representer.SafeRepresenter.add_representer(str, str_presenter)


def get_generation_prompt(renderer, impersonate=False, strip_trailing_spaces=True):
    '''
    Given a Jinja template, reverse-engineers the prefix and the suffix for
    an assistant message (if impersonate=False) or an user message
    (if impersonate=True)
    '''

    if impersonate:
        messages = [
            {"role": "user", "content": "<<|user-message-1|>>"},
            {"role": "user", "content": "<<|user-message-2|>>"},
        ]
    else:
        messages = [
            {"role": "assistant", "content": "<<|user-message-1|>>"},
            {"role": "assistant", "content": "<<|user-message-2|>>"},
        ]

    prompt = renderer(messages=messages)

    suffix_plus_prefix = prompt.split("<<|user-message-1|>>")[1].split("<<|user-message-2|>>")[0]
    suffix = prompt.split("<<|user-message-2|>>")[1]
    prefix = suffix_plus_prefix[len(suffix):]

    if strip_trailing_spaces:
        prefix = prefix.rstrip(' ')

    return prefix, suffix


def generate_chat_prompt(user_input, state, **kwargs):
    impersonate = kwargs.get('impersonate', False)
    _continue = kwargs.get('_continue', False)
    also_return_rows = kwargs.get('also_return_rows', False)
    history = kwargs.get('history', state['history'])['internal']

    # Templates
    chat_template = jinja_env.from_string(state['chat_template_str'])
    instruction_template = jinja_env.from_string(state['instruction_template_str'])
    chat_renderer = partial(chat_template.render, add_generation_prompt=False, name1=state['name1'], name2=state['name2'])
    instruct_renderer = partial(instruction_template.render, add_generation_prompt=False)

    messages = []

    if state['mode'] == 'instruct':
        renderer = instruct_renderer
        if state['custom_system_message'].strip() != '':
            messages.append({"role": "system", "content": state['custom_system_message']})
    else:
        renderer = chat_renderer
        if state['context'].strip() != '':
            messages.append({"role": "system", "content": state['context']})

    insert_pos = len(messages)
    for user_msg, assistant_msg in reversed(history):
        user_msg = user_msg.strip()
        assistant_msg = assistant_msg.strip()

        if assistant_msg:
            messages.insert(insert_pos, {"role": "assistant", "content": assistant_msg})

        if user_msg not in ['', '<|BEGIN-VISIBLE-CHAT|>']:
            messages.insert(insert_pos, {"role": "user", "content": user_msg})

    user_input = user_input.strip()
    if user_input and not impersonate and not _continue:
        messages.append({"role": "user", "content": user_input})

    def make_prompt(messages):
        if state['mode'] == 'chat-instruct' and _continue:
            prompt = renderer(messages=messages[:-1])
        else:
            prompt = renderer(messages=messages)

        if state['mode'] == 'chat-instruct':
            outer_messages = []
            if state['custom_system_message'].strip() != '':
                outer_messages.append({"role": "system", "content": state['custom_system_message']})

            command = state['chat-instruct_command']
            command = command.replace('<|character|>', state['name2'] if not impersonate else state['name1'])
            command = command.replace('<|prompt|>', prompt)

            if _continue:
                prefix = get_generation_prompt(renderer, impersonate=impersonate, strip_trailing_spaces=False)[0]
                prefix += messages[-1]["content"]
            else:
                prefix = get_generation_prompt(renderer, impersonate=impersonate)[0]
                if not impersonate:
                    prefix = apply_extensions('bot_prefix', prefix, state)

            outer_messages.append({"role": "user", "content": command})
            outer_messages.append({"role": "assistant", "content": prefix})

            prompt = instruction_template.render(messages=outer_messages)
            suffix = get_generation_prompt(instruct_renderer, impersonate=False)[1]
            prompt = prompt[:-len(suffix)]

        else:
            if _continue:
                suffix = get_generation_prompt(renderer, impersonate=impersonate)[1]
                prompt = prompt[:-len(suffix)]
            else:
                prefix = get_generation_prompt(renderer, impersonate=impersonate)[0]
                if state['mode'] == 'chat' and not impersonate:
                    prefix = apply_extensions('bot_prefix', prefix, state)

                prompt += prefix

        return prompt

    prompt = make_prompt(messages)

    # Handle truncation
    max_length = get_max_prompt_length(state)
    while len(messages) > 0 and get_encoded_length(prompt) > max_length:
        # Try to save the system message
        if len(messages) > 1 and messages[0]['role'] == 'system':
            messages.pop(1)
        else:
            messages.pop(0)

        prompt = make_prompt(messages)

    if also_return_rows:
        return prompt, [message['content'] for message in messages]
    else:
        return prompt


def get_stopping_strings(state):
    stopping_strings = []
    renderers = []

    if state['mode'] in ['instruct', 'chat-instruct']:
        template = jinja_env.from_string(state['instruction_template_str'])
        renderer = partial(template.render, add_generation_prompt=False)
        renderers.append(renderer)

    if state['mode'] in ['chat', 'chat-instruct']:
        template = jinja_env.from_string(state['chat_template_str'])
        renderer = partial(template.render, add_generation_prompt=False, name1=state['name1'], name2=state['name2'])
        renderers.append(renderer)

    for renderer in renderers:
        prefix_bot, suffix_bot = get_generation_prompt(renderer, impersonate=False)
        prefix_user, suffix_user = get_generation_prompt(renderer, impersonate=True)

        stopping_strings += [
            suffix_user + prefix_bot,
            suffix_user + prefix_user,
            suffix_bot + prefix_bot,
            suffix_bot + prefix_user,
        ]

    if 'stopping_strings' in state and isinstance(state['stopping_strings'], list):
        stopping_strings += state.pop('stopping_strings')

    return list(set(stopping_strings))


def chatbot_wrapper(text, state, regenerate=False, _continue=False, loading_message=True, for_ui=False):
    history = state['history']
    output = copy.deepcopy(history)
    output = apply_extensions('history', output)
    state = apply_extensions('state', state)
    if shared.model_name == 'None' or shared.model is None:
        logger.error("No model is loaded! Select one in the Model tab.")
        yield output
        return

    just_started = True
    visible_text = None
    stopping_strings = get_stopping_strings(state)
    is_stream = state['stream']

    # Prepare the input
    if not any((regenerate, _continue)):
        visible_text = html.escape(text)

        # Apply extensions
        text, visible_text = apply_extensions('chat_input', text, visible_text, state)
        text = apply_extensions('input', text, state, is_chat=True)

        # *Is typing...*
        if loading_message:
            yield {'visible': output['visible'] + [[visible_text, shared.processing_message]], 'internal': output['internal']}
    else:
        text, visible_text = output['internal'][-1][0], output['visible'][-1][0]
        if regenerate:
            output['visible'].pop()
            output['internal'].pop()

            # *Is typing...*
            if loading_message:
                yield {'visible': output['visible'] + [[visible_text, shared.processing_message]], 'internal': output['internal']}
        elif _continue:
            last_reply = [output['internal'][-1][1], output['visible'][-1][1]]
            if loading_message:
                yield {'visible': output['visible'][:-1] + [[visible_text, last_reply[1] + '...']], 'internal': output['internal']}

    # Generate the prompt
    kwargs = {
        '_continue': _continue,
        'history': output,
    }
    prompt = apply_extensions('custom_generate_chat_prompt', text, state, **kwargs)
    if prompt is None:
        prompt = generate_chat_prompt(text, state, **kwargs)

    # Generate
    reply = None
    for j, reply in enumerate(generate_reply(prompt, state, stopping_strings=stopping_strings, is_chat=True, for_ui=for_ui)):

        # Extract the reply
        visible_reply = reply
        if state['mode'] in ['chat', 'chat-instruct']:
            visible_reply = re.sub("(<USER>|<user>|{{user}})", state['name1'], reply)

        visible_reply = html.escape(visible_reply)

        if shared.stop_everything:
            output['visible'][-1][1] = apply_extensions('output', output['visible'][-1][1], state, is_chat=True)
            yield output
            return

        if just_started:
            just_started = False
            if not _continue:
                output['internal'].append(['', ''])
                output['visible'].append(['', ''])

        if _continue:
            output['internal'][-1] = [text, last_reply[0] + reply]
            output['visible'][-1] = [visible_text, last_reply[1] + visible_reply]
            if is_stream:
                yield output
        elif not (j == 0 and visible_reply.strip() == ''):
            output['internal'][-1] = [text, reply.lstrip(' ')]
            output['visible'][-1] = [visible_text, visible_reply.lstrip(' ')]
            if is_stream:
                yield output

    output['visible'][-1][1] = apply_extensions('output', output['visible'][-1][1], state, is_chat=True)
    yield output


def impersonate_wrapper(text, state):

    static_output = chat_html_wrapper(state['history'], state['name1'], state['name2'], state['mode'], state['chat_style'], state['character_menu'])

    if shared.model_name == 'None' or shared.model is None:
        logger.error("No model is loaded! Select one in the Model tab.")
        yield '', static_output
        return

    prompt = generate_chat_prompt('', state, impersonate=True)
    stopping_strings = get_stopping_strings(state)

    yield text + '...', static_output
    reply = None
    for reply in generate_reply(prompt + text, state, stopping_strings=stopping_strings, is_chat=True):
        yield (text + reply).lstrip(' '), static_output
        if shared.stop_everything:
            return


def generate_chat_reply(text, state, regenerate=False, _continue=False, loading_message=True, for_ui=False):
    history = state['history']
    if regenerate or _continue:
        text = ''
        if (len(history['visible']) == 1 and not history['visible'][0][0]) or len(history['internal']) == 0:
            yield history
            return

    for history in chatbot_wrapper(text, state, regenerate=regenerate, _continue=_continue, loading_message=loading_message, for_ui=for_ui):
        yield history


def character_is_loaded(state, raise_exception=False):
    if state['mode'] in ['chat', 'chat-instruct'] and state['name2'] == '':
        logger.error('It looks like no character is loaded. Please load one under Parameters > Character.')
        if raise_exception:
            raise ValueError

        return False
    else:
        return True


def generate_chat_reply_wrapper(text, state, regenerate=False, _continue=False):
    '''
    Same as above but returns HTML for the UI
    '''

    if not character_is_loaded(state):
        return

    if state['start_with'] != '' and not _continue:
        if regenerate:
            text, state['history'] = remove_last_message(state['history'])
            regenerate = False

        _continue = True
        send_dummy_message(text, state)
        send_dummy_reply(state['start_with'], state)

    for i, history in enumerate(generate_chat_reply(text, state, regenerate, _continue, loading_message=True, for_ui=True)):
        yield chat_html_wrapper(history, state['name1'], state['name2'], state['mode'], state['chat_style'], state['character_menu']), history


def remove_last_message(history):
    if len(history['visible']) > 0 and history['internal'][-1][0] != '<|BEGIN-VISIBLE-CHAT|>':
        last = history['visible'].pop()
        history['internal'].pop()
    else:
        last = ['', '']

    return html.unescape(last[0]), history


def send_last_reply_to_input(history):
    if len(history['visible']) > 0:
        return html.unescape(history['visible'][-1][1])
    else:
        return ''


def replace_last_reply(text, state):
    history = state['history']

    if len(text.strip()) == 0:
        return history
    elif len(history['visible']) > 0:
        history['visible'][-1][1] = html.escape(text)
        history['internal'][-1][1] = apply_extensions('input', text, state, is_chat=True)

    return history


def send_dummy_message(text, state):
    history = state['history']
    history['visible'].append([html.escape(text), ''])
    history['internal'].append([apply_extensions('input', text, state, is_chat=True), ''])
    return history


def send_dummy_reply(text, state):
    history = state['history']
    if len(history['visible']) > 0 and not history['visible'][-1][1] == '':
        history['visible'].append(['', ''])
        history['internal'].append(['', ''])

    history['visible'][-1][1] = html.escape(text)
    history['internal'][-1][1] = apply_extensions('input', text, state, is_chat=True)
    return history


def redraw_html(history, name1, name2, mode, style, character, reset_cache=False):
    return chat_html_wrapper(history, name1, name2, mode, style, character, reset_cache=reset_cache)


def start_new_chat(state):
    mode = state['mode']
    history = {'internal': [], 'visible': []}

    if mode != 'instruct':
        greeting = replace_character_names(state['greeting'], state['name1'], state['name2'])
        if greeting != '':
            history['internal'] += [['<|BEGIN-VISIBLE-CHAT|>', greeting]]
            history['visible'] += [['', apply_extensions('output', greeting, state, is_chat=True)]]

    unique_id = datetime.now().strftime('%Y%m%d-%H-%M-%S')
    save_history(history, unique_id, state['character_menu'], state['mode'])

    return history


def get_history_file_path(unique_id, character, mode):
    if mode == 'instruct':
        p = Path(f'logs/instruct/{unique_id}.json')
    else:
        p = Path(f'logs/chat/{character}/{unique_id}.json')

    return p


def save_history(history, unique_id, character, mode):
    if shared.args.multi_user:
        return

    p = get_history_file_path(unique_id, character, mode)
    if not p.parent.is_dir():
        p.parent.mkdir(parents=True)

    with open(p, 'w', encoding='utf-8') as f:
        f.write(json.dumps(history, indent=4))


def rename_history(old_id, new_id, character, mode):
    if shared.args.multi_user:
        return

    old_p = get_history_file_path(old_id, character, mode)
    new_p = get_history_file_path(new_id, character, mode)
    if new_p.parent != old_p.parent:
        logger.error(f"The following path is not allowed: {new_p}.")
    elif new_p == old_p:
        logger.info("The provided path is identical to the old one.")
    else:
        logger.info(f"Renaming {old_p} to {new_p}")
        old_p.rename(new_p)


def find_all_histories(state):
    if shared.args.multi_user:
        return ['']

    if state['mode'] == 'instruct':
        paths = Path('logs/instruct').glob('*.json')
    else:
        character = state['character_menu']

        # Handle obsolete filenames and paths
        old_p = Path(f'logs/{character}_persistent.json')
        new_p = Path(f'logs/persistent_{character}.json')
        if old_p.exists():
            logger.warning(f"Renaming {old_p} to {new_p}")
            old_p.rename(new_p)
        if new_p.exists():
            unique_id = datetime.now().strftime('%Y%m%d-%H-%M-%S')
            p = get_history_file_path(unique_id, character, state['mode'])
            logger.warning(f"Moving {new_p} to {p}")
            p.parent.mkdir(exist_ok=True)
            new_p.rename(p)

        paths = Path(f'logs/chat/{character}').glob('*.json')

    histories = sorted(paths, key=lambda x: x.stat().st_mtime, reverse=True)
    histories = [path.stem for path in histories]

    return histories


def load_latest_history(state):
    '''
    Loads the latest history for the given character in chat or chat-instruct
    mode, or the latest instruct history for instruct mode.
    '''

    if shared.args.multi_user:
        return start_new_chat(state)

    histories = find_all_histories(state)

    if len(histories) > 0:
        unique_id = Path(histories[0]).stem
        history = load_history(unique_id, state['character_menu'], state['mode'])
    else:
        history = start_new_chat(state)

    return history


def load_history(unique_id, character, mode):
    p = get_history_file_path(unique_id, character, mode)

    f = json.loads(open(p, 'rb').read())
    if 'internal' in f and 'visible' in f:
        history = f
    else:
        history = {
            'internal': f['data'],
            'visible': f['data_visible']
        }

    return history


def load_history_json(file, history):
    try:
        file = file.decode('utf-8')
        f = json.loads(file)
        if 'internal' in f and 'visible' in f:
            history = f
        else:
            history = {
                'internal': f['data'],
                'visible': f['data_visible']
            }

        return history
    except:
        return history


def delete_history(unique_id, character, mode):
    p = get_history_file_path(unique_id, character, mode)
    delete_file(p)


def replace_character_names(text, name1, name2):
    text = text.replace('{{user}}', name1).replace('{{char}}', name2)
    return text.replace('<USER>', name1).replace('<BOT>', name2)


def generate_pfp_cache(character):
    cache_folder = Path("cache")
    if not cache_folder.exists():
        cache_folder.mkdir()

    for path in [Path(f"characters/{character}.{extension}") for extension in ['png', 'jpg', 'jpeg']]:
        if path.exists():
            original_img = Image.open(path)
            original_img.save(Path('cache/pfp_character.png'), format='PNG')

            thumb = make_thumbnail(original_img)
            thumb.save(Path('cache/pfp_character_thumb.png'), format='PNG')

            return thumb

    return None


def load_character(character, name1, name2):
    context = greeting = ""
    greeting_field = 'greeting'
    picture = None

    filepath = None
    for extension in ["yml", "yaml", "json"]:
        filepath = Path(f'characters/{character}.{extension}')
        if filepath.exists():
            break

    if filepath is None or not filepath.exists():
        logger.error(f"Could not find the character \"{character}\" inside characters/. No character has been loaded.")
        raise ValueError

    file_contents = open(filepath, 'r', encoding='utf-8').read()
    data = json.loads(file_contents) if extension == "json" else yaml.safe_load(file_contents)

    for path in [Path("cache/pfp_character.png"), Path("cache/pfp_character_thumb.png")]:
        if path.exists():
            path.unlink()

    picture = generate_pfp_cache(character)

    # Finding the bot's name
    for k in ['name', 'bot', '<|bot|>', 'char_name']:
        if k in data and data[k] != '':
            name2 = data[k]
            break

    # Find the user name (if any)
    for k in ['your_name', 'user', '<|user|>']:
        if k in data and data[k] != '':
            name1 = data[k]
            break

    if 'context' in data:
        context = data['context'].strip()
    elif "char_persona" in data:
        context = build_pygmalion_style_context(data)
        greeting_field = 'char_greeting'

    greeting = data.get(greeting_field, greeting)
    return name1, name2, picture, greeting, context


def load_instruction_template(template):
    for filepath in [Path(f'instruction-templates/{template}.yaml'), Path('instruction-templates/Alpaca.yaml')]:
        if filepath.exists():
            break
    else:
        return ''

    file_contents = open(filepath, 'r', encoding='utf-8').read()
    data = yaml.safe_load(file_contents)
    if 'instruction_template' in data:
        return data['instruction_template']
    else:
        return jinja_template_from_old_format(data)


@functools.cache
def load_character_memoized(character, name1, name2):
    return load_character(character, name1, name2)


@functools.cache
def load_instruction_template_memoized(template):
    return load_instruction_template(template)


def upload_character(file, img, tavern=False):
    decoded_file = file if isinstance(file, str) else file.decode('utf-8')
    try:
        data = json.loads(decoded_file)
    except:
        data = yaml.safe_load(decoded_file)

    if 'char_name' in data:
        name = data['char_name']
        greeting = data['char_greeting']
        context = build_pygmalion_style_context(data)
        yaml_data = generate_character_yaml(name, greeting, context)
    else:
        name = data['name']
        yaml_data = generate_character_yaml(data['name'], data['greeting'], data['context'])

    outfile_name = name
    i = 1
    while Path(f'characters/{outfile_name}.yaml').exists():
        outfile_name = f'{name}_{i:03d}'
        i += 1

    with open(Path(f'characters/{outfile_name}.yaml'), 'w', encoding='utf-8') as f:
        f.write(yaml_data)

    if img is not None:
        img.save(Path(f'characters/{outfile_name}.png'))

    logger.info(f'New character saved to "characters/{outfile_name}.yaml".')
    return gr.update(value=outfile_name, choices=get_available_characters())


def build_pygmalion_style_context(data):
    context = ""
    if 'char_persona' in data and data['char_persona'] != '':
        context += f"{data['char_name']}'s Persona: {data['char_persona']}\n"

    if 'world_scenario' in data and data['world_scenario'] != '':
        context += f"Scenario: {data['world_scenario']}\n"

    if 'example_dialogue' in data and data['example_dialogue'] != '':
        context += f"{data['example_dialogue'].strip()}\n"

    context = f"{context.strip()}\n"
    return context


def upload_tavern_character(img, _json):
    _json = {'char_name': _json['name'], 'char_persona': _json['description'], 'char_greeting': _json['first_mes'], 'example_dialogue': _json['mes_example'], 'world_scenario': _json['scenario']}
    return upload_character(json.dumps(_json), img, tavern=True)


def check_tavern_character(img):
    if "chara" not in img.info:
        return "Not a TavernAI card", None, None, gr.update(interactive=False)

    decoded_string = base64.b64decode(img.info['chara']).replace(b'\\r\\n', b'\\n')
    _json = json.loads(decoded_string)
    if "data" in _json:
        _json = _json["data"]

    return _json['name'], _json['description'], _json, gr.update(interactive=True)


def upload_your_profile_picture(img):
    cache_folder = Path("cache")
    if not cache_folder.exists():
        cache_folder.mkdir()

    if img is None:
        if Path("cache/pfp_me.png").exists():
            Path("cache/pfp_me.png").unlink()
    else:
        img = make_thumbnail(img)
        img.save(Path('cache/pfp_me.png'))
        logger.info('Profile picture saved to "cache/pfp_me.png"')


def generate_character_yaml(name, greeting, context):
    data = {
        'name': name,
        'greeting': greeting,
        'context': context,
    }

    data = {k: v for k, v in data.items() if v}  # Strip falsy
    return yaml.dump(data, sort_keys=False, width=float("inf"))


def generate_instruction_template_yaml(instruction_template):
    data = {
        'instruction_template': instruction_template
    }

    return my_yaml_output(data)


def save_character(name, greeting, context, picture, filename):
    if filename == "":
        logger.error("The filename is empty, so the character will not be saved.")
        return

    data = generate_character_yaml(name, greeting, context)
    filepath = Path(f'characters/{filename}.yaml')
    save_file(filepath, data)
    path_to_img = Path(f'characters/{filename}.png')
    if picture is not None:
        picture.save(path_to_img)
        logger.info(f'Saved {path_to_img}.')


def delete_character(name, instruct=False):
    for extension in ["yml", "yaml", "json"]:
        delete_file(Path(f'characters/{name}.{extension}'))

    delete_file(Path(f'characters/{name}.png'))


def jinja_template_from_old_format(params, verbose=False):
    MASTER_TEMPLATE = """
{%- set found_item = false -%}
{%- for message in messages -%}
    {%- if message['role'] == 'system' -%}
        {%- set found_item = true -%}
    {%- endif -%}
{%- endfor -%}
{%- if not found_item -%}
    {{- '<|PRE-SYSTEM|>' + '<|SYSTEM-MESSAGE|>' + '<|POST-SYSTEM|>' -}}
{%- endif %}
{%- for message in messages %}
    {%- if message['role'] == 'system' -%}
        {{- '<|PRE-SYSTEM|>' + message['content'] + '<|POST-SYSTEM|>' -}}
    {%- else -%}
        {%- if message['role'] == 'user' -%}
            {{-'<|PRE-USER|>' + message['content'] + '<|POST-USER|>'-}}
        {%- else -%}
            {{-'<|PRE-ASSISTANT|>' + message['content'] + '<|POST-ASSISTANT|>' -}}
        {%- endif -%}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{-'<|PRE-ASSISTANT-GENERATE|>'-}}
{%- endif -%}
"""

    if 'context' in params and '<|system-message|>' in params['context']:
        pre_system = params['context'].split('<|system-message|>')[0]
        post_system = params['context'].split('<|system-message|>')[1]
    else:
        pre_system = ''
        post_system = ''

    pre_user = params['turn_template'].split('<|user-message|>')[0].replace('<|user|>', params['user'])
    post_user = params['turn_template'].split('<|user-message|>')[1].split('<|bot|>')[0]

    pre_assistant = '<|bot|>' + params['turn_template'].split('<|bot-message|>')[0].split('<|bot|>')[1]
    pre_assistant = pre_assistant.replace('<|bot|>', params['bot'])
    post_assistant = params['turn_template'].split('<|bot-message|>')[1]

    pre_system = pre_system.replace('\n', '\\n')
    post_system = post_system.replace('\n', '\\n')
    pre_user = pre_user.replace('\n', '\\n')
    post_user = post_user.replace('\n', '\\n')
    pre_assistant = pre_assistant.replace('\n', '\\n')
    post_assistant = post_assistant.replace('\n', '\\n')

    if verbose:
        print(
            '\n',
            repr(pre_system) + '\n',
            repr(post_system) + '\n',
            repr(pre_user) + '\n',
            repr(post_user) + '\n',
            repr(pre_assistant) + '\n',
            repr(post_assistant) + '\n',
        )

    result = MASTER_TEMPLATE
    if 'system_message' in params:
        result = result.replace('<|SYSTEM-MESSAGE|>', params['system_message'].replace('\n', '\\n'))
    else:
        result = result.replace('<|SYSTEM-MESSAGE|>', '')

    result = result.replace('<|PRE-SYSTEM|>', pre_system)
    result = result.replace('<|POST-SYSTEM|>', post_system)
    result = result.replace('<|PRE-USER|>', pre_user)
    result = result.replace('<|POST-USER|>', post_user)
    result = result.replace('<|PRE-ASSISTANT|>', pre_assistant)
    result = result.replace('<|PRE-ASSISTANT-GENERATE|>', pre_assistant.strip())
    result = result.replace('<|POST-ASSISTANT|>', post_assistant)

    result = result.strip()

    return result


def my_yaml_output(data):
    '''
    pyyaml is very inconsistent with multiline strings.
    for simple instruction template outputs, this is enough.
    '''
    result = ""
    for k in data:
        result += k + ": |-\n"
        for line in data[k].splitlines():
            result += "  " + line.rstrip(' ') + "\n"

    return result
