import ast
import base64
import copy
import io
import json
import logging
import re
from datetime import datetime
from pathlib import Path

import yaml
from PIL import Image

import modules.shared as shared
from modules.extensions import apply_extensions
from modules.html_generator import chat_html_wrapper, make_thumbnail
from modules.text_generation import (encode, generate_reply,
                                     get_max_prompt_length)


# Replace multiple string pairs in a string
def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)

    return text


def generate_chat_prompt(user_input, state, **kwargs):
    # Check if an extension is sending its modified history.
    # If not, use the regular history
    history = state['history'] if 'history' in state else shared.history['internal']

    # Define some variables
    impersonate = kwargs['impersonate'] if 'impersonate' in kwargs else False
    _continue = kwargs['_continue'] if '_continue' in kwargs else False
    also_return_rows = kwargs['also_return_rows'] if 'also_return_rows' in kwargs else False
    is_instruct = state['mode'] == 'instruct'
    rows = [state['context_instruct'] if is_instruct else f"{state['context'].strip()}\n"]
    min_rows = 3

    # Finding the maximum prompt size
    chat_prompt_size = state['chat_prompt_size']
    if shared.soft_prompt:
        chat_prompt_size -= shared.soft_prompt_tensor.shape[1]

    max_length = min(get_max_prompt_length(state), chat_prompt_size)

    # Building the turn templates
    if is_instruct:
        if 'turn_template' not in state or state['turn_template'] == '':
            template = '<|user|>\n<|user-message|>\n<|bot|>\n<|bot-message|>\n'
        else:
            template = state['turn_template'].replace(r'\n', '\n')
    else:
        template = '<|user|>: <|user-message|>\n<|bot|>: <|bot-message|>\n'

    replacements = {
        '<|user|>': state['name1_instruct' if is_instruct else 'name1'].strip(),
        '<|bot|>': state['name2_instruct' if is_instruct else 'name2'].strip(),
    }

    user_turn = replace_all(template.split('<|bot|>')[0], replacements)
    bot_turn = replace_all('<|bot|>' + template.split('<|bot|>')[1], replacements)
    user_turn_stripped = replace_all(user_turn.split('<|user-message|>')[0], replacements)
    bot_turn_stripped = replace_all(bot_turn.split('<|bot-message|>')[0], replacements)

    # Building the prompt
    i = len(history) - 1
    while i >= 0 and len(encode(''.join(rows))[0]) < max_length:
        if _continue and i == len(history) - 1:
            rows.insert(1, bot_turn_stripped + history[i][1].strip())
        else:
            rows.insert(1, bot_turn.replace('<|bot-message|>', history[i][1].strip()))

        string = history[i][0]
        if string not in ['', '<|BEGIN-VISIBLE-CHAT|>']:
            rows.insert(1, replace_all(user_turn, {'<|user-message|>': string.strip(), '<|round|>': str(i)}))

        i -= 1

    if impersonate:
        min_rows = 2
        rows.append(user_turn_stripped.rstrip(' '))
    elif not _continue:
        # Adding the user message
        if len(user_input) > 0:
            rows.append(replace_all(user_turn, {'<|user-message|>': user_input.strip(), '<|round|>': str(len(history))}))

        # Adding the Character prefix
        rows.append(apply_extensions("bot_prefix", bot_turn_stripped.rstrip(' ')))

    while len(rows) > min_rows and len(encode(''.join(rows))[0]) >= max_length:
        rows.pop(1)

    prompt = ''.join(rows)
    if also_return_rows:
        return prompt, rows
    else:
        return prompt


def get_stopping_strings(state):
    if state['mode'] == 'instruct':
        stopping_strings = [f"\n{state['name1_instruct']}", f"\n{state['name2_instruct']}"]
    else:
        stopping_strings = [f"\n{state['name1']}:", f"\n{state['name2']}:"]

    stopping_strings += ast.literal_eval(f"[{state['custom_stopping_strings']}]")
    return stopping_strings


def extract_message_from_reply(reply, state):
    next_character_found = False
    stopping_strings = get_stopping_strings(state)

    if state['stop_at_newline']:
        lines = reply.split('\n')
        reply = lines[0].strip()
        if len(lines) > 1:
            next_character_found = True
    else:
        for string in stopping_strings:
            idx = reply.find(string)
            if idx != -1:
                reply = reply[:idx]
                next_character_found = True

        # If something like "\nYo" is generated just before "\nYou:"
        # is completed, trim it
        if not next_character_found:
            for string in stopping_strings:
                for j in range(len(string) - 1, 0, -1):
                    if reply[-j:] == string[:j]:
                        reply = reply[:-j]
                        break
                else:
                    continue

                break

    return reply, next_character_found


def chatbot_wrapper(text, state, regenerate=False, _continue=False):
    if shared.model_name == 'None' or shared.model is None:
        logging.error("No model is loaded! Select one in the Model tab.")
        yield shared.history['visible']
        return

    # Defining some variables
    cumulative_reply = ''
    just_started = True
    visible_text = None
    eos_token = '\n' if state['stop_at_newline'] else None
    stopping_strings = get_stopping_strings(state)

    # Preparing the input
    if not any((regenerate, _continue)):
        text, visible_text = apply_extensions('input_hijack', text, visible_text)
        if visible_text is None:
            visible_text = text

        text = apply_extensions('input', text)
        # *Is typing...*
        yield shared.history['visible'] + [[visible_text, shared.processing_message]]
    else:
        text, visible_text = shared.history['internal'][-1][0], shared.history['visible'][-1][0]
        if regenerate:
            shared.history['visible'].pop()
            shared.history['internal'].pop()
            # *Is typing...*
            yield shared.history['visible'] + [[visible_text, shared.processing_message]]
        elif _continue:
            last_reply = [shared.history['internal'][-1][1], shared.history['visible'][-1][1]]
            yield shared.history['visible'][:-1] + [[visible_text, last_reply[1] + '...']]

    # Generating the prompt
    kwargs = {'_continue': _continue}
    prompt = apply_extensions('custom_generate_chat_prompt', text, state, **kwargs)
    if prompt is None:
        prompt = generate_chat_prompt(text, state, **kwargs)

    # Generate
    for i in range(state['chat_generation_attempts']):
        reply = None
        for reply in generate_reply(f"{prompt}{' ' if len(cumulative_reply) > 0 else ''}{cumulative_reply}", state, eos_token=eos_token, stopping_strings=stopping_strings):
            reply = cumulative_reply + reply

            # Extracting the reply
            reply, next_character_found = extract_message_from_reply(reply, state)
            visible_reply = re.sub("(<USER>|<user>|{{user}})", state['name1'], reply)
            visible_reply = apply_extensions("output", visible_reply)
            if _continue:
                sep = ' ' if last_reply[0][-1] not in [' ', '\n'] else ''
                reply = last_reply[0] + sep + reply
                sep = ' ' if last_reply[1][-1] not in [' ', '\n'] else ''
                visible_reply = last_reply[1] + sep + visible_reply

            # We need this global variable to handle the Stop event,
            # otherwise gradio gets confused
            if shared.stop_everything:
                return shared.history['visible']

            if just_started:
                just_started = False
                if not _continue:
                    shared.history['internal'].append(['', ''])
                    shared.history['visible'].append(['', ''])

            shared.history['internal'][-1] = [text, reply]
            shared.history['visible'][-1] = [visible_text, visible_reply]
            yield shared.history['visible']
            if next_character_found:
                break

        if reply is not None:
            cumulative_reply = reply

    yield shared.history['visible']


def impersonate_wrapper(text, state):
    if shared.model_name == 'None' or shared.model is None:
        logging.error("No model is loaded! Select one in the Model tab.")
        yield ''
        return

    # Defining some variables
    cumulative_reply = ''
    eos_token = '\n' if state['stop_at_newline'] else None
    prompt = generate_chat_prompt(text, state, impersonate=True)
    stopping_strings = get_stopping_strings(state)

    # Yield *Is typing...*
    yield shared.processing_message
    for i in range(state['chat_generation_attempts']):
        reply = None
        for reply in generate_reply(f"{prompt}{' ' if len(cumulative_reply) > 0 else ''}{cumulative_reply}", state, eos_token=eos_token, stopping_strings=stopping_strings):
            reply = cumulative_reply + reply
            reply, next_character_found = extract_message_from_reply(reply, state)
            yield reply
            if next_character_found:
                break

        if reply is not None:
            cumulative_reply = reply

    yield reply


def cai_chatbot_wrapper(text, state):
    for history in chatbot_wrapper(text, state):
        yield chat_html_wrapper(history, state['name1'], state['name2'], state['mode'], state['chat_style'])


def regenerate_wrapper(text, state):
    if (len(shared.history['visible']) == 1 and not shared.history['visible'][0][0]) or len(shared.history['internal']) == 0:
        yield chat_html_wrapper(shared.history['visible'], state['name1'], state['name2'], state['mode'], state['chat_style'])
    else:
        for history in chatbot_wrapper('', state, regenerate=True):
            yield chat_html_wrapper(history, state['name1'], state['name2'], state['mode'], state['chat_style'])


def continue_wrapper(text, state):
    if (len(shared.history['visible']) == 1 and not shared.history['visible'][0][0]) or len(shared.history['internal']) == 0:
        yield chat_html_wrapper(shared.history['visible'], state['name1'], state['name2'], state['mode'], state['chat_style'])
    else:
        for history in chatbot_wrapper('', state, _continue=True):
            yield chat_html_wrapper(history, state['name1'], state['name2'], state['mode'], state['chat_style'])


def remove_last_message(name1, name2, mode, style):
    if len(shared.history['visible']) > 0 and shared.history['internal'][-1][0] != '<|BEGIN-VISIBLE-CHAT|>':
        last = shared.history['visible'].pop()
        shared.history['internal'].pop()
    else:
        last = ['', '']

    return chat_html_wrapper(shared.history['visible'], name1, name2, mode, style), last[0]


def send_last_reply_to_input():
    if len(shared.history['internal']) > 0:
        return shared.history['internal'][-1][1]
    else:
        return ''


def replace_last_reply(text, name1, name2, mode, style):
    if len(shared.history['visible']) > 0:
        shared.history['visible'][-1][1] = text
        shared.history['internal'][-1][1] = apply_extensions("input", text)

    return chat_html_wrapper(shared.history['visible'], name1, name2, mode, style)


def send_dummy_message(text, name1, name2, mode, style):
    shared.history['visible'].append([text, ''])
    shared.history['internal'].append([apply_extensions("input", text), ''])
    return chat_html_wrapper(shared.history['visible'], name1, name2, mode, style)


def send_dummy_reply(text, name1, name2, mode, style):
    if len(shared.history['visible']) > 0 and not shared.history['visible'][-1][1] == '':
        shared.history['visible'].append(['', ''])
        shared.history['internal'].append(['', ''])

    shared.history['visible'][-1][1] = text
    shared.history['internal'][-1][1] = apply_extensions("input", text)
    return chat_html_wrapper(shared.history['visible'], name1, name2, mode, style)


def clear_html():
    return chat_html_wrapper([], "", "")


def clear_chat_log(name1, name2, greeting, mode, style):
    shared.history['visible'] = []
    shared.history['internal'] = []

    if mode != 'instruct':
        if greeting != '':
            shared.history['internal'] += [['<|BEGIN-VISIBLE-CHAT|>', greeting]]
            shared.history['visible'] += [['', apply_extensions("output", greeting)]]

        save_history(mode)

    return chat_html_wrapper(shared.history['visible'], name1, name2, mode, style)


def redraw_html(name1, name2, mode, style):
    return chat_html_wrapper(shared.history['visible'], name1, name2, mode, style)


def tokenize_dialogue(dialogue, name1, name2, mode, style):
    history = []
    messages = []
    dialogue = re.sub('<START>', '', dialogue)
    dialogue = re.sub('<start>', '', dialogue)
    dialogue = re.sub('(\n|^)[Aa]non:', '\\1You:', dialogue)
    dialogue = re.sub('(\n|^)\[CHARACTER\]:', f'\\g<1>{name2}:', dialogue)
    idx = [m.start() for m in re.finditer(f"(^|\n)({re.escape(name1)}|{re.escape(name2)}):", dialogue)]
    if len(idx) == 0:
        return history

    for i in range(len(idx) - 1):
        messages.append(dialogue[idx[i]:idx[i + 1]].strip())

    messages.append(dialogue[idx[-1]:].strip())
    entry = ['', '']
    for i in messages:
        if i.startswith(f'{name1}:'):
            entry[0] = i[len(f'{name1}:'):].strip()
        elif i.startswith(f'{name2}:'):
            entry[1] = i[len(f'{name2}:'):].strip()
            if not (len(entry[0]) == 0 and len(entry[1]) == 0):
                history.append(entry)

            entry = ['', '']

    print("\033[1;32;1m\nDialogue tokenized to:\033[0;37;0m\n", end='')
    for row in history:
        for column in row:
            print("\n")
            for line in column.strip().split('\n'):
                print("|  " + line + "\n")

            print("|\n")
        print("------------------------------")

    return history


def save_history(mode, timestamp=False):
    # Instruct mode histories should not be saved as if
    # Alpaca or Vicuna were characters
    if mode == 'instruct':
        if not timestamp:
            return

        fname = f"Instruct_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    else:
        if timestamp:
            fname = f"{shared.character}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        else:
            fname = f"{shared.character}_persistent.json"

    if not Path('logs').exists():
        Path('logs').mkdir()

    with open(Path(f'logs/{fname}'), 'w', encoding='utf-8') as f:
        f.write(json.dumps({'data': shared.history['internal'], 'data_visible': shared.history['visible']}, indent=2))

    return Path(f'logs/{fname}')


def load_history(file, name1, name2):
    file = file.decode('utf-8')
    try:
        j = json.loads(file)
        if 'data' in j:
            shared.history['internal'] = j['data']
            if 'data_visible' in j:
                shared.history['visible'] = j['data_visible']
            else:
                shared.history['visible'] = copy.deepcopy(shared.history['internal'])
    except:
        shared.history['internal'] = tokenize_dialogue(file, name1, name2)
        shared.history['visible'] = copy.deepcopy(shared.history['internal'])


def replace_character_names(text, name1, name2):
    text = text.replace('{{user}}', name1).replace('{{char}}', name2)
    return text.replace('<USER>', name1).replace('<BOT>', name2)


def build_pygmalion_style_context(data):
    context = ""
    if 'char_persona' in data and data['char_persona'] != '':
        context += f"{data['char_name']}'s Persona: {data['char_persona']}\n"

    if 'world_scenario' in data and data['world_scenario'] != '':
        context += f"Scenario: {data['world_scenario']}\n"

    context = f"{context.strip()}\n<START>\n"
    return context


def generate_pfp_cache(character):
    cache_folder = Path("cache")
    if not cache_folder.exists():
        cache_folder.mkdir()

    for path in [Path(f"characters/{character}.{extension}") for extension in ['png', 'jpg', 'jpeg']]:
        if path.exists():
            img = make_thumbnail(Image.open(path))
            img.save(Path('cache/pfp_character.png'), format='PNG')
            return img

    return None


def load_character(character, name1, name2, mode, style):
    shared.character = character
    context = greeting = turn_template = ""
    greeting_field = 'greeting'
    picture = None

    # Deleting the profile picture cache, if any
    if Path("cache/pfp_character.png").exists():
        Path("cache/pfp_character.png").unlink()

    if character != 'None':
        folder = 'characters' if not mode == 'instruct' else 'characters/instruction-following'
        picture = generate_pfp_cache(character)
        for extension in ["yml", "yaml", "json"]:
            filepath = Path(f'{folder}/{character}.{extension}')
            if filepath.exists():
                break

        file_contents = open(filepath, 'r', encoding='utf-8').read()
        data = json.loads(file_contents) if extension == "json" else yaml.safe_load(file_contents)

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

        for field in ['context', 'greeting', 'example_dialogue', 'char_persona', 'char_greeting', 'world_scenario']:
            if field in data:
                data[field] = replace_character_names(data[field], name1, name2)

        if 'context' in data:
            context = data['context']
            if mode != 'instruct':
                context = context.strip() + '\n\n'
        elif "char_persona" in data:
            context = build_pygmalion_style_context(data)
            greeting_field = 'char_greeting'

        if 'example_dialogue' in data:
            context += f"{data['example_dialogue'].strip()}\n"

        if greeting_field in data:
            greeting = data[greeting_field]

        if 'turn_template' in data:
            turn_template = data['turn_template']

    else:
        context = shared.settings['context']
        name2 = shared.settings['name2']
        greeting = shared.settings['greeting']
        turn_template = shared.settings['turn_template']

    if mode != 'instruct':
        shared.history['internal'] = []
        shared.history['visible'] = []
        if Path(f'logs/{shared.character}_persistent.json').exists():
            load_history(open(Path(f'logs/{shared.character}_persistent.json'), 'rb').read(), name1, name2)
        else:
            # Insert greeting if it exists
            if greeting != "":
                shared.history['internal'] += [['<|BEGIN-VISIBLE-CHAT|>', greeting]]
                shared.history['visible'] += [['', apply_extensions("output", greeting)]]

            # Create .json log files since they don't already exist
            save_history(mode)

    return name1, name2, picture, greeting, context, repr(turn_template)[1:-1], chat_html_wrapper(shared.history['visible'], name1, name2, mode, style)


def upload_character(json_file, img, tavern=False):
    json_file = json_file if type(json_file) == str else json_file.decode('utf-8')
    data = json.loads(json_file)
    outfile_name = data["char_name"]
    i = 1
    while Path(f'characters/{outfile_name}.json').exists():
        outfile_name = f'{data["char_name"]}_{i:03d}'
        i += 1

    if tavern:
        outfile_name = f'TavernAI-{outfile_name}'

    with open(Path(f'characters/{outfile_name}.json'), 'w', encoding='utf-8') as f:
        f.write(json_file)

    if img is not None:
        img = Image.open(io.BytesIO(img))
        img.save(Path(f'characters/{outfile_name}.png'))

    logging.info(f'New character saved to "characters/{outfile_name}.json".')
    return outfile_name


def upload_tavern_character(img, name1, name2):
    _img = Image.open(io.BytesIO(img))
    _img.getexif()
    decoded_string = base64.b64decode(_img.info['chara'])
    _json = json.loads(decoded_string)
    _json = {"char_name": _json['name'], "char_persona": _json['description'], "char_greeting": _json["first_mes"], "example_dialogue": _json['mes_example'], "world_scenario": _json['scenario']}
    return upload_character(json.dumps(_json), img, tavern=True)


def upload_your_profile_picture(img, name1, name2, mode, style):
    cache_folder = Path("cache")
    if not cache_folder.exists():
        cache_folder.mkdir()

    if img is None:
        if Path("cache/pfp_me.png").exists():
            Path("cache/pfp_me.png").unlink()
    else:
        img = make_thumbnail(img)
        img.save(Path('cache/pfp_me.png'))
        logging.info('Profile picture saved to "cache/pfp_me.png"')

    return chat_html_wrapper(shared.history['visible'], name1, name2, mode, style, reset_cache=True)
