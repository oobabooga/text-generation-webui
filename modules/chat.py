import ast
import base64
import copy
import io
import json
import re
from datetime import datetime
from pathlib import Path

import yaml
from PIL import Image

import modules.extensions as extensions_module
import modules.shared as shared
from modules.extensions import apply_extensions
from modules.html_generator import chat_html_wrapper, make_thumbnail
from modules.text_generation import (encode, generate_reply,
                                     get_max_prompt_length)


def generate_chat_prompt(user_input, state, **kwargs):
    impersonate = kwargs['impersonate'] if 'impersonate' in kwargs else False
    _continue = kwargs['_continue'] if '_continue' in kwargs else False
    also_return_rows = kwargs['also_return_rows'] if 'also_return_rows' in kwargs else False
    is_instruct = state['mode'] == 'instruct'
    rows = [f"{state['context'].strip()}\n"]
    min_rows = 3

    # Finding the maximum prompt size
    chat_prompt_size = state['chat_prompt_size']
    if shared.soft_prompt:
        chat_prompt_size -= shared.soft_prompt_tensor.shape[1]
    max_length = min(get_max_prompt_length(state), chat_prompt_size)

    if is_instruct:
        prefix1 = f"{state['name1']}\n"
        prefix2 = f"{state['name2']}\n"
    else:
        prefix1 = f"{state['name1']}: "
        prefix2 = f"{state['name2']}: "

    i = len(shared.history['internal']) - 1
    while i >= 0 and len(encode(''.join(rows))[0]) < max_length:
        if _continue and i == len(shared.history['internal']) - 1:
            rows.insert(1, f"{prefix2}{shared.history['internal'][i][1]}")
        else:
            rows.insert(1, f"{prefix2}{shared.history['internal'][i][1].strip()}{state['end_of_turn']}\n")

        string = shared.history['internal'][i][0]
        if string not in ['', '<|BEGIN-VISIBLE-CHAT|>']:
            this_prefix1 = prefix1.replace('<|round|>', f'{i}')  # for ChatGLM
            rows.insert(1, f"{this_prefix1}{string.strip()}{state['end_of_turn']}\n")

        i -= 1

    if impersonate:
        min_rows = 2
        rows.append(f"{prefix1.strip() if not is_instruct else prefix1}")
    elif not _continue:

        # Adding the user message
        if len(user_input) > 0:
            this_prefix1 = prefix1.replace('<|round|>', f'{len(shared.history["internal"])}')  # for ChatGLM
            rows.append(f"{this_prefix1}{user_input}{state['end_of_turn']}\n")

        # Adding the Character prefix
        rows.append(apply_extensions(f"{prefix2.strip() if not is_instruct else prefix2}", "bot_prefix"))

    while len(rows) > min_rows and len(encode(''.join(rows))[0]) >= max_length:
        rows.pop(1)
    prompt = ''.join(rows)

    if also_return_rows:
        return prompt, rows
    else:
        return prompt


def get_stopping_strings(state):
    if state['mode'] == 'instruct':
        stopping_strings = [f"\n{state['name1']}", f"\n{state['name2']}"]
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
        print("No model is loaded! Select one in the Model tab.")
        yield shared.history['visible']
        return

    # Defining some variables
    cumulative_reply = ''
    last_reply = [shared.history['internal'][-1][1], shared.history['visible'][-1][1]] if _continue else None
    just_started = True
    visible_text = custom_generate_chat_prompt = None
    eos_token = '\n' if state['stop_at_newline'] else None
    stopping_strings = get_stopping_strings(state)

    # Check if any extension wants to hijack this function call
    for extension, _ in extensions_module.iterator():
        if hasattr(extension, 'input_hijack') and extension.input_hijack['state']:
            extension.input_hijack['state'] = False
            text, visible_text = extension.input_hijack['value']
        if custom_generate_chat_prompt is None and hasattr(extension, 'custom_generate_chat_prompt'):
            custom_generate_chat_prompt = extension.custom_generate_chat_prompt

    if visible_text is None:
        visible_text = text
    if not _continue:
        text = apply_extensions(text, "input")

    # Generating the prompt
    kwargs = {'_continue': _continue}
    if custom_generate_chat_prompt is None:
        prompt = generate_chat_prompt(text, state, **kwargs)
    else:
        prompt = custom_generate_chat_prompt(text, state, **kwargs)

    # Yield *Is typing...*
    if not any((regenerate, _continue)):
        yield shared.history['visible'] + [[visible_text, shared.processing_message]]

    # Generate
    for i in range(state['chat_generation_attempts']):
        reply = None
        for reply in generate_reply(f"{prompt}{' ' if len(cumulative_reply) > 0 else ''}{cumulative_reply}", state, eos_token=eos_token, stopping_strings=stopping_strings):
            reply = cumulative_reply + reply

            # Extracting the reply
            reply, next_character_found = extract_message_from_reply(reply, state)
            visible_reply = re.sub("(<USER>|<user>|{{user}})", state['name1'], reply)
            visible_reply = apply_extensions(visible_reply, "output")

            # We need this global variable to handle the Stop event,
            # otherwise gradio gets confused
            if shared.stop_everything:
                return shared.history['visible']
            if just_started:
                just_started = False
                if not _continue:
                    shared.history['internal'].append(['', ''])
                    shared.history['visible'].append(['', ''])

            if _continue:
                sep = list(map(lambda x: ' ' if len(x) > 0 and x[-1] != ' ' else '', last_reply))
                shared.history['internal'][-1] = [text, f'{last_reply[0]}{sep[0]}{reply}']
                shared.history['visible'][-1] = [visible_text, f'{last_reply[1]}{sep[1]}{visible_reply}']
            else:
                shared.history['internal'][-1] = [text, reply]
                shared.history['visible'][-1] = [visible_text, visible_reply]
            if not shared.args.no_stream:
                yield shared.history['visible']
            if next_character_found:
                break

        if reply is not None:
            cumulative_reply = reply

    yield shared.history['visible']


def impersonate_wrapper(text, state):

    if shared.model_name == 'None' or shared.model is None:
        print("No model is loaded! Select one in the Model tab.")
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
        yield chat_html_wrapper(history, state['name1'], state['name2'], state['mode'])


def regenerate_wrapper(text, state):
    if (len(shared.history['visible']) == 1 and not shared.history['visible'][0][0]) or len(shared.history['internal']) == 0:
        yield chat_html_wrapper(shared.history['visible'], state['name1'], state['name2'], state['mode'])
    else:
        last_visible = shared.history['visible'].pop()
        last_internal = shared.history['internal'].pop()
        # Yield '*Is typing...*'
        yield chat_html_wrapper(shared.history['visible'] + [[last_visible[0], shared.processing_message]], state['name1'], state['name2'], state['mode'])
        for history in chatbot_wrapper(last_internal[0], state, regenerate=True):
            shared.history['visible'][-1] = [last_visible[0], history[-1][1]]
            yield chat_html_wrapper(shared.history['visible'], state['name1'], state['name2'], state['mode'])


def continue_wrapper(text, state):
    if (len(shared.history['visible']) == 1 and not shared.history['visible'][0][0]) or len(shared.history['internal']) == 0:
        yield chat_html_wrapper(shared.history['visible'], state['name1'], state['name2'], state['mode'])
    else:
        # Yield ' ...'
        yield chat_html_wrapper(shared.history['visible'][:-1] + [[shared.history['visible'][-1][0], shared.history['visible'][-1][1] + ' ...']], state['name1'], state['name2'], state['mode'])
        for history in chatbot_wrapper(shared.history['internal'][-1][0], state, _continue=True):
            yield chat_html_wrapper(shared.history['visible'], state['name1'], state['name2'], state['mode'])


def remove_last_message(name1, name2, mode):
    if len(shared.history['visible']) > 0 and shared.history['internal'][-1][0] != '<|BEGIN-VISIBLE-CHAT|>':
        last = shared.history['visible'].pop()
        shared.history['internal'].pop()
    else:
        last = ['', '']

    return chat_html_wrapper(shared.history['visible'], name1, name2, mode), last[0]


def send_last_reply_to_input():
    if len(shared.history['internal']) > 0:
        return shared.history['internal'][-1][1]
    else:
        return ''


def replace_last_reply(text, name1, name2, mode):
    if len(shared.history['visible']) > 0:
        shared.history['visible'][-1][1] = text
        shared.history['internal'][-1][1] = apply_extensions(text, "input")

    return chat_html_wrapper(shared.history['visible'], name1, name2, mode)


def send_dummy_message(text, name1, name2, mode):
    shared.history['visible'].append([text, ''])
    shared.history['internal'].append([apply_extensions(text, "input"), ''])
    return chat_html_wrapper(shared.history['visible'], name1, name2, mode)


def send_dummy_reply(text, name1, name2, mode):
    if len(shared.history['visible']) > 0 and not shared.history['visible'][-1][1] == '':
        shared.history['visible'].append(['', ''])
        shared.history['internal'].append(['', ''])
    shared.history['visible'][-1][1] = text
    shared.history['internal'][-1][1] = apply_extensions(text, "input")
    return chat_html_wrapper(shared.history['visible'], name1, name2, mode)


def clear_html():
    return chat_html_wrapper([], "", "")


def clear_chat_log(name1, name2, greeting, mode):
    shared.history['visible'] = []
    shared.history['internal'] = []

    if greeting != '':
        shared.history['internal'] += [['<|BEGIN-VISIBLE-CHAT|>', greeting]]
        shared.history['visible'] += [['', apply_extensions(greeting, "output")]]

    # Save cleared logs
    save_history(mode)

    return chat_html_wrapper(shared.history['visible'], name1, name2, mode)


def redraw_html(name1, name2, mode):
    return chat_html_wrapper(shared.history['visible'], name1, name2, mode)


def tokenize_dialogue(dialogue, name1, name2, mode):
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


def load_character(character, name1, name2, mode):
    shared.character = character
    context = greeting = end_of_turn = ""
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
        name2 = data['name'] if 'name' in data else data['char_name']
        if 'your_name' in data and data['your_name'] != '':
            name1 = data['your_name']
        else:
            name1 = shared.settings['name1']

        for field in ['context', 'greeting', 'example_dialogue', 'char_persona', 'char_greeting', 'world_scenario']:
            if field in data:
                data[field] = replace_character_names(data[field], name1, name2)

        if 'context' in data:
            context = f"{data['context'].strip()}\n\n"
        elif "char_persona" in data:
            context = build_pygmalion_style_context(data)
            greeting_field = 'char_greeting'

        if 'example_dialogue' in data:
            context += f"{data['example_dialogue'].strip()}\n"

        if greeting_field in data:
            greeting = data[greeting_field]

        if 'end_of_turn' in data:
            end_of_turn = data['end_of_turn']

    else:
        context = shared.settings['context']
        name2 = shared.settings['name2']
        greeting = shared.settings['greeting']
        end_of_turn = shared.settings['end_of_turn']

    if mode != 'instruct':
        shared.history['internal'] = []
        shared.history['visible'] = []
        if Path(f'logs/{shared.character}_persistent.json').exists():
            load_history(open(Path(f'logs/{shared.character}_persistent.json'), 'rb').read(), name1, name2)
        else:
            # Insert greeting if it exists
            if greeting != "":
                shared.history['internal'] += [['<|BEGIN-VISIBLE-CHAT|>', greeting]]
                shared.history['visible'] += [['', apply_extensions(greeting, "output")]]

            # Create .json log files since they don't already exist
            save_history(mode)

    return name1, name2, picture, greeting, context, end_of_turn, chat_html_wrapper(shared.history['visible'], name1, name2, mode)


def load_default_history(name1, name2):
    load_character("None", name1, name2, "chat")


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
    print(f'New character saved to "characters/{outfile_name}.json".')
    return outfile_name


def upload_tavern_character(img, name1, name2):
    _img = Image.open(io.BytesIO(img))
    _img.getexif()
    decoded_string = base64.b64decode(_img.info['chara'])
    _json = json.loads(decoded_string)
    _json = {"char_name": _json['name'], "char_persona": _json['description'], "char_greeting": _json["first_mes"], "example_dialogue": _json['mes_example'], "world_scenario": _json['scenario']}
    return upload_character(json.dumps(_json), img, tavern=True)


def upload_your_profile_picture(img, name1, name2, mode):
    cache_folder = Path("cache")
    if not cache_folder.exists():
        cache_folder.mkdir()

    if img is None:
        if Path("cache/pfp_me.png").exists():
            Path("cache/pfp_me.png").unlink()
    else:
        img = make_thumbnail(img)
        img.save(Path('cache/pfp_me.png'))
        print('Profile picture saved to "cache/pfp_me.png"')

    return chat_html_wrapper(shared.history['visible'], name1, name2, mode, reset_cache=True)
