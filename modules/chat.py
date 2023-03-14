import base64
import copy
import io
import json
import yaml
import re
from datetime import datetime
from pathlib import Path

from PIL import Image

import modules.extensions as extensions_module
import modules.shared as shared
from modules.extensions import apply_extensions
from modules.html_generator import generate_chat_html
from modules.text_generation import encode, generate_reply, get_max_prompt_length


# This gets the new line characters right.
def clean_chat_message(text):
    text = text.replace('\n', '\n\n')
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text

def generate_chat_output(history, name1, name2, character):
    if shared.args.cai_chat:
        return generate_chat_html(history, name1, name2, character)
    else:
        return history

def generate_chat_prompt(user_input, max_new_tokens, name1, name2, context, chat_prompt_size, impersonate=False):
    user_input = clean_chat_message(user_input)
    rows = [f"{context.strip()}\n"]

    if shared.soft_prompt:
       chat_prompt_size -= shared.soft_prompt_tensor.shape[1]
    max_length = min(get_max_prompt_length(max_new_tokens), chat_prompt_size)

    i = len(shared.history['internal'])-1
    while i >= 0 and len(encode(''.join(rows), max_new_tokens)[0]) < max_length:
        rows.insert(1, f"{name2}: {shared.history['internal'][i][1].strip()}\n")
        if not (shared.history['internal'][i][0] == '<|BEGIN-VISIBLE-CHAT|>'):
            rows.insert(1, f"{name1}: {shared.history['internal'][i][0].strip()}\n")
        i -= 1

    if not impersonate:
        rows.append(f"{name1}: {user_input}\n")
        rows.append(apply_extensions(f"{name2}:", "bot_prefix"))
        limit = 3
    else:
        rows.append(f"{name1}:")
        limit = 2

    while len(rows) > limit and len(encode(''.join(rows), max_new_tokens)[0]) >= max_length:
        rows.pop(1)

    prompt = ''.join(rows)
    return prompt

def extract_message_from_reply(question, reply, name1, name2, check, impersonate=False):
    next_character_found = False

    asker = name1 if not impersonate else name2
    replier = name2 if not impersonate else name1

    previous_idx = [m.start() for m in re.finditer(f"(^|\n){re.escape(replier)}:", question)]
    idx = [m.start() for m in re.finditer(f"(^|\n){re.escape(replier)}:", reply)]
    idx = idx[max(len(previous_idx)-1, 0)]

    if not impersonate:
        reply = reply[idx + 1 + len(apply_extensions(f"{replier}:", "bot_prefix")):]
    else:
        reply = reply[idx + 1 + len(f"{replier}:"):]

    if check:
        lines = reply.split('\n')
        reply = lines[0].strip()
        if len(lines) > 1:
            next_character_found = True
    else:
        idx = reply.find(f"\n{asker}:")
        if idx != -1:
            reply = reply[:idx]
            next_character_found = True
        reply = clean_chat_message(reply)

        # If something like "\nYo" is generated just before "\nYou:"
        # is completed, trim it
        next_turn = f"\n{asker}:"
        for j in range(len(next_turn)-1, 0, -1):
            if reply[-j:] == next_turn[:j]:
                reply = reply[:-j]
                break

    return reply, next_character_found

def stop_everything_event():
    shared.stop_everything = True

def chatbot_wrapper(text, max_new_tokens, do_sample, temperature, top_p, typical_p, repetition_penalty, encoder_repetition_penalty, top_k, min_length, no_repeat_ngram_size, num_beams, penalty_alpha, length_penalty, early_stopping, name1, name2, context, check, chat_prompt_size, chat_generation_attempts=1, regenerate=False):
    shared.stop_everything = False
    just_started = True
    eos_token = '\n' if check else None
    name1_original = name1
    if 'pygmalion' in shared.model_name.lower():
        name1 = "You"

    # Check if any extension wants to hijack this function call
    visible_text = None
    custom_generate_chat_prompt = None
    for extension, _ in extensions_module.iterator():
        if hasattr(extension, 'input_hijack') and extension.input_hijack['state'] == True:
            extension.input_hijack['state'] = False
            text, visible_text = extension.input_hijack['value']
        if custom_generate_chat_prompt is None and hasattr(extension, 'custom_generate_chat_prompt'):
            custom_generate_chat_prompt = extension.custom_generate_chat_prompt

    if visible_text is None:
        visible_text = text
    if shared.args.chat:
        visible_text = visible_text.replace('\n', '<br>')
    text = apply_extensions(text, "input")

    if custom_generate_chat_prompt is None:
        prompt = generate_chat_prompt(text, max_new_tokens, name1, name2, context, chat_prompt_size)
    else:
        prompt = custom_generate_chat_prompt(text, max_new_tokens, name1, name2, context, chat_prompt_size)

    # Yield *Is typing...*
    if not regenerate:
        yield shared.history['visible']+[[visible_text, shared.processing_message]]

    # Generate
    reply = ''
    for i in range(chat_generation_attempts):
        for reply in generate_reply(f"{prompt}{' ' if len(reply) > 0 else ''}{reply}", max_new_tokens, do_sample, temperature, top_p, typical_p, repetition_penalty, encoder_repetition_penalty, top_k, min_length, no_repeat_ngram_size, num_beams, penalty_alpha, length_penalty, early_stopping, eos_token=eos_token, stopping_string=f"\n{name1}:"):

            # Extracting the reply
            reply, next_character_found = extract_message_from_reply(prompt, reply, name1, name2, check)
            visible_reply = re.sub("(<USER>|<user>|{{user}})", name1_original, reply)
            visible_reply = apply_extensions(visible_reply, "output")
            if shared.args.chat:
                visible_reply = visible_reply.replace('\n', '<br>')

            # We need this global variable to handle the Stop event,
            # otherwise gradio gets confused
            if shared.stop_everything:
                return shared.history['visible']
            if just_started:
                just_started = False
                shared.history['internal'].append(['', ''])
                shared.history['visible'].append(['', ''])

            shared.history['internal'][-1] = [text, reply]
            shared.history['visible'][-1] = [visible_text, visible_reply]
            if not shared.args.no_stream:
                yield shared.history['visible']
            if next_character_found:
                break

    yield shared.history['visible']

def impersonate_wrapper(text, max_new_tokens, do_sample, temperature, top_p, typical_p, repetition_penalty, encoder_repetition_penalty, top_k, min_length, no_repeat_ngram_size, num_beams, penalty_alpha, length_penalty, early_stopping, name1, name2, context, check, chat_prompt_size, chat_generation_attempts=1):
    eos_token = '\n' if check else None

    if 'pygmalion' in shared.model_name.lower():
        name1 = "You"

    prompt = generate_chat_prompt(text, max_new_tokens, name1, name2, context, chat_prompt_size, impersonate=True)

    reply = ''
    # Yield *Is typing...*
    yield shared.processing_message
    for i in range(chat_generation_attempts):
        for reply in generate_reply(prompt+reply, max_new_tokens, do_sample, temperature, top_p, typical_p, repetition_penalty, encoder_repetition_penalty, top_k, min_length, no_repeat_ngram_size, num_beams, penalty_alpha, length_penalty, early_stopping, eos_token=eos_token, stopping_string=f"\n{name2}:"):
            reply, next_character_found = extract_message_from_reply(prompt, reply, name1, name2, check, impersonate=True)
            yield reply
            if next_character_found:
                break
        yield reply

def cai_chatbot_wrapper(text, max_new_tokens, do_sample, temperature, top_p, typical_p, repetition_penalty, encoder_repetition_penalty, top_k, min_length, no_repeat_ngram_size, num_beams, penalty_alpha, length_penalty, early_stopping, name1, name2, context, check, chat_prompt_size, chat_generation_attempts=1):
    for _history in chatbot_wrapper(text, max_new_tokens, do_sample, temperature, top_p, typical_p, repetition_penalty, encoder_repetition_penalty, top_k, min_length, no_repeat_ngram_size, num_beams, penalty_alpha, length_penalty, early_stopping, name1, name2, context, check, chat_prompt_size, chat_generation_attempts):
        yield generate_chat_html(_history, name1, name2, shared.character)

def regenerate_wrapper(text, max_new_tokens, do_sample, temperature, top_p, typical_p, repetition_penalty, encoder_repetition_penalty, top_k, min_length, no_repeat_ngram_size, num_beams, penalty_alpha, length_penalty, early_stopping, name1, name2, context, check, chat_prompt_size, chat_generation_attempts=1):
    if (shared.character != 'None' and len(shared.history['visible']) == 1) or len(shared.history['internal']) == 0:
        yield generate_chat_output(shared.history['visible'], name1, name2, shared.character)
    else:
        last_visible = shared.history['visible'].pop()
        last_internal = shared.history['internal'].pop()
        # Yield '*Is typing...*'
        yield generate_chat_output(shared.history['visible']+[[last_visible[0], shared.processing_message]], name1, name2, shared.character)
        for _history in chatbot_wrapper(last_internal[0], max_new_tokens, do_sample, temperature, top_p, typical_p, repetition_penalty, encoder_repetition_penalty, top_k, min_length, no_repeat_ngram_size, num_beams, penalty_alpha, length_penalty, early_stopping, name1, name2, context, check, chat_prompt_size, chat_generation_attempts, regenerate=True):
            if shared.args.cai_chat:
                shared.history['visible'][-1] = [last_visible[0], _history[-1][1]]
            else:
                shared.history['visible'][-1] = (last_visible[0], _history[-1][1])
            yield generate_chat_output(shared.history['visible'], name1, name2, shared.character)

def remove_last_message(name1, name2):
    if len(shared.history['visible']) > 0 and not shared.history['internal'][-1][0] == '<|BEGIN-VISIBLE-CHAT|>':
        last = shared.history['visible'].pop()
        shared.history['internal'].pop()
    else:
        last = ['', '']

    if shared.args.cai_chat:
        return generate_chat_html(shared.history['visible'], name1, name2, shared.character), last[0]
    else:
        return shared.history['visible'], last[0]

def send_last_reply_to_input():
    if len(shared.history['internal']) > 0:
        return shared.history['internal'][-1][1]
    else:
        return ''

def replace_last_reply(text, name1, name2):
    if len(shared.history['visible']) > 0:
        if shared.args.cai_chat:
            shared.history['visible'][-1][1] = text
        else:
            shared.history['visible'][-1] = (shared.history['visible'][-1][0], text)
        shared.history['internal'][-1][1] = apply_extensions(text, "input")

    return generate_chat_output(shared.history['visible'], name1, name2, shared.character)

def clear_html():
    return generate_chat_html([], "", "", shared.character)

def clear_chat_log(name1, name2):
    if shared.character != 'None':
        found = False
        for i in range(len(shared.history['internal'])):
            if '<|BEGIN-VISIBLE-CHAT|>' in shared.history['internal'][i][0]:
                shared.history['visible'] = [['', apply_extensions(shared.history['internal'][i][1], "output")]]
                shared.history['internal'] = [shared.history['internal'][i]]
                found = True
                break
        if not found:
            shared.history['visible'] = []
            shared.history['internal'] = []
    else:
        shared.history['internal'] = []
        shared.history['visible'] = []

    return generate_chat_output(shared.history['visible'], name1, name2, shared.character)

def redraw_html(name1, name2):
    return generate_chat_html(shared.history['visible'], name1, name2, shared.character)

def tokenize_dialogue(dialogue, name1, name2):
    _history = []

    dialogue = re.sub('<START>', '', dialogue)
    dialogue = re.sub('<start>', '', dialogue)
    dialogue = re.sub('(\n|^)[Aa]non:', '\\1You:', dialogue)
    dialogue = re.sub('(\n|^)\[CHARACTER\]:', f'\\g<1>{name2}:', dialogue)
    idx = [m.start() for m in re.finditer(f"(^|\n)({re.escape(name1)}|{re.escape(name2)}):", dialogue)]
    if len(idx) == 0:
        return _history

    messages = []
    for i in range(len(idx)-1):
        messages.append(dialogue[idx[i]:idx[i+1]].strip())
    messages.append(dialogue[idx[-1]:].strip())

    entry = ['', '']
    for i in messages:
        if i.startswith(f'{name1}:'):
            entry[0] = i[len(f'{name1}:'):].strip()
        elif i.startswith(f'{name2}:'):
            entry[1] = i[len(f'{name2}:'):].strip()
            if not (len(entry[0]) == 0 and len(entry[1]) == 0):
                _history.append(entry)
            entry = ['', '']

    print("\033[1;32;1m\nDialogue tokenized to:\033[0;37;0m\n", end='')
    for row in _history:
        for column in row:
            print("\n")
            for line in column.strip().split('\n'):
                print("|  "+line+"\n")
            print("|\n")
        print("------------------------------")

    return _history

def save_history(timestamp=True):
    prefix = '' if shared.character == 'None' else f"{shared.character}_"
    if timestamp:
        fname = f"{prefix}{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    else:
        fname = f"{prefix}persistent.json"
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
        # Compatibility with Pygmalion AI's official web UI
        elif 'chat' in j:
            shared.history['internal'] = [':'.join(x.split(':')[1:]).strip() for x in j['chat']]
            if len(j['chat']) > 0 and j['chat'][0].startswith(f'{name2}:'):
                shared.history['internal'] = [['<|BEGIN-VISIBLE-CHAT|>', shared.history['internal'][0]]] + [[shared.history['internal'][i], shared.history['internal'][i+1]] for i in range(1, len(shared.history['internal'])-1, 2)]
                shared.history['visible'] = copy.deepcopy(shared.history['internal'])
                shared.history['visible'][0][0] = ''
            else:
                shared.history['internal'] = [[shared.history['internal'][i], shared.history['internal'][i+1]] for i in range(0, len(shared.history['internal'])-1, 2)]
                shared.history['visible'] = copy.deepcopy(shared.history['internal'])
    except:
        shared.history['internal'] = tokenize_dialogue(file, name1, name2)
        shared.history['visible'] = copy.deepcopy(shared.history['internal'])

def load_default_history(name1, name2):
    if Path('logs/persistent.json').exists():
        load_history(open(Path('logs/persistent.json'), 'rb').read(), name1, name2)
    else:
        shared.history['internal'] = []
        shared.history['visible'] = []

def load_character(_character, name1, name2):
    context = ""
    shared.history['internal'] = []
    shared.history['visible'] = []
    if _character != 'None':
        shared.character = _character
        
        extensions = ["yml", "yaml", "json"]
        for extension in extensions:
            filepath = Path(f'characters/{_character}.{extension}')
            if filepath.exists():
                break
        data = yaml.safe_load(open(filepath, 'r', encoding='utf-8').read())
        name2 = data['char_name']
        if 'char_persona' in data and data['char_persona'] != '':
            context += f"{data['char_name']}'s Persona: {data['char_persona']}\n"
        if 'world_scenario' in data and data['world_scenario'] != '':
            context += f"Scenario: {data['world_scenario']}\n"
        context = f"{context.strip()}\n<START>\n"
        if 'example_dialogue' in data and data['example_dialogue'] != '':
            data['example_dialogue'] = data['example_dialogue'].replace('{{user}}', name1).replace('{{char}}', name2)
            data['example_dialogue'] = data['example_dialogue'].replace('<USER>', name1).replace('<BOT>', name2)
            context += f"{data['example_dialogue'].strip()}\n"
        if 'char_greeting' in data and len(data['char_greeting'].strip()) > 0:
            shared.history['internal'] += [['<|BEGIN-VISIBLE-CHAT|>', data['char_greeting']]]
            shared.history['visible'] += [['', apply_extensions(data['char_greeting'], "output")]]
        else:
            shared.history['internal'] += [['<|BEGIN-VISIBLE-CHAT|>', "Hello there!"]]
            shared.history['visible'] += [['', "Hello there!"]]
    else:
        shared.character = None
        context = shared.settings['context_pygmalion']
        name2 = shared.settings['name2_pygmalion']

    if Path(f'logs/{shared.character}_persistent.json').exists():
        load_history(open(Path(f'logs/{shared.character}_persistent.json'), 'rb').read(), name1, name2)

    if shared.args.cai_chat:
        return name2, context, generate_chat_html(shared.history['visible'], name1, name2, shared.character)
    else:
        return name2, context, shared.history['visible']

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

def upload_your_profile_picture(img):
    img = Image.open(io.BytesIO(img))
    img.save(Path('img_me.png'))
    print('Profile picture saved to "img_me.png"')
