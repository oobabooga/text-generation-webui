import base64
import copy
import io
import json
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path

import modules.shared as shared
from modules.extensions import apply_extensions
from modules.html_generator import generate_chat_html
from modules.text_generation import encode
from modules.text_generation import generate_reply
from modules.text_generation import get_max_prompt_length
from PIL import Image

if shared.args.picture and (shared.args.cai_chat or shared.args.chat):
    import modules.bot_picture as bot_picture

history = {'internal': [], 'visible': []}
character = None

# This gets the new line characters right.
def clean_chat_message(text):
    text = text.replace('\n', '\n\n')
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text

def generate_chat_prompt(text, tokens, name1, name2, context, chat_prompt_size, impersonate=False):
    text = clean_chat_message(text)
    rows = [f"{context.strip()}\n"]
    i = len(history['internal'])-1
    count = 0

    if shared.soft_prompt:
        chat_prompt_size -= shared.soft_prompt_tensor.shape[1]
    max_length = min(get_max_prompt_length(tokens), chat_prompt_size)

    while i >= 0 and len(encode(''.join(rows), tokens)[0]) < max_length:
        rows.insert(1, f"{name2}: {history['internal'][i][1].strip()}\n")
        count += 1
        if not (history['internal'][i][0] == '<|BEGIN-VISIBLE-CHAT|>'):
            rows.insert(1, f"{name1}: {history['internal'][i][0].strip()}\n")
            count += 1
        i -= 1

    if not impersonate:
        rows.append(f"{name1}: {text}\n")
        rows.append(apply_extensions(f"{name2}:", "bot_prefix"))
        limit = 3
    else:
        rows.append(f"{name1}:")
        limit = 2

    while len(rows) > limit and len(encode(''.join(rows), tokens)[0]) >= max_length:
        rows.pop(1)
        rows.pop(1)

    question = ''.join(rows)
    return question

def extract_message_from_reply(question, reply, current, other, check, extensions=False):
    next_character_found = False
    substring_found = False

    previous_idx = [m.start() for m in re.finditer(f"(^|\n){re.escape(current)}:", question)]
    idx = [m.start() for m in re.finditer(f"(^|\n){re.escape(current)}:", reply)]
    idx = idx[len(previous_idx)-1]

    if extensions:
        reply = reply[idx + 1 + len(apply_extensions(f"{current}:", "bot_prefix")):]
    else:
        reply = reply[idx + 1 + len(f"{current}:"):]

    if check:
        reply = reply.split('\n')[0].strip()
    else:
        idx = reply.find(f"\n{other}:")
        if idx != -1:
            reply = reply[:idx]
            next_character_found = True
        reply = clean_chat_message(reply)

        # Detect if something like "\nYo" is generated just before
        # "\nYou:" is completed
        tmp = f"\n{other}:"
        for j in range(1, len(tmp)):
            if reply[-j:] == tmp[:j]:
                substring_found = True

    return reply, next_character_found, substring_found

def generate_chat_picture(picture, name1, name2):
    text = f'*{name1} sends {name2} a picture that contains the following: "{bot_picture.caption_image(picture)}"*'
    buffer = BytesIO()
    picture.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    visible_text = f'<img src="data:image/jpeg;base64,{img_str}">'
    return text, visible_text

def stop_everything_event():
    global stop_everything
    stop_everything = True

def chatbot_wrapper(text, tokens, do_sample, max_new_tokens, temperature, top_p, typical_p, repetition_penalty, top_k, min_length, no_repeat_ngram_size, num_beams, penalty_alpha, length_penalty, early_stopping, name1, name2, context, check, chat_prompt_size, picture=None):
    global stop_everything
    stop_everything = False

    if 'pygmalion' in shared.model_name.lower():
        name1 = "You"

    if shared.args.picture and picture is not None:
        text, visible_text = generate_chat_picture(picture, name1, name2)
    else:
        visible_text = text
        if shared.args.chat:
            visible_text = visible_text.replace('\n', '<br>')

    text = apply_extensions(text, "input")
    question = generate_chat_prompt(text, tokens, name1, name2, context, chat_prompt_size)
    eos_token = '\n' if check else None
    first = True
    for reply in generate_reply(question, tokens, do_sample, max_new_tokens, temperature, top_p, typical_p, repetition_penalty, top_k, min_length, no_repeat_ngram_size, num_beams, penalty_alpha, length_penalty, early_stopping, eos_token=eos_token, stopping_string=f"\n{name1}:"):
        reply, next_character_found, substring_found = extract_message_from_reply(question, reply, name2, name1, check, extensions=True)
        visible_reply = apply_extensions(reply, "output")
        if shared.args.chat:
            visible_reply = visible_reply.replace('\n', '<br>')

        # We need this global variable to handle the Stop event,
        # otherwise gradio gets confused
        if stop_everything:
            return history['visible']

        if first:
            first = False
            history['internal'].append(['', ''])
            history['visible'].append(['', ''])

        history['internal'][-1] = [text, reply]
        history['visible'][-1] = [visible_text, visible_reply]
        if not substring_found:
            yield history['visible']
        if next_character_found:
            break
    yield history['visible']

def impersonate_wrapper(text, tokens, do_sample, max_new_tokens, temperature, top_p, typical_p, repetition_penalty, top_k, min_length, no_repeat_ngram_size, num_beams, penalty_alpha, length_penalty, early_stopping, name1, name2, context, check, chat_prompt_size, picture=None):
    if 'pygmalion' in shared.model_name.lower():
        name1 = "You"

    question = generate_chat_prompt(text, tokens, name1, name2, context, chat_prompt_size, impersonate=True)
    eos_token = '\n' if check else None
    for reply in generate_reply(question, tokens, do_sample, max_new_tokens, temperature, top_p, typical_p, repetition_penalty, top_k, min_length, no_repeat_ngram_size, num_beams, penalty_alpha, length_penalty, early_stopping, eos_token=eos_token, stopping_string=f"\n{name2}:"):
        reply, next_character_found, substring_found = extract_message_from_reply(question, reply, name1, name2, check, extensions=False)
        if not substring_found:
            yield reply
        if next_character_found:
            break
    yield reply

def cai_chatbot_wrapper(text, tokens, do_sample, max_new_tokens, temperature, top_p, typical_p, repetition_penalty, top_k, min_length, no_repeat_ngram_size, num_beams, penalty_alpha, length_penalty, early_stopping, name1, name2, context, check, chat_prompt_size, picture=None):
    for _history in chatbot_wrapper(text, tokens, do_sample, max_new_tokens, temperature, top_p, typical_p, repetition_penalty, top_k, min_length, no_repeat_ngram_size, num_beams, penalty_alpha, length_penalty, early_stopping, name1, name2, context, check, chat_prompt_size, picture):
        yield generate_chat_html(_history, name1, name2, character)

def regenerate_wrapper(text, tokens, do_sample, max_new_tokens, temperature, top_p, typical_p, repetition_penalty, top_k, min_length, no_repeat_ngram_size, num_beams, penalty_alpha, length_penalty, early_stopping, name1, name2, context, check, chat_prompt_size, picture=None):
    if character is not None and len(history['visible']) == 1:
        if shared.args.cai_chat:
            yield generate_chat_html(history['visible'], name1, name2, character)
        else:
            yield history['visible']
    else:
        last_visible = history['visible'].pop()
        last_internal = history['internal'].pop()

        for _history in chatbot_wrapper(last_internal[0], tokens, do_sample, max_new_tokens, temperature, top_p, typical_p, repetition_penalty, top_k, min_length, no_repeat_ngram_size, num_beams, penalty_alpha, length_penalty, early_stopping, name1, name2, context, check, chat_prompt_size, picture):
            if shared.args.cai_chat:
                history['visible'][-1] = [last_visible[0], _history[-1][1]]
                yield generate_chat_html(history['visible'], name1, name2, character)
            else:
                history['visible'][-1] = (last_visible[0], _history[-1][1])
                yield history['visible']

def remove_last_message(name1, name2):
    if not history['internal'][-1][0] == '<|BEGIN-VISIBLE-CHAT|>':
        last = history['visible'].pop()
        history['internal'].pop()
    else:
        last = ['', '']
    if shared.args.cai_chat:
        return generate_chat_html(history['visible'], name1, name2, character), last[0]
    else:
        return history['visible'], last[0]

def send_last_reply_to_input():
    if len(history['internal']) > 0:
        return history['internal'][-1][1]
    else:
        return ''

def replace_last_reply(text, name1, name2):
    if len(history['visible']) > 0:
        if shared.args.cai_chat:
            history['visible'][-1][1] = text
        else:
            history['visible'][-1] = (history['visible'][-1][0], text)
        history['internal'][-1][1] = apply_extensions(text, "input")

    if shared.args.cai_chat:
        return generate_chat_html(history['visible'], name1, name2, character)
    else:
        return history['visible']

def clear_html():
    return generate_chat_html([], "", "", character)

def clear_chat_log(_character, name1, name2):
    global history
    if _character != 'None':
        for i in range(len(history['internal'])):
            if '<|BEGIN-VISIBLE-CHAT|>' in history['internal'][i][0]:
                history['visible'] = [['', history['internal'][i][1]]]
                history['internal'] = history['internal'][:i+1]
                break
    else:
        history['internal'] = []
        history['visible'] = []
    if shared.args.cai_chat:
        return generate_chat_html(history['visible'], name1, name2, character)
    else:
        return history['visible'] 

def redraw_html(name1, name2):
    global history
    return generate_chat_html(history['visible'], name1, name2, character)

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

    print(f"\033[1;32;1m\nDialogue tokenized to:\033[0;37;0m\n", end='')
    for row in _history:
        for column in row:
            print("\n")
            for line in column.strip().split('\n'):
                print("|  "+line+"\n")
            print("|\n")
        print("------------------------------")

    return _history

def save_history(timestamp=True):
    if timestamp:
        fname = f"{character or ''}{'_' if character else ''}{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    else:
        fname = f"{character or ''}{'_' if character else ''}persistent.json"
    if not Path('logs').exists():
        Path('logs').mkdir()
    with open(Path(f'logs/{fname}'), 'w') as f:
        f.write(json.dumps({'data': history['internal'], 'data_visible': history['visible']}, indent=2))
    return Path(f'logs/{fname}')

def load_history(file, name1, name2):
    global history
    file = file.decode('utf-8')
    try:
        j = json.loads(file)
        if 'data' in j:
            history['internal'] = j['data']
            if 'data_visible' in j:
                history['visible'] = j['data_visible']
            else:
                history['visible'] = copy.deepcopy(history['internal'])
        # Compatibility with Pygmalion AI's official web UI
        elif 'chat' in j:
            history['internal'] = [':'.join(x.split(':')[1:]).strip() for x in j['chat']]
            if len(j['chat']) > 0 and j['chat'][0].startswith(f'{name2}:'):
                history['internal'] = [['<|BEGIN-VISIBLE-CHAT|>', history['internal'][0]]] + [[history['internal'][i], history['internal'][i+1]] for i in range(1, len(history['internal'])-1, 2)]
                history['visible'] = copy.deepcopy(history['internal'])
                history['visible'][0][0] = ''
            else:
                history['internal'] = [[history['internal'][i], history['internal'][i+1]] for i in range(0, len(history['internal'])-1, 2)]
                history['visible'] = copy.deepcopy(history['internal'])
    except:
        history['internal'] = tokenize_dialogue(file, name1, name2)
        history['visible'] = copy.deepcopy(history['internal'])

def load_character(_character, name1, name2):
    global history, character
    context = ""
    history['internal'] = []
    history['visible'] = []
    if _character != 'None':
        character = _character
        data = json.loads(open(Path(f'characters/{_character}.json'), 'r').read())
        name2 = data['char_name']
        if 'char_persona' in data and data['char_persona'] != '':
            context += f"{data['char_name']}'s Persona: {data['char_persona']}\n"
        if 'world_scenario' in data and data['world_scenario'] != '':
            context += f"Scenario: {data['world_scenario']}\n"
        context = f"{context.strip()}\n<START>\n"
        if 'example_dialogue' in data and data['example_dialogue'] != '':
            history['internal'] = tokenize_dialogue(data['example_dialogue'], name1, name2)
        if 'char_greeting' in data and len(data['char_greeting'].strip()) > 0:
            history['internal'] += [['<|BEGIN-VISIBLE-CHAT|>', data['char_greeting']]]
            history['visible'] += [['', apply_extensions(data['char_greeting'], "output")]]
        else:
            history['internal'] += [['<|BEGIN-VISIBLE-CHAT|>', "Hello there!"]]
            history['visible'] += [['', "Hello there!"]]
    else:
        character = None
        context = shared.settings['context_pygmalion']
        name2 = shared.settings['name2_pygmalion']

    if Path(f'logs/{character}_persistent.json').exists():
        load_history(open(Path(f'logs/{character}_persistent.json'), 'rb').read(), name1, name2)

    if shared.args.cai_chat:
        return name2, context, generate_chat_html(history['visible'], name1, name2, character)
    else:
        return name2, context, history['visible']

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
    with open(Path(f'characters/{outfile_name}.json'), 'w') as f:
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
    _json['example_dialogue'] = _json['example_dialogue'].replace('{{user}}', name1).replace('{{char}}', _json['char_name'])
    return upload_character(json.dumps(_json), img, tavern=True)

def upload_your_profile_picture(img):
    img = Image.open(io.BytesIO(img))
    img.save(Path(f'img_me.png'))
    print(f'Profile picture saved to "img_me.png"')
