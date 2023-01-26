import re
import gc
import time
import glob
import torch
import argparse
import json
from sys import exit
from pathlib import Path
import gradio as gr
import warnings
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from modules.html_generator import *
from modules.ui import *
from modules.stopping_criteria import _SentinelTokenStoppingCriteria

transformers.logging.set_verbosity_error()

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='Name of the model to load by default.')
parser.add_argument('--notebook', action='store_true', help='Launch the web UI in notebook mode, where the output is written to the same text box as the input.')
parser.add_argument('--chat', action='store_true', help='Launch the web UI in chat mode.')
parser.add_argument('--cai-chat', action='store_true', help='Launch the web UI in chat mode with a style similar to Character.AI\'s. If the file img_bot.png or img_bot.jpg exists in the same folder as server.py, this image will be used as the bot\'s profile picture. Similarly, img_me.png or img_me.jpg will be used as your profile picture.')
parser.add_argument('--cpu', action='store_true', help='Use the CPU to generate text.')
parser.add_argument('--load-in-8bit', action='store_true', help='Load the model with 8-bit precision.')
parser.add_argument('--auto-devices', action='store_true', help='Automatically split the model across the available GPU(s) and CPU.')
parser.add_argument('--disk', action='store_true', help='If the model is too large for your GPU(s) and CPU combined, send the remaining layers to the disk.')
parser.add_argument('--disk-cache-dir', type=str, help='Directory to save the disk cache to. Defaults to "cache/".')
parser.add_argument('--gpu-memory', type=int, help='Maximum GPU memory in GiB to allocate. This is useful if you get out of memory errors while trying to generate text. Must be an integer number.')
parser.add_argument('--cpu-memory', type=int, help='Maximum CPU memory in GiB to allocate for offloaded weights. Must be an integer number. Defaults to 99.')
parser.add_argument('--no-stream', action='store_true', help='Don\'t stream the text output in real time. This improves the text generation performance.')
parser.add_argument('--settings', type=str, help='Load the default interface settings from this json file. See settings-template.json for an example.')
parser.add_argument('--listen', action='store_true', help='Make the web UI reachable from your local network.')
parser.add_argument('--share', action='store_true', help='Create a public URL. This is useful for running the web UI on Google Colab or similar.')
args = parser.parse_args()

if (args.chat or args.cai_chat) and not args.no_stream:
    print("Warning: chat mode currently becomes somewhat slower with text streaming on.\nConsider starting the web UI with the --no-stream option.\n")

settings = {
    'max_new_tokens': 200,
    'max_new_tokens_min': 1,
    'max_new_tokens_max': 2000,
    'preset': 'NovelAI-Sphinx Moth',
    'name1': 'Person 1',
    'name2': 'Person 2',
    'context': 'This is a conversation between two people.',
    'prompt': 'Common sense questions and answers\n\nQuestion: \nFactual answer:',
    'prompt_gpt4chan': '-----\n--- 865467536\nInput text\n--- 865467537\n',
    'stop_at_newline': True,
    'history_size': 0,
    'history_size_min': 0,
    'history_size_max': 64,
    'preset_pygmalion': 'Pygmalion',
    'name1_pygmalion': 'You',
    'name2_pygmalion': 'Kawaii',
    'context_pygmalion': "Kawaii's persona: Kawaii is a cheerful person who loves to make others smile. She is an optimist who loves to spread happiness and positivity wherever she goes.\n<START>",
    'stop_at_newline_pygmalion': False,
}

if args.settings is not None and Path(args.settings).exists():
    with open(Path(args.settings), 'r') as f:
        new_settings = json.load(f)
    for item in new_settings:
        if item in settings:
            settings[item] = new_settings[item]

def load_model(model_name):
    print(f"Loading {model_name}...")
    t0 = time.time()

    # Default settings
    if not (args.cpu or args.load_in_8bit or args.auto_devices or args.disk or args.gpu_memory is not None):
        if Path(f"torch-dumps/{model_name}.pt").exists():
            print("Loading in .pt format...")
            model = torch.load(Path(f"torch-dumps/{model_name}.pt"))
        elif model_name.lower().startswith(('gpt-neo', 'opt-', 'galactica')) and any(size in model_name.lower() for size in ('13b', '20b', '30b')):
            model = AutoModelForCausalLM.from_pretrained(Path(f"models/{model_name}"), device_map='auto', load_in_8bit=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(Path(f"models/{model_name}"), low_cpu_mem_usage=True, torch_dtype=torch.float16).cuda()
    # Custom
    else:
        settings = ["low_cpu_mem_usage=True"]
        command = "AutoModelForCausalLM.from_pretrained"

        if args.cpu:
            settings.append("torch_dtype=torch.float32")
        else:
            settings.append("device_map='auto'")
            if args.gpu_memory is not None:
                if args.cpu_memory is not None:
                    settings.append(f"max_memory={{0: '{args.gpu_memory}GiB', 'cpu': '{args.cpu_memory}GiB'}}")
                else:
                    settings.append(f"max_memory={{0: '{args.gpu_memory}GiB', 'cpu': '99GiB'}}")
            if args.disk:
                if args.disk_cache_dir is not None:
                    settings.append(f"offload_folder='{args.disk_cache_dir}'")
                else:
                    settings.append("offload_folder='cache'")
            if args.load_in_8bit:
                settings.append("load_in_8bit=True")
            else:
                settings.append("torch_dtype=torch.float16")

        settings = ', '.join(set(settings))
        command = f"{command}(Path(f'models/{model_name}'), {settings})"
        model = eval(command)

    # Loading the tokenizer
    if model_name.lower().startswith(('gpt4chan', 'gpt-4chan', '4chan')) and Path(f"models/gpt-j-6B/").exists():
        tokenizer = AutoTokenizer.from_pretrained(Path("models/gpt-j-6B/"))
    else:
        tokenizer = AutoTokenizer.from_pretrained(Path(f"models/{model_name}/"))
    tokenizer.truncation_side = 'left'

    print(f"Loaded the model in {(time.time()-t0):.2f} seconds.")
    return model, tokenizer

# Removes empty replies from gpt4chan outputs
def fix_gpt4chan(s):
    for i in range(10):
        s = re.sub("--- [0-9]*\n>>[0-9]*\n---", "---", s)
        s = re.sub("--- [0-9]*\n *\n---", "---", s)
        s = re.sub("--- [0-9]*\n\n\n---", "---", s)
    return s

# Fix the LaTeX equations in galactica
def fix_galactica(s):
    s = s.replace(r'\[', r'$')
    s = s.replace(r'\]', r'$')
    s = s.replace(r'\(', r'$')
    s = s.replace(r'\)', r'$')
    s = s.replace(r'$$', r'$')
    return s

def encode(prompt, tokens_to_generate=0, add_special_tokens=True):
    if args.cpu:
        input_ids = tokenizer.encode(str(prompt), return_tensors='pt', truncation=True, max_length=2048-tokens_to_generate, add_special_tokens=add_special_tokens)
    else:
        torch.cuda.empty_cache()
        input_ids = tokenizer.encode(str(prompt), return_tensors='pt', truncation=True, max_length=2048-tokens_to_generate, add_special_tokens=add_special_tokens).cuda()
    return input_ids

def decode(output_ids):
    reply = tokenizer.decode(output_ids, skip_special_tokens=True)
    reply = reply.replace(r'<|endoftext|>', '')
    return reply

def formatted_outputs(reply, model_name):
    if not (args.chat or args.cai_chat):
        if model_name.lower().startswith('galactica'):
            reply = fix_galactica(reply)
            return reply, reply, generate_basic_html(reply)
        elif model_name.lower().startswith(('gpt4chan', 'gpt-4chan', '4chan')):
            reply = fix_gpt4chan(reply)
            return reply, 'Only applicable for GALACTICA models.', generate_4chan_html(reply)
        else:
            return reply, 'Only applicable for GALACTICA models.', generate_basic_html(reply)
    else:
        return reply

def generate_reply(question, tokens, inference_settings, selected_model, eos_token=None, stopping_string=None):
    global model, tokenizer, model_name, loaded_preset, preset

    if selected_model != model_name:
        model_name = selected_model
        model = tokenizer = None
        if not args.cpu:
            gc.collect()
            torch.cuda.empty_cache()
        model, tokenizer = load_model(model_name)
    if inference_settings != loaded_preset:
        with open(Path(f'presets/{inference_settings}.txt'), 'r') as infile:
            preset = infile.read()
        loaded_preset = inference_settings

    cuda = "" if args.cpu else ".cuda()"
    n = tokenizer.eos_token_id if eos_token is None else tokenizer.encode(eos_token, return_tensors='pt')[0][-1]
    input_ids = encode(question, tokens)
    # The stopping_criteria code below was copied from
    # https://github.com/PygmalionAI/gradio-ui/blob/master/src/model.py
    if stopping_string is not None:
        t = encode(stopping_string, 0, add_special_tokens=False)
        stopping_criteria_list = transformers.StoppingCriteriaList([
            _SentinelTokenStoppingCriteria(
                sentinel_token_ids=t,
                starting_idx=len(input_ids[0])
            )
        ])
    else:
        stopping_criteria_list = None

    # Generate the entire reply at once
    if args.no_stream:
        t0 = time.time()
        output = eval(f"model.generate(input_ids, eos_token_id={n}, stopping_criteria=stopping_criteria_list, {preset}){cuda}")
        reply = decode(output[0])
        t1 = time.time()
        print(f"Output generated in {(t1-t0):.2f} seconds ({(len(output[0])-len(input_ids[0]))/(t1-t0):.2f} it/s)")
        yield formatted_outputs(reply, model_name)

    # Generate the reply 1 token at a time
    else:
        yield formatted_outputs(question, model_name)
        preset = preset.replace('max_new_tokens=tokens', 'max_new_tokens=8')
        for i in tqdm(range(tokens//8+1)):
            output = eval(f"model.generate(input_ids, eos_token_id={n}, stopping_criteria=stopping_criteria_list, {preset}){cuda}")
            reply = decode(output[0])
            yield formatted_outputs(reply, model_name)
            input_ids = output
            if output[0][-1] == n:
                break

def get_available_models():
    return sorted(set([item.replace('.pt', '') for item in map(lambda x : str(x.name), list(Path('models/').glob('*'))+list(Path('torch-dumps/').glob('*'))) if not item.endswith('.txt')]), key=str.lower)

def get_available_presets():
    return sorted(set(map(lambda x : '.'.join(str(x.name).split('.')[:-1]), Path('presets').glob('*.txt'))), key=str.lower)

def get_available_characters():
    return ["None"] + sorted(set(map(lambda x : '.'.join(str(x.name).split('.')[:-1]), Path('characters').glob('*.json'))), key=str.lower)

available_models = get_available_models()
available_presets = get_available_presets()
available_characters = get_available_characters()

# Choosing the default model
if args.model is not None:
    model_name = args.model
else:
    if len(available_models) == 0:
        print("No models are available! Please download at least one.")
        exit(0)
    elif len(available_models) == 1:
        i = 0
    else:
        print("The following models are available:\n")
        for i,model in enumerate(available_models):
            print(f"{i+1}. {model}")
        print(f"\nWhich one do you want to load? 1-{len(available_models)}\n")
        i = int(input())-1
        print()
    model_name = available_models[i]
model, tokenizer = load_model(model_name)
loaded_preset = None

# UI settings
default_text = settings['prompt_gpt4chan'] if model_name.lower().startswith(('gpt4chan', 'gpt-4chan', '4chan')) else settings['prompt']
description = f"\n\n# Text generation lab\nGenerate text using Large Language Models.\n"
css = ".my-4 {margin-top: 0} .py-6 {padding-top: 2.5rem} #refresh-button {flex: none; margin: 0; padding: 0; min-width: 50px; border: none; box-shadow: none; border-radius: 0} #download-label, #upload-label {min-height: 0}"

if args.chat or args.cai_chat:
    history = []
    character = None

    # This gets the new line characters right.
    def clean_chat_message(text):
        text = text.replace('\n', '\n\n')
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        return text

    def generate_chat_prompt(text, tokens, name1, name2, context, history_size):
        text = clean_chat_message(text)

        rows = [f"{context.strip()}\n"]
        i = len(history)-1
        count = 0
        while i >= 0 and len(encode(''.join(rows), tokens)[0]) < 2048-tokens:
            rows.insert(1, f"{name2}: {history[i][1].strip()}\n")
            count += 1
            if not (history[i][0] == '<|BEGIN-VISIBLE-CHAT|>'):
                rows.insert(1, f"{name1}: {history[i][0].strip()}\n")
                count += 1
            i -= 1
            if history_size != 0 and count >= history_size:
                break
        rows.append(f"{name1}: {text}\n")
        rows.append(f"{name2}:")

        while len(rows) > 3 and len(encode(''.join(rows), tokens)[0]) >= 2048-tokens:
            rows.pop(1)
            rows.pop(1)

        question = ''.join(rows)
        return question

    def remove_example_dialogue_from_history(history):
        _history = copy.deepcopy(history)
        for i in range(len(_history)):
            if '<|BEGIN-VISIBLE-CHAT|>' in _history[i][0]:
                _history[i][0] = _history[i][0].replace('<|BEGIN-VISIBLE-CHAT|>', '')
                _history = _history[i:]
                break
        return _history

    def chatbot_wrapper(text, tokens, inference_settings, selected_model, name1, name2, context, check, history_size):
        question = generate_chat_prompt(text, tokens, name1, name2, context, history_size)
        history.append(['', ''])
        eos_token = '\n' if check else None
        for reply in generate_reply(question, tokens, inference_settings, selected_model, eos_token=eos_token, stopping_string=f"\n{name1}:"):
            next_character_found = False

            previous_idx = [m.start() for m in re.finditer(f"(^|\n){name2}:", question)]
            idx = [m.start() for m in re.finditer(f"(^|\n){name2}:", reply)]
            idx = idx[len(previous_idx)-1]

            reply = reply[idx + len(f"\n{name2}:"):]

            if check:
                reply = reply.split('\n')[0].strip()
            else:
                idx = reply.find(f"\n{name1}:")
                if idx != -1:
                    reply = reply[:idx]
                    next_character_found = True
                reply = clean_chat_message(reply)

            history[-1] = [text, reply]
            if next_character_found:
                break

            # Prevent the chat log from flashing if something like "\nYo" is generated just
            # before "\nYou:" is completed
            tmp = f"\n{name1}:"
            next_character_substring_found = False
            for j in range(1, len(tmp)):
                if reply[-j:] == tmp[:j]:
                    next_character_substring_found = True

            if not next_character_substring_found:
                yield remove_example_dialogue_from_history(history)

        yield remove_example_dialogue_from_history(history)

    def cai_chatbot_wrapper(text, tokens, inference_settings, selected_model, name1, name2, context, check, history_size):
        for history in chatbot_wrapper(text, tokens, inference_settings, selected_model, name1, name2, context, check, history_size):
            yield generate_chat_html(history, name1, name2, character)

    def regenerate_wrapper(text, tokens, inference_settings, selected_model, name1, name2, context, check, history_size):
        last = history.pop()
        text = last[0]
        if args.cai_chat:
            for i in cai_chatbot_wrapper(text, tokens, inference_settings, selected_model, name1, name2, context, check, history_size):
                yield i
        else:
            for i in chatbot_wrapper(text, tokens, inference_settings, selected_model, name1, name2, context, check, history_size):
                yield i

    def remove_last_message(name1, name2):
        last = history.pop()
        _history = remove_example_dialogue_from_history(history)
        if args.cai_chat:
            return generate_chat_html(_history, name1, name2, character), last[0]
        else:
            return _history, last[0]

    def clear_html():
        return generate_chat_html([], "", "", character)

    def clear_chat_log(_character, name1, name2):
        global history
        if _character != 'None':
            load_character(_character, name1, name2)
        else:
            history = []
        _history = remove_example_dialogue_from_history(history)
        if args.cai_chat:
            return generate_chat_html(_history, name1, name2, character)
        else:
            return _history

    def redraw_html(name1, name2):
        global history
        _history = remove_example_dialogue_from_history(history)
        return generate_chat_html(_history, name1, name2, character)

    def tokenize_dialogue(dialogue, name1, name2):
        dialogue = re.sub('<START>', '', dialogue)
        dialogue = re.sub('(\n|^)[Aa]non:', '\\1You:', dialogue)

        idx = [m.start() for m in re.finditer(f"(^|\n)({name1}|{name2}):", dialogue)]
        messages = []
        for i in range(len(idx)-1):
            messages.append(dialogue[idx[i]:idx[i+1]].strip())
        messages.append(dialogue[idx[-1]:].strip())

        history = []
        entry = ['', '']
        for i in messages:
            if i.startswith(f'{name1}:'):
                entry[0] = i[len(f'{name1}:'):].strip()
            elif i.startswith(f'{name2}:'):
                entry[1] = i[len(f'{name2}:'):].strip()
                if not (len(entry[0]) == 0 and len(entry[1]) == 0):
                    history.append(entry)
                entry = ['', '']

        return history

    def save_history():
        if not Path('logs').exists():
            Path('logs').mkdir()
        with open(Path('logs/conversation.json'), 'w') as f:
            f.write(json.dumps({'data': history}, indent=2))
        return Path('logs/conversation.json')

    def upload_history(file, name1, name2):
        global history
        file = file.decode('utf-8')
        try:
            j = json.loads(file)
            if 'data' in j:
                history = j['data']
            # Compatibility with Pygmalion AI's official web UI
            elif 'chat' in j:
                history = [':'.join(x.split(':')[1:]).strip() for x in j['chat']]
                if len(j['chat']) > 0 and j['chat'][0].startswith(f'{name2}:'):
                    history = [['<|BEGIN-VISIBLE-CHAT|>', history[0]]] + [[history[i], history[i+1]] for i in range(1, len(history)-1, 2)]
                else:
                    history = [[history[i], history[i+1]] for i in range(0, len(history)-1, 2)]
        except:
            history = tokenize_dialogue(file, name1, name2)

    def load_character(_character, name1, name2):
        global history, character
        context = ""
        history = []
        if _character != 'None':
            character = _character
            with open(Path(f'characters/{_character}.json'), 'r') as f:
                data = json.loads(f.read())
            name2 = data['char_name']
            if 'char_persona' in data and data['char_persona'] != '':
                context += f"{data['char_name']}'s Persona: {data['char_persona']}\n"
            if 'world_scenario' in data and data['world_scenario'] != '':
                context += f"Scenario: {data['world_scenario']}\n"
            context = f"{context.strip()}\n<START>\n"
            if 'example_dialogue' in data and data['example_dialogue'] != '':
                history = tokenize_dialogue(data['example_dialogue'], name1, name2)
            if 'char_greeting' in data and len(data['char_greeting'].strip()) > 0:
                history += [['<|BEGIN-VISIBLE-CHAT|>', data['char_greeting']]]
            else:
                history += [['<|BEGIN-VISIBLE-CHAT|>', "Hello there!"]]
        else:
            character = None
            context = settings['context_pygmalion']
            name2 = settings['name2_pygmalion']

        _history = remove_example_dialogue_from_history(history)
        if args.cai_chat:
            return name2, context, generate_chat_html(_history, name1, name2, character)
        else:
            return name2, context, _history

    def upload_character(file, name1, name2):
        global history
        file = file.decode('utf-8')
        data = json.loads(file)
        outfile_name = data["char_name"]
        i = 1
        while Path(f'characters/{outfile_name}.json').exists():
            outfile_name = f'{data["char_name"]}_{i:03d}'
            i += 1
        with open(Path(f'characters/{outfile_name}.json'), 'w') as f:
            f.write(file)
        print(f'New character saved to "characters/{outfile_name}.json".')
        return outfile_name

    suffix = '_pygmalion' if 'pygmalion' in model_name.lower() else ''
    with gr.Blocks(css=css+".h-\[40vh\] {height: 66.67vh} .gradio-container {max-width: 800px; margin-left: auto; margin-right: auto}", analytics_enabled=False) as interface:
        if args.cai_chat:
            display1 = gr.HTML(value=generate_chat_html([], "", "", character))
        else:
            display1 = gr.Chatbot()
        textbox = gr.Textbox(label='Input')
        btn = gr.Button("Generate")
        with gr.Row():
            stop = gr.Button("Stop")
            btn_regenerate = gr.Button("Regenerate")
            btn_remove_last = gr.Button("Remove last")
            btn_clear = gr.Button("Clear history")

        with gr.Row():
            with gr.Column():
                length_slider = gr.Slider(minimum=settings['max_new_tokens_min'], maximum=settings['max_new_tokens_max'], step=1, label='max_new_tokens', value=settings['max_new_tokens'])
                with gr.Row():
                    model_menu = gr.Dropdown(choices=available_models, value=model_name, label='Model')
                    create_refresh_button(model_menu, lambda : None, lambda : {"choices": get_available_models()}, "refresh-button")
            with gr.Column():
                history_size_slider = gr.Slider(minimum=settings['history_size_min'], maximum=settings['history_size_max'], step=1, label='Chat history size in prompt (0 for no limit)', value=settings['history_size'])
                with gr.Row():
                    preset_menu = gr.Dropdown(choices=available_presets, value=settings[f'preset{suffix}'], label='Generation parameters preset')
                    create_refresh_button(preset_menu, lambda : None, lambda : {"choices": get_available_presets()}, "refresh-button")

        name1 = gr.Textbox(value=settings[f'name1{suffix}'], lines=1, label='Your name')
        name2 = gr.Textbox(value=settings[f'name2{suffix}'], lines=1, label='Bot\'s name')
        context = gr.Textbox(value=settings[f'context{suffix}'], lines=2, label='Context')
        with gr.Row():
            character_menu = gr.Dropdown(choices=available_characters, value="None", label='Character')
            create_refresh_button(character_menu, lambda : None, lambda : {"choices": get_available_characters()}, "refresh-button")

        with gr.Row():
            check = gr.Checkbox(value=settings[f'stop_at_newline{suffix}'], label='Stop generating at new line character?')
        with gr.Row():
            with gr.Tab('Download chat history'):
                download = gr.File()
                save_btn = gr.Button(value="Click me")
            with gr.Tab('Upload chat history'):
                upload = gr.File(type='binary')
            with gr.Tab('Upload character'):
                upload_char = gr.File(type='binary')

        input_params = [textbox, length_slider, preset_menu, model_menu, name1, name2, context, check, history_size_slider]
        if args.cai_chat:
            gen_event = btn.click(cai_chatbot_wrapper, input_params, display1, show_progress=args.no_stream, api_name="textgen")
            gen_event2 = textbox.submit(cai_chatbot_wrapper, input_params, display1, show_progress=args.no_stream)
        else:
            gen_event = btn.click(chatbot_wrapper, input_params, display1, show_progress=args.no_stream, api_name="textgen")
            gen_event2 = textbox.submit(chatbot_wrapper, input_params, display1, show_progress=args.no_stream)
        gen_event3 = btn_regenerate.click(regenerate_wrapper, input_params, display1, show_progress=args.no_stream)

        btn_clear.click(clear_chat_log, [character_menu, name1, name2], display1)
        btn_remove_last.click(remove_last_message, [name1, name2], [display1, textbox], show_progress=False)
        btn.click(lambda x: "", textbox, textbox, show_progress=False)
        btn_regenerate.click(lambda x: "", textbox, textbox, show_progress=False)
        textbox.submit(lambda x: "", textbox, textbox, show_progress=False)
        stop.click(None, None, None, cancels=[gen_event, gen_event2, gen_event3])
        save_btn.click(save_history, inputs=[], outputs=[download])
        character_menu.change(load_character, [character_menu, name1, name2], [name2, context, display1])
        upload.upload(upload_history, [upload, name1, name2], [])
        upload_char.upload(upload_character, [upload_char, name1, name2], [character_menu])

        if args.cai_chat:
            upload.upload(redraw_html, [name1, name2], [display1])
        else:
            upload.upload(lambda : remove_example_dialogue_from_history(history), [], [display1])

elif args.notebook:
    with gr.Blocks(css=css, analytics_enabled=False) as interface:
        gr.Markdown(description)
        with gr.Tab('Raw'):
            textbox = gr.Textbox(value=default_text, lines=23)
        with gr.Tab('Markdown'):
            markdown = gr.Markdown()
        with gr.Tab('HTML'):
            html = gr.HTML()
        btn = gr.Button("Generate")
        stop = gr.Button("Stop")

        length_slider = gr.Slider(minimum=settings['max_new_tokens_min'], maximum=settings['max_new_tokens_max'], step=1, label='max_new_tokens', value=settings['max_new_tokens'])
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    model_menu = gr.Dropdown(choices=available_models, value=model_name, label='Model')
                    create_refresh_button(model_menu, lambda : None, lambda : {"choices": get_available_models()}, "refresh-button")
            with gr.Column():
                with gr.Row():
                    preset_menu = gr.Dropdown(choices=available_presets, value=settings['preset'], label='Generation parameters preset')
                    create_refresh_button(preset_menu, lambda : None, lambda : {"choices": get_available_presets()}, "refresh-button")

        gen_event = btn.click(generate_reply, [textbox, length_slider, preset_menu, model_menu], [textbox, markdown, html], show_progress=args.no_stream, api_name="textgen")
        gen_event2 = textbox.submit(generate_reply, [textbox, length_slider, preset_menu, model_menu], [textbox, markdown, html], show_progress=args.no_stream)
        stop.click(None, None, None, cancels=[gen_event, gen_event2])

else:
    with gr.Blocks(css=css, analytics_enabled=False) as interface:
        gr.Markdown(description)
        with gr.Row():
            with gr.Column():
                textbox = gr.Textbox(value=default_text, lines=15, label='Input')
                length_slider = gr.Slider(minimum=settings['max_new_tokens_min'], maximum=settings['max_new_tokens_max'], step=1, label='max_new_tokens', value=settings['max_new_tokens'])
                with gr.Row():
                    preset_menu = gr.Dropdown(choices=available_presets, value=settings['preset'], label='Generation parameters preset')
                    create_refresh_button(preset_menu, lambda : None, lambda : {"choices": get_available_presets()}, "refresh-button")
                with gr.Row():
                    model_menu = gr.Dropdown(choices=available_models, value=model_name, label='Model')
                    create_refresh_button(model_menu, lambda : None, lambda : {"choices": get_available_models()}, "refresh-button")
                btn = gr.Button("Generate")
                with gr.Row():
                    with gr.Column():
                        cont = gr.Button("Continue")
                    with gr.Column():
                        stop = gr.Button("Stop")
            with gr.Column():
                with gr.Tab('Raw'):
                    output_textbox = gr.Textbox(lines=15, label='Output')
                with gr.Tab('Markdown'):
                    markdown = gr.Markdown()
                with gr.Tab('HTML'):
                    html = gr.HTML()

        gen_event = btn.click(generate_reply, [textbox, length_slider, preset_menu, model_menu], [output_textbox, markdown, html], show_progress=args.no_stream, api_name="textgen")
        gen_event2 = textbox.submit(generate_reply, [textbox, length_slider, preset_menu, model_menu], [output_textbox, markdown, html], show_progress=args.no_stream)
        cont_event = cont.click(generate_reply, [output_textbox, length_slider, preset_menu, model_menu], [output_textbox, markdown, html], show_progress=args.no_stream)
        stop.click(None, None, None, cancels=[gen_event, gen_event2, cont_event])

interface.queue()
if args.listen:
    interface.launch(share=args.share, server_name="0.0.0.0")
else:
    interface.launch(share=args.share)
