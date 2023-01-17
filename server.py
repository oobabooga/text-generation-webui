import re
import time
import glob
from sys import exit
import torch
import argparse
import json
from pathlib import Path
import gradio as gr
import transformers
from html_generator import *
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings


transformers.logging.set_verbosity_error()

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='Name of the model to load by default.')
parser.add_argument('--notebook', action='store_true', help='Launch the web UI in notebook mode, where the output is written to the same text box as the input.')
parser.add_argument('--chat', action='store_true', help='Launch the web UI in chat mode.')
parser.add_argument('--cai-chat', action='store_true', help='Launch the web UI in chat mode with a style similar to Character.AI\'s. If the file profile.png or profile.jpg exists in the same folder as server.py, this image will be used as the bot\'s profile picture.')
parser.add_argument('--cpu', action='store_true', help='Use the CPU to generate text.')
parser.add_argument('--auto-devices', action='store_true', help='Automatically split the model across the available GPU(s) and CPU.')
parser.add_argument('--load-in-8bit', action='store_true', help='Load the model with 8-bit precision.')
parser.add_argument('--max-gpu-memory', type=int, help='Maximum memory in GiB to allocate to the GPU when loading the model. This is useful if you get out of memory errors while trying to generate text. Must be an integer number.')
parser.add_argument('--no-listen', action='store_true', help='Make the web UI unreachable from your local network.')
parser.add_argument('--settings', type=str, help='Load the default interface settings from this json file. See settings-template.json for an example.')
args = parser.parse_args()

loaded_preset = None
available_models = sorted(set([item.replace('.pt', '') for item in map(lambda x : str(x.name), list(Path('models/').glob('*'))+list(Path('torch-dumps/').glob('*'))) if not item.endswith('.txt')]), key=str.lower)
available_presets = sorted(set(map(lambda x : str(x.name).split('.')[0], Path('presets').glob('*.txt'))), key=str.lower)

settings = {
    'max_new_tokens': 200,
    'max_new_tokens_min': 1,
    'max_new_tokens_max': 2000,
    'preset': 'NovelAI-Sphinx Moth',
    'name1': 'Person 1',
    'name2': 'Person 2',
    'name1_pygmalion': 'You',
    'name2_pygmalion': 'Kawaii',
    'context': 'This is a conversation between two people.',
    'context_pygmalion': 'This is a conversation between two people.\n<START>',
    'prompt': 'Common sense questions and answers\n\nQuestion: \nFactual answer:',
    'prompt_gpt4chan': '-----\n--- 865467536\nInput text\n--- 865467537\n',
    'stop_at_newline': True,
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
    if not (args.cpu or args.auto_devices or args.load_in_8bit or args.max_gpu_memory is not None):
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
            if args.max_gpu_memory is not None:
                settings.append(f"max_memory={{0: '{args.max_gpu_memory}GiB', 'cpu': '99GiB'}}")
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

def encode(prompt, tokens):
    if not args.cpu:
        torch.cuda.empty_cache()
        input_ids = tokenizer.encode(str(prompt), return_tensors='pt', truncation=True, max_length=2048-tokens).cuda()
    else:
        input_ids = tokenizer.encode(str(prompt), return_tensors='pt', truncation=True, max_length=2048-tokens)
    return input_ids

def generate_reply(question, tokens, inference_settings, selected_model, eos_token=None):
    global model, tokenizer, model_name, loaded_preset, preset

    if selected_model != model_name:
        model_name = selected_model
        model = None
        tokenizer = None
        if not args.cpu:
            torch.cuda.empty_cache()
        model, tokenizer = load_model(model_name)
    if inference_settings != loaded_preset:
        with open(Path(f'presets/{inference_settings}.txt'), 'r') as infile:
            preset = infile.read()
        loaded_preset = inference_settings

    input_ids = encode(question, tokens)

    cuda = ".cuda()" if args.cpu else ""
    if eos_token is None:
        output = eval(f"model.generate(input_ids, {preset}){cuda}")
    else:
        n = tokenizer.encode(eos_token, return_tensors='pt')[0][-1]
        output = eval(f"model.generate(input_ids, eos_token_id={n}, {preset}){cuda}")

    reply = tokenizer.decode(output[0], skip_special_tokens=True)
    reply = reply.replace(r'<|endoftext|>', '')
    if model_name.lower().startswith('galactica'):
        reply = fix_galactica(reply)
        return reply, reply, generate_basic_html(reply)
    elif model_name.lower().startswith('gpt4chan'):
        reply = fix_gpt4chan(reply)
        return reply, 'Only applicable for GALACTICA models.', generate_4chan_html(reply)
    else:
        return reply, 'Only applicable for GALACTICA models.', generate_basic_html(reply)

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

# UI settings
if model_name.lower().startswith('gpt4chan'):
    default_text = settings['prompt_gpt4chan']
else:
    default_text = settings['prompt']

description = f"\n\n# Text generation lab\nGenerate text using Large Language Models.\n"
css = ".my-4 {margin-top: 0} .py-6 {padding-top: 2.5rem}"

if args.notebook:
    with gr.Blocks(css=css, analytics_enabled=False) as interface:
        gr.Markdown(description)
        with gr.Tab('Raw'):
            textbox = gr.Textbox(value=default_text, lines=23)
        with gr.Tab('Markdown'):
            markdown = gr.Markdown()
        with gr.Tab('HTML'):
            html = gr.HTML()
        btn = gr.Button("Generate")

        length_slider = gr.Slider(minimum=settings['max_new_tokens_min'], maximum=settings['max_new_tokens_max'], step=1, label='max_new_tokens', value=settings['max_new_tokens'])
        with gr.Row():
            with gr.Column():
                model_menu = gr.Dropdown(choices=available_models, value=model_name, label='Model')
            with gr.Column():
                preset_menu = gr.Dropdown(choices=available_presets, value=settings['preset'], label='Settings preset')

        btn.click(generate_reply, [textbox, length_slider, preset_menu, model_menu], [textbox, markdown, html], show_progress=True, api_name="textgen")
        textbox.submit(generate_reply, [textbox, length_slider, preset_menu, model_menu], [textbox, markdown, html], show_progress=True)
elif args.chat or args.cai_chat:
    history = []

    # This gets the new line characters right.
    def chat_response_cleaner(text):
        text = text.replace('\n', '\n\n')
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        return text

    def chatbot_wrapper(text, tokens, inference_settings, selected_model, name1, name2, context, check):
        text = chat_response_cleaner(text)

        rows = [f"{context}\n\n"]
        i = len(history)-1
        while i >= 0 and len(encode(''.join(rows), tokens)[0]) < 2048-tokens:
            rows.insert(1, f"{name2}: {history[i][1].strip()}\n")
            rows.insert(1, f"{name1}: {history[i][0].strip()}\n")
            i -= 1
        rows.append(f"{name1}: {text}\n")
        rows.append(f"{name2}:")

        while len(rows) > 3 and len(encode(''.join(rows), tokens)[0]) >= 2048-tokens:
            rows.pop(1)
            rows.pop(1)

        question = ''.join(rows)

        if check:
            reply = generate_reply(question, tokens, inference_settings, selected_model, eos_token='\n')[0]
            idx = reply.rfind(question[-1024:])
            reply = reply[idx+min(1024, len(question)):].split('\n')[0].strip()
        else:
            reply = generate_reply(question, tokens, inference_settings, selected_model)[0]
            idx = reply.rfind(question[-1024:])
            reply = reply[idx+min(1024, len(question)):]
            idx = reply.find(f"\n{name1}:")
            if idx != -1:
                reply = reply[:idx]
            reply = chat_response_cleaner(reply)

        history.append((text, reply))
        return history

    def cai_chatbot_wrapper(text, tokens, inference_settings, selected_model, name1, name2, context, check):
        history = chatbot_wrapper(text, tokens, inference_settings, selected_model, name1, name2, context, check)
        return generate_chat_html(history, name1, name2)

    def remove_last_message(name1, name2):
        history.pop()
        if args.cai_chat:
            return generate_chat_html(history, name1, name2)
        else:
            return history

    def clear():
        global history
        history = []

    def clear_html():
        return generate_chat_html([], "", "")

    if 'pygmalion' in model_name.lower():
        context_str = settings['context_pygmalion']
        name1_str = settings['name1_pygmalion']
        name2_str = settings['name2_pygmalion']
    else:
        context_str = settings['context']
        name1_str = settings['name1']
        name2_str = settings['name2']

    with gr.Blocks(css=css+".h-\[40vh\] {height: 66.67vh} .gradio-container {max-width: 800px; margin-left: auto; margin-right: auto}", analytics_enabled=False) as interface:
        if args.cai_chat:
            display1 = gr.HTML(value=generate_chat_html([], "", ""))
        else:
            display1 = gr.Chatbot()
        textbox = gr.Textbox(lines=2, label='Input')
        btn = gr.Button("Generate")
        with gr.Row():
            with gr.Column():
                btn3 = gr.Button("Remove last message")
            with gr.Column():
                btn2 = gr.Button("Clear history")

        length_slider = gr.Slider(minimum=settings['max_new_tokens_min'], maximum=settings['max_new_tokens_max'], step=1, label='max_new_tokens', value=settings['max_new_tokens'])
        with gr.Row():
            with gr.Column():
                model_menu = gr.Dropdown(choices=available_models, value=model_name, label='Model')
            with gr.Column():
                preset_menu = gr.Dropdown(choices=available_presets, value=settings['preset'], label='Settings preset')

        name1 = gr.Textbox(value=name1_str, lines=1, label='Your name')
        name2 = gr.Textbox(value=name2_str, lines=1, label='Bot\'s name')
        context = gr.Textbox(value=context_str, lines=2, label='Context')
        with gr.Row():
            check = gr.Checkbox(value=settings['stop_at_newline'], label='Stop generating at new line character?')

        if args.cai_chat:
            btn.click(cai_chatbot_wrapper, [textbox, length_slider, preset_menu, model_menu, name1, name2, context, check], display1, show_progress=True, api_name="textgen")
            textbox.submit(cai_chatbot_wrapper, [textbox, length_slider, preset_menu, model_menu, name1, name2, context, check], display1, show_progress=True)
            btn2.click(clear_html, [], display1, show_progress=False)
        else:
            btn.click(chatbot_wrapper, [textbox, length_slider, preset_menu, model_menu, name1, name2, context, check], display1, show_progress=True, api_name="textgen")
            textbox.submit(chatbot_wrapper, [textbox, length_slider, preset_menu, model_menu, name1, name2, context, check], display1, show_progress=True)
            btn2.click(lambda x: "", display1, display1)

        btn2.click(clear)
        btn3.click(remove_last_message, [name1, name2], display1, show_progress=False)
        btn.click(lambda x: "", textbox, textbox, show_progress=False)
        textbox.submit(lambda x: "", textbox, textbox, show_progress=False)
else:

    def continue_wrapper(question, tokens, inference_settings, selected_model):
        a, b, c = generate_reply(question, tokens, inference_settings, selected_model)
        return a, a, b, c

    with gr.Blocks(css=css, analytics_enabled=False) as interface:
        gr.Markdown(description)
        with gr.Row():
            with gr.Column():
                textbox = gr.Textbox(value=default_text, lines=15, label='Input')
                length_slider = gr.Slider(minimum=settings['max_new_tokens_min'], maximum=settings['max_new_tokens_max'], step=1, label='max_new_tokens', value=settings['max_new_tokens'])
                preset_menu = gr.Dropdown(choices=available_presets, value=settings['preset'], label='Settings preset')
                model_menu = gr.Dropdown(choices=available_models, value=model_name, label='Model')
                btn = gr.Button("Generate")
                cont = gr.Button("Continue")
            with gr.Column():
                with gr.Tab('Raw'):
                    output_textbox = gr.Textbox(lines=15, label='Output')
                with gr.Tab('Markdown'):
                    markdown = gr.Markdown()
                with gr.Tab('HTML'):
                    html = gr.HTML()

        btn.click(generate_reply, [textbox, length_slider, preset_menu, model_menu], [output_textbox, markdown, html], show_progress=True, api_name="textgen")
        cont.click(continue_wrapper, [output_textbox, length_slider, preset_menu, model_menu], [output_textbox, textbox, markdown, html], show_progress=True)
        textbox.submit(generate_reply, [textbox, length_slider, preset_menu, model_menu], [output_textbox, markdown, html], show_progress=True)

if args.no_listen:
    interface.launch(share=False)
else:
    interface.launch(share=False, server_name="0.0.0.0")
