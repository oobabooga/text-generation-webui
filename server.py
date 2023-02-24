import gc
import io
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import gradio as gr
import torch

import modules.chat as chat
import modules.extensions as extensions_module
import modules.shared as shared
import modules.ui as ui
from modules.html_generator import generate_chat_html
from modules.models import load_model, load_soft_prompt
from modules.text_generation import generate_reply

if (shared.args.chat or shared.args.cai_chat) and not shared.args.no_stream:
    print("Warning: chat mode currently becomes somewhat slower with text streaming on.\nConsider starting the web UI with the --no-stream option.\n")
    
# Loading custom settings
if shared.args.settings is not None and Path(shared.args.settings).exists():
    new_settings = json.loads(open(Path(shared.args.settings), 'r').read())
    for item in new_settings:
        shared.settings[item] = new_settings[item]

def get_available_models():
    return sorted([item.name for item in list(Path('models/').glob('*')) if not item.name.endswith(('.txt', '-np'))], key=str.lower)

def get_available_presets():
    return sorted(set(map(lambda x : '.'.join(str(x.name).split('.')[:-1]), Path('presets').glob('*.txt'))), key=str.lower)

def get_available_characters():
    return ["None"] + sorted(set(map(lambda x : '.'.join(str(x.name).split('.')[:-1]), Path('characters').glob('*.json'))), key=str.lower)

def get_available_extensions():
    return sorted(set(map(lambda x : x.parts[1], Path('extensions').glob('*/script.py'))), key=str.lower)

def get_available_softprompts():
    return ["None"] + sorted(set(map(lambda x : '.'.join(str(x.name).split('.')[:-1]), Path('softprompts').glob('*.zip'))), key=str.lower)

def load_model_wrapper(selected_model):
    if selected_model != shared.model_name:
        shared.model_name = selected_model
        shared.model = shared.tokenizer = None
        if not shared.args.cpu:
            gc.collect()
            torch.cuda.empty_cache()
        shared.model, shared.tokenizer = load_model(shared.model_name)

    return selected_model

def load_preset_values(preset_menu, return_dict=False):
    generate_params = {
        'do_sample': True,
        'temperature': 1,
        'top_p': 1,
        'typical_p': 1,
        'repetition_penalty': 1,
        'top_k': 50,
        'num_beams': 1,
        'penalty_alpha': 0,
        'min_length': 0,
        'length_penalty': 1,
        'no_repeat_ngram_size': 0,
        'early_stopping': False,
    }
    with open(Path(f'presets/{preset_menu}.txt'), 'r') as infile:
        preset = infile.read()
    for i in preset.splitlines():
        i = i.rstrip(',').strip().split('=')
        if len(i) == 2 and i[0].strip() != 'tokens':
            generate_params[i[0].strip()] = eval(i[1].strip())

    generate_params['temperature'] = min(1.99, generate_params['temperature'])

    if return_dict:
        return generate_params
    else:
        return generate_params['do_sample'], generate_params['temperature'], generate_params['top_p'], generate_params['typical_p'], generate_params['repetition_penalty'], generate_params['top_k'], generate_params['min_length'], generate_params['no_repeat_ngram_size'], generate_params['num_beams'], generate_params['penalty_alpha'], generate_params['length_penalty'], generate_params['early_stopping']

def upload_soft_prompt(file):
    with zipfile.ZipFile(io.BytesIO(file)) as zf:
        zf.extract('meta.json')
        j = json.loads(open('meta.json', 'r').read())
        name = j['name']
        Path('meta.json').unlink()

    with open(Path(f'softprompts/{name}.zip'), 'wb') as f:
        f.write(file)

    return name

def create_settings_menus():
    generate_params = load_preset_values(shared.settings[f'preset{suffix}'] if not shared.args.flexgen else 'Naive', return_dict=True)

    with gr.Row():
        with gr.Column():
            with gr.Row():
                model_menu = gr.Dropdown(choices=available_models, value=shared.model_name, label='Model')
                ui.create_refresh_button(model_menu, lambda : None, lambda : {"choices": get_available_models()}, "refresh-button")
        with gr.Column():
            with gr.Row():
                preset_menu = gr.Dropdown(choices=available_presets, value=shared.settings[f'preset{suffix}'] if not shared.args.flexgen else 'Naive', label='Generation parameters preset')
                ui.create_refresh_button(preset_menu, lambda : None, lambda : {"choices": get_available_presets()}, "refresh-button")

    with gr.Accordion("Custom generation parameters", open=False, elem_id="accordion"):
        with gr.Row():
            do_sample = gr.Checkbox(value=generate_params['do_sample'], label="do_sample")
            temperature = gr.Slider(0.01, 1.99, value=generate_params['temperature'], step=0.01, label="temperature")
        with gr.Row():
            top_k = gr.Slider(0,200,value=generate_params['top_k'],step=1,label="top_k")
            top_p = gr.Slider(0.0,1.0,value=generate_params['top_p'],step=0.01,label="top_p")
        with gr.Row():
            repetition_penalty = gr.Slider(1.0,4.99,value=generate_params['repetition_penalty'],step=0.01,label="repetition_penalty")
            no_repeat_ngram_size = gr.Slider(0, 20, step=1, value=generate_params["no_repeat_ngram_size"], label="no_repeat_ngram_size")
        with gr.Row():
            typical_p = gr.Slider(0.0,1.0,value=generate_params['typical_p'],step=0.01,label="typical_p")
            min_length = gr.Slider(0, 2000, step=1, value=generate_params["min_length"] if shared.args.no_stream else 0, label="min_length", interactive=shared.args.no_stream)

        gr.Markdown("Contrastive search:")
        penalty_alpha = gr.Slider(0, 5, value=generate_params["penalty_alpha"], label="penalty_alpha")

        gr.Markdown("Beam search (uses a lot of VRAM):")
        with gr.Row():
            num_beams = gr.Slider(1, 20, step=1, value=generate_params["num_beams"], label="num_beams")
            length_penalty = gr.Slider(-5, 5, value=generate_params["length_penalty"], label="length_penalty")
        early_stopping = gr.Checkbox(value=generate_params["early_stopping"], label="early_stopping")

    with gr.Accordion("Soft prompt", open=False, elem_id="accordion"):
        with gr.Row():
            softprompts_menu = gr.Dropdown(choices=available_softprompts, value="None", label='Soft prompt')
            ui.create_refresh_button(softprompts_menu, lambda : None, lambda : {"choices": get_available_softprompts()}, "refresh-button")

        gr.Markdown('Upload a soft prompt (.zip format):')
        with gr.Row():
            upload_softprompt = gr.File(type='binary', file_types=[".zip"])

    model_menu.change(load_model_wrapper, [model_menu], [model_menu], show_progress=True)
    preset_menu.change(load_preset_values, [preset_menu], [do_sample, temperature, top_p, typical_p, repetition_penalty, top_k, min_length, no_repeat_ngram_size, num_beams, penalty_alpha, length_penalty, early_stopping])
    softprompts_menu.change(load_soft_prompt, [softprompts_menu], [softprompts_menu], show_progress=True)
    upload_softprompt.upload(upload_soft_prompt, [upload_softprompt], [softprompts_menu])
    return preset_menu, do_sample, temperature, top_p, typical_p, repetition_penalty, top_k, min_length, no_repeat_ngram_size, num_beams, penalty_alpha, length_penalty, early_stopping

available_models = get_available_models()
available_presets = get_available_presets()
available_characters = get_available_characters()
available_softprompts = get_available_softprompts()

extensions_module.available_extensions = get_available_extensions()
if shared.args.extensions is not None:
    extensions_module.load_extensions()

# Choosing the default model
if shared.args.model is not None:
    shared.model_name = shared.args.model
else:
    if len(available_models) == 0:
        print("No models are available! Please download at least one.")
        sys.exit(0)
    elif len(available_models) == 1:
        i = 0
    else:
        print("The following models are available:\n")
        for i, model in enumerate(available_models):
            print(f"{i+1}. {model}")
        print(f"\nWhich one do you want to load? 1-{len(available_models)}\n")
        i = int(input())-1
        print()
    shared.model_name = available_models[i]
shared.model, shared.tokenizer = load_model(shared.model_name)

# UI settings
buttons = {}
gen_events = []
suffix = '_pygmalion' if 'pygmalion' in shared.model_name.lower() else ''
description = f"\n\n# Text generation lab\nGenerate text using Large Language Models.\n"
if shared.model_name.lower().startswith(('gpt4chan', 'gpt-4chan', '4chan')):
    default_text = shared.settings['prompt_gpt4chan']
elif re.match('(rosey|chip|joi)_.*_instruct.*', shared.model_name.lower()) is not None:
    default_text = 'User: \n'
else:
    default_text = shared.settings['prompt']

if shared.args.chat or shared.args.cai_chat:
    with gr.Blocks(css=ui.css+ui.chat_css, analytics_enabled=False) as interface:
        interface.load(lambda : chat.load_default_history(shared.settings[f'name1{suffix}'], shared.settings[f'name2{suffix}']), None, None)
        if shared.args.cai_chat:
            display = gr.HTML(value=generate_chat_html(shared.history['visible'], shared.settings[f'name1{suffix}'], shared.settings[f'name2{suffix}'], shared.character))
        else:
            display = gr.Chatbot(value=shared.history['visible'])
        textbox = gr.Textbox(label='Input')
        with gr.Row():
            buttons["Stop"] = gr.Button("Stop")
            buttons["Generate"] = gr.Button("Generate")
            buttons["Regenerate"] = gr.Button("Regenerate")
        with gr.Row():
            buttons["Impersonate"] = gr.Button("Impersonate")
            buttons["Remove last"] = gr.Button("Remove last")
            buttons["Clear history"] = gr.Button("Clear history")
        with gr.Row():
            buttons["Send last reply to input"] = gr.Button("Send last reply to input")
            buttons["Replace last reply"] = gr.Button("Replace last reply")
        if shared.args.picture:
            with gr.Row():
                picture_select = gr.Image(label="Send a picture", type='pil')

        with gr.Tab("Chat settings"):
            name1 = gr.Textbox(value=shared.settings[f'name1{suffix}'], lines=1, label='Your name')
            name2 = gr.Textbox(value=shared.settings[f'name2{suffix}'], lines=1, label='Bot\'s name')
            context = gr.Textbox(value=shared.settings[f'context{suffix}'], lines=2, label='Context')
            with gr.Row():
                character_menu = gr.Dropdown(choices=available_characters, value="None", label='Character')
                ui.create_refresh_button(character_menu, lambda : None, lambda : {"choices": get_available_characters()}, "refresh-button")

            with gr.Row():
                check = gr.Checkbox(value=shared.settings[f'stop_at_newline{suffix}'], label='Stop generating at new line character?')
            with gr.Row():
                with gr.Tab('Chat history'):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown('Upload')
                            upload_chat_history = gr.File(type='binary', file_types=[".json", ".txt"])
                        with gr.Column():
                            gr.Markdown('Download')
                            download = gr.File()
                            buttons["Download"] = gr.Button(value="Click me")
                with gr.Tab('Upload character'):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown('1. Select the JSON file')
                            upload_char = gr.File(type='binary', file_types=[".json"])
                        with gr.Column():
                            gr.Markdown('2. Select your character\'s profile picture (optional)')
                            upload_img = gr.File(type='binary', file_types=["image"])
                    buttons["Upload character"] = gr.Button(value="Submit")
                with gr.Tab('Upload your profile picture'):
                    upload_img_me = gr.File(type='binary', file_types=["image"])
                with gr.Tab('Upload TavernAI Character Card'):
                    upload_img_tavern = gr.File(type='binary', file_types=["image"])

        with gr.Tab("Generation settings"):
            with gr.Row():
                with gr.Column():
                    max_new_tokens = gr.Slider(minimum=shared.settings['max_new_tokens_min'], maximum=shared.settings['max_new_tokens_max'], step=1, label='max_new_tokens', value=shared.settings['max_new_tokens'])
                with gr.Column():
                    chat_prompt_size_slider = gr.Slider(minimum=shared.settings['chat_prompt_size_min'], maximum=shared.settings['chat_prompt_size_max'], step=1, label='Maximum prompt size in tokens', value=shared.settings['chat_prompt_size'])

            preset_menu, do_sample, temperature, top_p, typical_p, repetition_penalty, top_k, min_length, no_repeat_ngram_size, num_beams, penalty_alpha, length_penalty, early_stopping = create_settings_menus()

        if shared.args.extensions is not None:
            with gr.Tab("Extensions"):
                extensions_module.create_extensions_block()

        input_params = [textbox, max_new_tokens, do_sample, max_new_tokens, temperature, top_p, typical_p, repetition_penalty, top_k, min_length, no_repeat_ngram_size, num_beams, penalty_alpha, length_penalty, early_stopping, name1, name2, context, check, chat_prompt_size_slider]
        if shared.args.picture:
            input_params.append(picture_select)
        function_call = "chat.cai_chatbot_wrapper" if shared.args.cai_chat else "chat.chatbot_wrapper"

        gen_events.append(buttons["Generate"].click(eval(function_call), input_params, display, show_progress=shared.args.no_stream, api_name="textgen"))
        gen_events.append(textbox.submit(eval(function_call), input_params, display, show_progress=shared.args.no_stream))
        if shared.args.picture:
            picture_select.upload(eval(function_call), input_params, display, show_progress=shared.args.no_stream)
        gen_events.append(buttons["Regenerate"].click(chat.regenerate_wrapper, input_params, display, show_progress=shared.args.no_stream))
        gen_events.append(buttons["Impersonate"].click(chat.impersonate_wrapper, input_params, textbox, show_progress=shared.args.no_stream))
        buttons["Stop"].click(chat.stop_everything_event, [], [], cancels=gen_events)

        buttons["Send last reply to input"].click(chat.send_last_reply_to_input, [], textbox, show_progress=shared.args.no_stream)
        buttons["Replace last reply"].click(chat.replace_last_reply, [textbox, name1, name2], display, show_progress=shared.args.no_stream)
        buttons["Clear history"].click(chat.clear_chat_log, [name1, name2], display)
        buttons["Remove last"].click(chat.remove_last_message, [name1, name2], [display, textbox], show_progress=False)
        buttons["Download"].click(chat.save_history, inputs=[], outputs=[download])
        buttons["Upload character"].click(chat.upload_character, [upload_char, upload_img], [character_menu])

        # Clearing stuff and saving the history
        for i in ["Generate", "Regenerate", "Replace last reply"]:
            buttons[i].click(lambda x: "", textbox, textbox, show_progress=False)
            buttons[i].click(lambda : chat.save_history(timestamp=False), [], [], show_progress=False)
        buttons["Clear history"].click(lambda : chat.save_history(timestamp=False), [], [], show_progress=False)
        textbox.submit(lambda x: "", textbox, textbox, show_progress=False)
        textbox.submit(lambda : chat.save_history(timestamp=False), [], [], show_progress=False)

        character_menu.change(chat.load_character, [character_menu, name1, name2], [name2, context, display])
        upload_chat_history.upload(chat.load_history, [upload_chat_history, name1, name2], [])
        upload_img_tavern.upload(chat.upload_tavern_character, [upload_img_tavern, name1, name2], [character_menu])
        upload_img_me.upload(chat.upload_your_profile_picture, [upload_img_me], [])
        if shared.args.picture:
            picture_select.upload(lambda : None, [], [picture_select], show_progress=False)

        reload_func = chat.redraw_html if shared.args.cai_chat else lambda : shared.history['visible']
        reload_inputs = [name1, name2] if shared.args.cai_chat else []
        upload_chat_history.upload(reload_func, reload_inputs, [display])
        upload_img_me.upload(reload_func, reload_inputs, [display])
        interface.load(reload_func, reload_inputs, [display], show_progress=True)

elif shared.args.notebook:
    with gr.Blocks(css=ui.css, analytics_enabled=False) as interface:
        gr.Markdown(description)
        with gr.Tab('Raw'):
            textbox = gr.Textbox(value=default_text, lines=23)
        with gr.Tab('Markdown'):
            markdown = gr.Markdown()
        with gr.Tab('HTML'):
            html = gr.HTML()

        buttons["Generate"] = gr.Button("Generate")
        buttons["Stop"] = gr.Button("Stop")

        max_new_tokens = gr.Slider(minimum=shared.settings['max_new_tokens_min'], maximum=shared.settings['max_new_tokens_max'], step=1, label='max_new_tokens', value=shared.settings['max_new_tokens'])

        preset_menu, do_sample, temperature, top_p, typical_p, repetition_penalty, top_k, min_length, no_repeat_ngram_size, num_beams, penalty_alpha, length_penalty, early_stopping = create_settings_menus()

        if shared.args.extensions is not None:
            extensions_module.create_extensions_block()

        gen_events.append(buttons["Generate"].click(generate_reply, [textbox, max_new_tokens, do_sample, max_new_tokens, temperature, top_p, typical_p, repetition_penalty, top_k, min_length, no_repeat_ngram_size, num_beams, penalty_alpha, length_penalty, early_stopping], [textbox, markdown, html], show_progress=shared.args.no_stream, api_name="textgen"))
        gen_events.append(textbox.submit(generate_reply, [textbox, max_new_tokens, do_sample, max_new_tokens, temperature, top_p, typical_p, repetition_penalty, top_k, min_length, no_repeat_ngram_size, num_beams, penalty_alpha, length_penalty, early_stopping], [textbox, markdown, html], show_progress=shared.args.no_stream))
        buttons["Stop"].click(None, None, None, cancels=gen_events)

else:
    with gr.Blocks(css=ui.css, analytics_enabled=False) as interface:
        gr.Markdown(description)
        with gr.Row():
            with gr.Column():
                textbox = gr.Textbox(value=default_text, lines=15, label='Input')
                max_new_tokens = gr.Slider(minimum=shared.settings['max_new_tokens_min'], maximum=shared.settings['max_new_tokens_max'], step=1, label='max_new_tokens', value=shared.settings['max_new_tokens'])
                buttons["Generate"] = gr.Button("Generate")
                with gr.Row():
                    with gr.Column():
                        buttons["Continue"] = gr.Button("Continue")
                    with gr.Column():
                        buttons["Stop"] = gr.Button("Stop")

                preset_menu, do_sample, temperature, top_p, typical_p, repetition_penalty, top_k, min_length, no_repeat_ngram_size, num_beams, penalty_alpha, length_penalty, early_stopping = create_settings_menus()
                if shared.args.extensions is not None:
                    extensions_module.create_extensions_block()

            with gr.Column():
                with gr.Tab('Raw'):
                    output_textbox = gr.Textbox(lines=15, label='Output')
                with gr.Tab('Markdown'):
                    markdown = gr.Markdown()
                with gr.Tab('HTML'):
                    html = gr.HTML()

        gen_events.append(buttons["Generate"].click(generate_reply, [textbox, max_new_tokens, do_sample, max_new_tokens, temperature, top_p, typical_p, repetition_penalty, top_k, min_length, no_repeat_ngram_size, num_beams, penalty_alpha, length_penalty, early_stopping], [output_textbox, markdown, html], show_progress=shared.args.no_stream, api_name="textgen"))
        gen_events.append(textbox.submit(generate_reply, [textbox, max_new_tokens, do_sample, max_new_tokens, temperature, top_p, typical_p, repetition_penalty, top_k, min_length, no_repeat_ngram_size, num_beams, penalty_alpha, length_penalty, early_stopping], [output_textbox, markdown, html], show_progress=shared.args.no_stream))
        gen_events.append(buttons["Continue"].click(generate_reply, [output_textbox, max_new_tokens, do_sample, max_new_tokens, temperature, top_p, typical_p, repetition_penalty, top_k, min_length, no_repeat_ngram_size, num_beams, penalty_alpha, length_penalty, early_stopping], [output_textbox, markdown, html], show_progress=shared.args.no_stream))
        buttons["Stop"].click(None, None, None, cancels=gen_events)

interface.queue()
if shared.args.listen:
    interface.launch(prevent_thread_lock=True, share=shared.args.share, server_name="0.0.0.0", server_port=shared.args.listen_port)
else:
    interface.launch(prevent_thread_lock=True, share=shared.args.share, server_port=shared.args.listen_port)

# I think that I will need this later
while True:
    time.sleep(0.5)
