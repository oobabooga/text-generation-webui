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
    print('Warning: chat mode currently becomes somewhat slower with text streaming on.\nConsider starting the web UI with the --no-stream option.\n')
    
# Loading custom settings
settings_file = None
if shared.args.settings is not None and Path(shared.args.settings).exists():
    settings_file = Path(shared.args.settings)
elif Path('settings.json').exists():
    settings_file = Path('settings.json')
if settings_file is not None:
    print(f"Loading settings from {settings_file}...")
    new_settings = json.loads(open(settings_file, 'r').read())
    for item in new_settings:
        shared.settings[item] = new_settings[item]

def get_available_models():
    if shared.args.flexgen:
        return sorted([re.sub('-np$', '', item.name) for item in list(Path('models/').glob('*')) if item.name.endswith('-np')], key=str.lower)
    else:
        return sorted([item.name for item in list(Path('models/').glob('*')) if not item.name.endswith(('.txt', '-np'))], key=str.lower)

def get_available_presets():
    return sorted(set(map(lambda x : '.'.join(str(x.name).split('.')[:-1]), Path('presets').glob('*.txt'))), key=str.lower)

def get_available_characters():
    return ['None'] + sorted(set(map(lambda x : '.'.join(str(x.name).split('.')[:-1]), Path('characters').glob('*.json'))), key=str.lower)

def get_available_extensions():
    return sorted(set(map(lambda x : x.parts[1], Path('extensions').glob('*/script.py'))), key=str.lower)

def get_available_softprompts():
    return ['None'] + sorted(set(map(lambda x : '.'.join(str(x.name).split('.')[:-1]), Path('softprompts').glob('*.zip'))), key=str.lower)

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

def create_settings_menus(default_preset):
    generate_params = load_preset_values(default_preset if not shared.args.flexgen else 'Naive', return_dict=True)

    with gr.Row():
        with gr.Column():
            with gr.Row():
                shared.gradio['model_menu'] = gr.Dropdown(choices=available_models, value=shared.model_name, label='Model')
                ui.create_refresh_button(shared.gradio['model_menu'], lambda : None, lambda : {'choices': get_available_models()}, 'refresh-button')
        with gr.Column():
            with gr.Row():
                shared.gradio['preset_menu'] = gr.Dropdown(choices=available_presets, value=default_preset if not shared.args.flexgen else 'Naive', label='Generation parameters preset')
                ui.create_refresh_button(shared.gradio['preset_menu'], lambda : None, lambda : {'choices': get_available_presets()}, 'refresh-button')

    with gr.Accordion('Custom generation parameters', open=False, elem_id='accordion'):
        with gr.Row():
            with gr.Column():
                shared.gradio['temperature'] = gr.Slider(0.01, 1.99, value=generate_params['temperature'], step=0.01, label='temperature')
                shared.gradio['repetition_penalty'] = gr.Slider(1.0, 2.99, value=generate_params['repetition_penalty'],step=0.01,label='repetition_penalty')
                shared.gradio['top_k'] = gr.Slider(0,200,value=generate_params['top_k'],step=1,label='top_k')
                shared.gradio['top_p'] = gr.Slider(0.0,1.0,value=generate_params['top_p'],step=0.01,label='top_p')
            with gr.Column():
                shared.gradio['do_sample'] = gr.Checkbox(value=generate_params['do_sample'], label='do_sample')
                shared.gradio['typical_p'] = gr.Slider(0.0,1.0,value=generate_params['typical_p'],step=0.01,label='typical_p')
                shared.gradio['no_repeat_ngram_size'] = gr.Slider(0, 20, step=1, value=generate_params['no_repeat_ngram_size'], label='no_repeat_ngram_size')
                shared.gradio['min_length'] = gr.Slider(0, 2000, step=1, value=generate_params['min_length'] if shared.args.no_stream else 0, label='min_length', interactive=shared.args.no_stream)

        gr.Markdown('Contrastive search:')
        shared.gradio['penalty_alpha'] = gr.Slider(0, 5, value=generate_params['penalty_alpha'], label='penalty_alpha')

        gr.Markdown('Beam search (uses a lot of VRAM):')
        with gr.Row():
            with gr.Column():
                shared.gradio['num_beams'] = gr.Slider(1, 20, step=1, value=generate_params['num_beams'], label='num_beams')
            with gr.Column():
                shared.gradio['length_penalty'] = gr.Slider(-5, 5, value=generate_params['length_penalty'], label='length_penalty')
        shared.gradio['early_stopping'] = gr.Checkbox(value=generate_params['early_stopping'], label='early_stopping')

    with gr.Accordion('Soft prompt', open=False, elem_id='accordion'):
        with gr.Row():
            shared.gradio['softprompts_menu'] = gr.Dropdown(choices=available_softprompts, value='None', label='Soft prompt')
            ui.create_refresh_button(shared.gradio['softprompts_menu'], lambda : None, lambda : {'choices': get_available_softprompts()}, 'refresh-button')

        gr.Markdown('Upload a soft prompt (.zip format):')
        with gr.Row():
            shared.gradio['upload_softprompt'] = gr.File(type='binary', file_types=['.zip'])

    shared.gradio['model_menu'].change(load_model_wrapper, [shared.gradio['model_menu']], [shared.gradio['model_menu']], show_progress=True)
    shared.gradio['preset_menu'].change(load_preset_values, [shared.gradio['preset_menu']], [shared.gradio['do_sample'], shared.gradio['temperature'], shared.gradio['top_p'], shared.gradio['typical_p'], shared.gradio['repetition_penalty'], shared.gradio['top_k'], shared.gradio['min_length'], shared.gradio['no_repeat_ngram_size'], shared.gradio['num_beams'], shared.gradio['penalty_alpha'], shared.gradio['length_penalty'], shared.gradio['early_stopping']])
    shared.gradio['softprompts_menu'].change(load_soft_prompt, [shared.gradio['softprompts_menu']], [shared.gradio['softprompts_menu']], show_progress=True)
    shared.gradio['upload_softprompt'].upload(upload_soft_prompt, [shared.gradio['upload_softprompt']], [shared.gradio['softprompts_menu']])

available_models = get_available_models()
available_presets = get_available_presets()
available_characters = get_available_characters()
available_softprompts = get_available_softprompts()

# Default extensions
extensions_module.available_extensions = get_available_extensions()
if shared.args.chat or shared.args.cai_chat:
    for extension in shared.settings['chat_default_extensions']:
        shared.args.extensions = shared.args.extensions or []
        if extension not in shared.args.extensions:
            shared.args.extensions.append(extension)
else:
    for extension in shared.settings['default_extensions']:
        shared.args.extensions = shared.args.extensions or []
        if extension not in shared.args.extensions:
            shared.args.extensions.append(extension)
if shared.args.extensions is not None and len(shared.args.extensions) > 0:
    extensions_module.load_extensions()

# Default model
if shared.args.model is not None:
    shared.model_name = shared.args.model
else:
    if len(available_models) == 0:
        print('No models are available! Please download at least one.')
        sys.exit(0)
    elif len(available_models) == 1:
        i = 0
    else:
        print('The following models are available:\n')
        for i, model in enumerate(available_models):
            print(f'{i+1}. {model}')
        print(f'\nWhich one do you want to load? 1-{len(available_models)}\n')
        i = int(input())-1
        print()
    shared.model_name = available_models[i]
shared.model, shared.tokenizer = load_model(shared.model_name)

# Default UI settings
gen_events = []
default_preset = shared.settings['presets'][next((k for k in shared.settings['presets'] if re.match(k.lower(), shared.model_name.lower())), 'default')]
default_text = shared.settings['prompts'][next((k for k in shared.settings['prompts'] if re.match(k.lower(), shared.model_name.lower())), 'default')]
title ='Text generation web UI'
description = '\n\n# Text generation lab\nGenerate text using Large Language Models.\n'
suffix = '_pygmalion' if 'pygmalion' in shared.model_name.lower() else ''

if shared.args.chat or shared.args.cai_chat:
    with gr.Blocks(css=ui.css+ui.chat_css, analytics_enabled=False, title=title) as shared.gradio['interface']:
        if shared.args.cai_chat:
            shared.gradio['display'] = gr.HTML(value=generate_chat_html(shared.history['visible'], shared.settings[f'name1{suffix}'], shared.settings[f'name2{suffix}'], shared.character))
        else:
            shared.gradio['display'] = gr.Chatbot(value=shared.history['visible']).style(color_map=("#326efd", "#212528"))
        shared.gradio['textbox'] = gr.Textbox(label='Input')
        with gr.Row():
            shared.gradio['Stop'] = gr.Button('Stop')
            shared.gradio['Generate'] = gr.Button('Generate')
        with gr.Row():
            shared.gradio['Impersonate'] = gr.Button('Impersonate')
            shared.gradio['Regenerate'] = gr.Button('Regenerate')
        with gr.Row():
            shared.gradio['Copy last reply'] = gr.Button('Copy last reply')
            shared.gradio['Replace last reply'] = gr.Button('Replace last reply')
            shared.gradio['Remove last'] = gr.Button('Remove last')

            shared.gradio['Clear history'] = gr.Button('Clear history')
            shared.gradio['Clear history-confirm'] = gr.Button('Confirm', variant="stop", visible=False)
            shared.gradio['Clear history-cancel'] = gr.Button('Cancel', visible=False)
        with gr.Tab('Chat settings'):
            shared.gradio['name1'] = gr.Textbox(value=shared.settings[f'name1{suffix}'], lines=1, label='Your name')
            shared.gradio['name2'] = gr.Textbox(value=shared.settings[f'name2{suffix}'], lines=1, label='Bot\'s name')
            shared.gradio['context'] = gr.Textbox(value=shared.settings[f'context{suffix}'], lines=5, label='Context')
            with gr.Row():
                shared.gradio['character_menu'] = gr.Dropdown(choices=available_characters, value='None', label='Character', elem_id='character-menu')
                ui.create_refresh_button(shared.gradio['character_menu'], lambda : None, lambda : {'choices': get_available_characters()}, 'refresh-button')

            with gr.Row():
                shared.gradio['check'] = gr.Checkbox(value=shared.settings[f'stop_at_newline{suffix}'], label='Stop generating at new line character?')
            with gr.Row():
                with gr.Tab('Chat history'):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown('Upload')
                            shared.gradio['upload_chat_history'] = gr.File(type='binary', file_types=['.json', '.txt'])
                        with gr.Column():
                            gr.Markdown('Download')
                            shared.gradio['download'] = gr.File()
                            shared.gradio['download_button'] = gr.Button(value='Click me')
                with gr.Tab('Upload character'):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown('1. Select the JSON file')
                            shared.gradio['upload_json'] = gr.File(type='binary', file_types=['.json'])
                        with gr.Column():
                            gr.Markdown('2. Select your character\'s profile picture (optional)')
                            shared.gradio['upload_img_bot'] = gr.File(type='binary', file_types=['image'])
                    shared.gradio['Upload character'] = gr.Button(value='Submit')
                with gr.Tab('Upload your profile picture'):
                    shared.gradio['upload_img_me'] = gr.File(type='binary', file_types=['image'])
                with gr.Tab('Upload TavernAI Character Card'):
                    shared.gradio['upload_img_tavern'] = gr.File(type='binary', file_types=['image'])

        with gr.Tab('Generation settings'):
            with gr.Row():
                with gr.Column():
                    shared.gradio['max_new_tokens'] = gr.Slider(minimum=shared.settings['max_new_tokens_min'], maximum=shared.settings['max_new_tokens_max'], step=1, label='max_new_tokens', value=shared.settings['max_new_tokens'])
                with gr.Column():
                    shared.gradio['chat_prompt_size_slider'] = gr.Slider(minimum=shared.settings['chat_prompt_size_min'], maximum=shared.settings['chat_prompt_size_max'], step=1, label='Maximum prompt size in tokens', value=shared.settings['chat_prompt_size'])
                shared.gradio['chat_generation_attempts'] = gr.Slider(minimum=shared.settings['chat_generation_attempts_min'], maximum=shared.settings['chat_generation_attempts_max'], value=shared.settings['chat_generation_attempts'], step=1, label='Generation attempts (for longer replies)')
            create_settings_menus(default_preset)

        shared.input_params = [shared.gradio[k] for k in ['textbox', 'max_new_tokens', 'do_sample', 'temperature', 'top_p', 'typical_p', 'repetition_penalty', 'top_k', 'min_length', 'no_repeat_ngram_size', 'num_beams', 'penalty_alpha', 'length_penalty', 'early_stopping', 'name1', 'name2', 'context', 'check', 'chat_prompt_size_slider', 'chat_generation_attempts']]
        if shared.args.extensions is not None:
            with gr.Tab('Extensions'):
                extensions_module.create_extensions_block()

        function_call = 'chat.cai_chatbot_wrapper' if shared.args.cai_chat else 'chat.chatbot_wrapper'

        gen_events.append(shared.gradio['Generate'].click(eval(function_call), shared.input_params, shared.gradio['display'], show_progress=shared.args.no_stream, api_name='textgen'))
        gen_events.append(shared.gradio['textbox'].submit(eval(function_call), shared.input_params, shared.gradio['display'], show_progress=shared.args.no_stream))
        gen_events.append(shared.gradio['Regenerate'].click(chat.regenerate_wrapper, shared.input_params, shared.gradio['display'], show_progress=shared.args.no_stream))
        gen_events.append(shared.gradio['Impersonate'].click(chat.impersonate_wrapper, shared.input_params, shared.gradio['textbox'], show_progress=shared.args.no_stream))
        shared.gradio['Stop'].click(chat.stop_everything_event, [], [], cancels=gen_events)

        shared.gradio['Copy last reply'].click(chat.send_last_reply_to_input, [], shared.gradio['textbox'], show_progress=shared.args.no_stream)
        shared.gradio['Replace last reply'].click(chat.replace_last_reply, [shared.gradio['textbox'], shared.gradio['name1'], shared.gradio['name2']], shared.gradio['display'], show_progress=shared.args.no_stream)

        # Clear history with confirmation
        clear_arr = [shared.gradio[k] for k in ['Clear history-confirm', 'Clear history', 'Clear history-cancel']]
        shared.gradio['Clear history'].click(lambda :[gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)], None, clear_arr)
        shared.gradio['Clear history-confirm'].click(lambda :[gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)], None, clear_arr)
        shared.gradio['Clear history-confirm'].click(chat.clear_chat_log, [shared.gradio['name1'], shared.gradio['name2']], shared.gradio['display'])
        shared.gradio['Clear history-cancel'].click(lambda :[gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)], None, clear_arr)

        shared.gradio['Remove last'].click(chat.remove_last_message, [shared.gradio['name1'], shared.gradio['name2']], [shared.gradio['display'], shared.gradio['textbox']], show_progress=False)
        shared.gradio['download_button'].click(chat.save_history, inputs=[], outputs=[shared.gradio['download']])
        shared.gradio['Upload character'].click(chat.upload_character, [shared.gradio['upload_json'], shared.gradio['upload_img_bot']], [shared.gradio['character_menu']])

        # Clearing stuff and saving the history
        for i in ['Generate', 'Regenerate', 'Replace last reply']:
            shared.gradio[i].click(lambda x: '', shared.gradio['textbox'], shared.gradio['textbox'], show_progress=False)
            shared.gradio[i].click(lambda : chat.save_history(timestamp=False), [], [], show_progress=False)
        shared.gradio['Clear history-confirm'].click(lambda : chat.save_history(timestamp=False), [], [], show_progress=False)
        shared.gradio['textbox'].submit(lambda x: '', shared.gradio['textbox'], shared.gradio['textbox'], show_progress=False)
        shared.gradio['textbox'].submit(lambda : chat.save_history(timestamp=False), [], [], show_progress=False)

        shared.gradio['character_menu'].change(chat.load_character, [shared.gradio['character_menu'], shared.gradio['name1'], shared.gradio['name2']], [shared.gradio['name2'], shared.gradio['context'], shared.gradio['display']])
        shared.gradio['upload_chat_history'].upload(chat.load_history, [shared.gradio['upload_chat_history'], shared.gradio['name1'], shared.gradio['name2']], [])
        shared.gradio['upload_img_tavern'].upload(chat.upload_tavern_character, [shared.gradio['upload_img_tavern'], shared.gradio['name1'], shared.gradio['name2']], [shared.gradio['character_menu']])
        shared.gradio['upload_img_me'].upload(chat.upload_your_profile_picture, [shared.gradio['upload_img_me']], [])

        reload_func = chat.redraw_html if shared.args.cai_chat else lambda : shared.history['visible']
        reload_inputs = [shared.gradio['name1'], shared.gradio['name2']] if shared.args.cai_chat else []
        shared.gradio['upload_chat_history'].upload(reload_func, reload_inputs, [shared.gradio['display']])
        shared.gradio['upload_img_me'].upload(reload_func, reload_inputs, [shared.gradio['display']])

        shared.gradio['interface'].load(lambda : chat.load_default_history(shared.settings[f'name1{suffix}'], shared.settings[f'name2{suffix}']), None, None)
        shared.gradio['interface'].load(reload_func, reload_inputs, [shared.gradio['display']], show_progress=True)

elif shared.args.notebook:
    with gr.Blocks(css=ui.css, analytics_enabled=False, title=title) as shared.gradio['interface']:
        gr.Markdown(description)
        with gr.Tab('Raw'):
            shared.gradio['textbox'] = gr.Textbox(value=default_text, lines=23)
        with gr.Tab('Markdown'):
            shared.gradio['markdown'] = gr.Markdown()
        with gr.Tab('HTML'):
            shared.gradio['html'] = gr.HTML()

        shared.gradio['Generate'] = gr.Button('Generate')
        shared.gradio['Stop'] = gr.Button('Stop')
        shared.gradio['max_new_tokens'] = gr.Slider(minimum=shared.settings['max_new_tokens_min'], maximum=shared.settings['max_new_tokens_max'], step=1, label='max_new_tokens', value=shared.settings['max_new_tokens'])

        create_settings_menus(default_preset)
        if shared.args.extensions is not None:
            extensions_module.create_extensions_block()

        shared.input_params = [shared.gradio[k] for k in ['textbox', 'max_new_tokens', 'do_sample', 'temperature', 'top_p', 'typical_p', 'repetition_penalty', 'top_k', 'min_length', 'no_repeat_ngram_size', 'num_beams', 'penalty_alpha', 'length_penalty', 'early_stopping']]
        output_params = [shared.gradio[k] for k in ['textbox', 'markdown', 'html']]
        gen_events.append(shared.gradio['Generate'].click(generate_reply, shared.input_params, output_params, show_progress=shared.args.no_stream, api_name='textgen'))
        gen_events.append(shared.gradio['textbox'].submit(generate_reply, shared.input_params, output_params, show_progress=shared.args.no_stream))
        shared.gradio['Stop'].click(None, None, None, cancels=gen_events)

else:
    with gr.Blocks(css=ui.css, analytics_enabled=False, title=title) as shared.gradio['interface']:
        gr.Markdown(description)
        with gr.Row():
            with gr.Column():
                shared.gradio['textbox'] = gr.Textbox(value=default_text, lines=15, label='Input')
                shared.gradio['max_new_tokens'] = gr.Slider(minimum=shared.settings['max_new_tokens_min'], maximum=shared.settings['max_new_tokens_max'], step=1, label='max_new_tokens', value=shared.settings['max_new_tokens'])
                shared.gradio['Generate'] = gr.Button('Generate')
                with gr.Row():
                    with gr.Column():
                        shared.gradio['Continue'] = gr.Button('Continue')
                    with gr.Column():
                        shared.gradio['Stop'] = gr.Button('Stop')

                create_settings_menus(default_preset)
                if shared.args.extensions is not None:
                    extensions_module.create_extensions_block()

            with gr.Column():
                with gr.Tab('Raw'):
                    shared.gradio['output_textbox'] = gr.Textbox(lines=15, label='Output')
                with gr.Tab('Markdown'):
                    shared.gradio['markdown'] = gr.Markdown()
                with gr.Tab('HTML'):
                    shared.gradio['html'] = gr.HTML()

        shared.input_params = [shared.gradio[k] for k in ['textbox', 'max_new_tokens', 'do_sample', 'temperature', 'top_p', 'typical_p', 'repetition_penalty', 'top_k', 'min_length', 'no_repeat_ngram_size', 'num_beams', 'penalty_alpha', 'length_penalty', 'early_stopping']]
        output_params = [shared.gradio[k] for k in ['output_textbox', 'markdown', 'html']]
        gen_events.append(shared.gradio['Generate'].click(generate_reply, shared.input_params, output_params, show_progress=shared.args.no_stream, api_name='textgen'))
        gen_events.append(shared.gradio['textbox'].submit(generate_reply, shared.input_params, output_params, show_progress=shared.args.no_stream))
        gen_events.append(shared.gradio['Continue'].click(generate_reply, [shared.gradio['output_textbox']] + shared.input_params[1:], output_params, show_progress=shared.args.no_stream))
        shared.gradio['Stop'].click(None, None, None, cancels=gen_events)

shared.gradio['interface'].queue()
if shared.args.listen:
    shared.gradio['interface'].launch(prevent_thread_lock=True, share=shared.args.share, server_name='0.0.0.0', server_port=shared.args.listen_port)
else:
    shared.gradio['interface'].launch(prevent_thread_lock=True, share=shared.args.share, server_port=shared.args.listen_port)

# I think that I will need this later
while True:
    time.sleep(0.5)
