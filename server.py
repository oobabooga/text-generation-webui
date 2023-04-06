import os

os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

import io
import json
import re
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import gradio as gr
from PIL import Image

import modules.extensions as extensions_module
from modules import chat, shared, training, ui, api
from modules.html_generator import chat_html_wrapper
from modules.LoRA import add_lora_to_model
from modules.models import load_model, load_soft_prompt
from modules.text_generation import (clear_torch_cache, generate_reply,
                                     stop_everything_event)

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
        return sorted([re.sub('-np$', '', item.name) for item in list(Path(f'{shared.args.model_dir}/').glob('*')) if item.name.endswith('-np')], key=str.lower)
    else:
        return sorted([re.sub('.pth$', '', item.name) for item in list(Path(f'{shared.args.model_dir}/').glob('*')) if not item.name.endswith(('.txt', '-np', '.pt', '.json'))], key=str.lower)

def get_available_presets():
    return sorted(set((k.stem for k in Path('presets').glob('*.txt'))), key=str.lower)

def get_available_prompts():
    prompts = []
    prompts += sorted(set((k.stem for k in Path('prompts').glob('[0-9]*.txt'))), key=str.lower, reverse=True)
    prompts += sorted(set((k.stem for k in Path('prompts').glob('*.txt'))), key=str.lower)
    prompts += ['None']
    return prompts

def get_available_characters():
    paths = (x for x in Path('characters').iterdir() if x.suffix in ('.json', '.yaml', '.yml'))
    return ['None'] + sorted(set((k.stem for k in paths if k.stem != "instruction-following")), key=str.lower)

def get_available_instruction_templates():
    path = "characters/instruction-following"
    paths = []
    if os.path.exists(path):
        paths = (x for x in Path(path).iterdir() if x.suffix in ('.json', '.yaml', '.yml'))
    return ['None'] + sorted(set((k.stem for k in paths)), key=str.lower)

def get_available_extensions():
    return sorted(set(map(lambda x : x.parts[1], Path('extensions').glob('*/script.py'))), key=str.lower)

def get_available_softprompts():
    return ['None'] + sorted(set((k.stem for k in Path('softprompts').glob('*.zip'))), key=str.lower)

def get_available_loras():
    return ['None'] + sorted([item.name for item in list(Path(shared.args.lora_dir).glob('*')) if not item.name.endswith(('.txt', '-np', '.pt', '.json'))], key=str.lower)

def unload_model():
    shared.model = shared.tokenizer = None
    clear_torch_cache()

def load_model_wrapper(selected_model):
    if selected_model != shared.model_name:
        shared.model_name = selected_model

        unload_model()
        if selected_model != '':
            shared.model, shared.tokenizer = load_model(shared.model_name)

    return selected_model

def load_lora_wrapper(selected_lora):
    add_lora_to_model(selected_lora)
    return selected_lora

def load_preset_values(preset_menu, state, return_dict=False):
    generate_params = {
        'do_sample': True,
        'temperature': 1,
        'top_p': 1,
        'typical_p': 1,
        'repetition_penalty': 1,
        'encoder_repetition_penalty': 1,
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
        state.update(generate_params)
        return state, *[generate_params[k] for k in ['do_sample', 'temperature', 'top_p', 'typical_p', 'repetition_penalty', 'encoder_repetition_penalty', 'top_k', 'min_length', 'no_repeat_ngram_size', 'num_beams', 'penalty_alpha', 'length_penalty', 'early_stopping']]

def upload_soft_prompt(file):
    with zipfile.ZipFile(io.BytesIO(file)) as zf:
        zf.extract('meta.json')
        j = json.loads(open('meta.json', 'r').read())
        name = j['name']
        Path('meta.json').unlink()

    with open(Path(f'softprompts/{name}.zip'), 'wb') as f:
        f.write(file)

    return name

def save_prompt(text):
    fname = f"{datetime.now().strftime('%Y-%m-%d-%H%M%S')}.txt"
    with open(Path(f'prompts/{fname}'), 'w', encoding='utf-8') as f:
        f.write(text)
    return f"Saved to prompts/{fname}"

def load_prompt(fname):
    if fname in ['None', '']:
        return ''
    else:
        with open(Path(f'prompts/{fname}.txt'), 'r', encoding='utf-8') as f:
            text = f.read()
            if text[-1] == '\n':
                text = text[:-1]
            return text

def create_prompt_menus():
    with gr.Row():
        with gr.Column():
            with gr.Row():
                shared.gradio['prompt_menu'] = gr.Dropdown(choices=get_available_prompts(), value='None', label='Prompt')
                ui.create_refresh_button(shared.gradio['prompt_menu'], lambda : None, lambda : {'choices': get_available_prompts()}, 'refresh-button')

        with gr.Column():
            with gr.Column():
                shared.gradio['save_prompt'] = gr.Button('Save prompt')
                shared.gradio['status'] = gr.Markdown('Ready')

    shared.gradio['prompt_menu'].change(load_prompt, [shared.gradio['prompt_menu']], [shared.gradio['textbox']], show_progress=False)
    shared.gradio['save_prompt'].click(save_prompt, [shared.gradio['textbox']], [shared.gradio['status']], show_progress=False)

def create_model_menus():
    with gr.Row():
        with gr.Column():
            with gr.Row():
                shared.gradio['model_menu'] = gr.Dropdown(choices=available_models, value=shared.model_name, label='Model')
                ui.create_refresh_button(shared.gradio['model_menu'], lambda : None, lambda : {'choices': get_available_models()}, 'refresh-button')
        with gr.Column():
            with gr.Row():
                shared.gradio['lora_menu'] = gr.Dropdown(choices=available_loras, value=shared.lora_name, label='LoRA')
                ui.create_refresh_button(shared.gradio['lora_menu'], lambda : None, lambda : {'choices': get_available_loras()}, 'refresh-button')

    shared.gradio['model_menu'].change(load_model_wrapper, shared.gradio['model_menu'], shared.gradio['model_menu'], show_progress=True)
    shared.gradio['lora_menu'].change(load_lora_wrapper, shared.gradio['lora_menu'], shared.gradio['lora_menu'], show_progress=True)

def create_settings_menus(default_preset):
    generate_params = load_preset_values(default_preset if not shared.args.flexgen else 'Naive', {}, return_dict=True)
    for k in ['max_new_tokens', 'seed', 'stop_at_newline', 'chat_prompt_size', 'chat_generation_attempts']:
        generate_params[k] = shared.settings[k]
    shared.gradio['generate_state'] = gr.State(generate_params)

    with gr.Row():
        with gr.Column():
            with gr.Row():
                shared.gradio['preset_menu'] = gr.Dropdown(choices=available_presets, value=default_preset if not shared.args.flexgen else 'Naive', label='Generation parameters preset')
                ui.create_refresh_button(shared.gradio['preset_menu'], lambda : None, lambda : {'choices': get_available_presets()}, 'refresh-button')
        with gr.Column():
            shared.gradio['seed'] = gr.Number(value=shared.settings['seed'], label='Seed (-1 for random)')

    with gr.Row():
        with gr.Column():
            with gr.Box():
                gr.Markdown('Custom generation parameters ([reference](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig))')
                with gr.Row():
                    with gr.Column():
                        shared.gradio['temperature'] = gr.Slider(0.01, 1.99, value=generate_params['temperature'], step=0.01, label='temperature')
                        shared.gradio['top_p'] = gr.Slider(0.0,1.0,value=generate_params['top_p'],step=0.01,label='top_p')
                        shared.gradio['top_k'] = gr.Slider(0,200,value=generate_params['top_k'],step=1,label='top_k')
                        shared.gradio['typical_p'] = gr.Slider(0.0,1.0,value=generate_params['typical_p'],step=0.01,label='typical_p')
                    with gr.Column():
                        shared.gradio['repetition_penalty'] = gr.Slider(1.0, 1.5, value=generate_params['repetition_penalty'],step=0.01,label='repetition_penalty')
                        shared.gradio['encoder_repetition_penalty'] = gr.Slider(0.8, 1.5, value=generate_params['encoder_repetition_penalty'],step=0.01,label='encoder_repetition_penalty')
                        shared.gradio['no_repeat_ngram_size'] = gr.Slider(0, 20, step=1, value=generate_params['no_repeat_ngram_size'], label='no_repeat_ngram_size')
                        shared.gradio['min_length'] = gr.Slider(0, 2000, step=1, value=generate_params['min_length'] if shared.args.no_stream else 0, label='min_length', interactive=shared.args.no_stream)
                shared.gradio['do_sample'] = gr.Checkbox(value=generate_params['do_sample'], label='do_sample')
        with gr.Column():
            with gr.Box():
                gr.Markdown('Contrastive search')
                shared.gradio['penalty_alpha'] = gr.Slider(0, 5, value=generate_params['penalty_alpha'], label='penalty_alpha')

            with gr.Box():
                gr.Markdown('Beam search (uses a lot of VRAM)')
                with gr.Row():
                    with gr.Column():
                        shared.gradio['num_beams'] = gr.Slider(1, 20, step=1, value=generate_params['num_beams'], label='num_beams')
                    with gr.Column():
                        shared.gradio['length_penalty'] = gr.Slider(-5, 5, value=generate_params['length_penalty'], label='length_penalty')
                shared.gradio['early_stopping'] = gr.Checkbox(value=generate_params['early_stopping'], label='early_stopping')


    with gr.Accordion('Soft prompt', open=False):
        with gr.Row():
            shared.gradio['softprompts_menu'] = gr.Dropdown(choices=available_softprompts, value='None', label='Soft prompt')
            ui.create_refresh_button(shared.gradio['softprompts_menu'], lambda : None, lambda : {'choices': get_available_softprompts()}, 'refresh-button')

        gr.Markdown('Upload a soft prompt (.zip format):')
        with gr.Row():
            shared.gradio['upload_softprompt'] = gr.File(type='binary', file_types=['.zip'])

    shared.gradio['preset_menu'].change(load_preset_values, [shared.gradio[k] for k in ['preset_menu', 'generate_state']], [shared.gradio[k] for k in ['generate_state', 'do_sample', 'temperature', 'top_p', 'typical_p', 'repetition_penalty', 'encoder_repetition_penalty', 'top_k', 'min_length', 'no_repeat_ngram_size', 'num_beams', 'penalty_alpha', 'length_penalty', 'early_stopping']])
    shared.gradio['softprompts_menu'].change(load_soft_prompt, shared.gradio['softprompts_menu'], shared.gradio['softprompts_menu'], show_progress=True)
    shared.gradio['upload_softprompt'].upload(upload_soft_prompt, shared.gradio['upload_softprompt'], shared.gradio['softprompts_menu'])

def set_interface_arguments(interface_mode, extensions, bool_active):
    modes = ["default", "notebook", "chat", "cai_chat"]
    cmd_list = vars(shared.args)
    bool_list = [k for k in cmd_list if type(cmd_list[k]) is bool and k not in modes]

    shared.args.extensions = extensions
    for k in modes[1:]:
        exec(f"shared.args.{k} = False")
    if interface_mode != "default":
        exec(f"shared.args.{interface_mode} = True")

    for k in bool_list:
        exec(f"shared.args.{k} = False")
    for k in bool_active:
        exec(f"shared.args.{k} = True")

    shared.need_restart = True

available_models = get_available_models()
available_presets = get_available_presets()
available_characters = get_available_characters()
available_softprompts = get_available_softprompts()
available_loras = get_available_loras()

# Default extensions
extensions_module.available_extensions = get_available_extensions()
if shared.is_chat():
    for extension in shared.settings['chat_default_extensions']:
        shared.args.extensions = shared.args.extensions or []
        if extension not in shared.args.extensions:
            shared.args.extensions.append(extension)
else:
    for extension in shared.settings['default_extensions']:
        shared.args.extensions = shared.args.extensions or []
        if extension not in shared.args.extensions:
            shared.args.extensions.append(extension)

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
if shared.args.lora:
    add_lora_to_model(shared.args.lora)

# Default UI settings
default_preset = shared.settings['presets'][next((k for k in shared.settings['presets'] if re.match(k.lower(), shared.model_name.lower())), 'default')]
if shared.lora_name != "None":
    default_text = load_prompt(shared.settings['lora_prompts'][next((k for k in shared.settings['lora_prompts'] if re.match(k.lower(), shared.lora_name.lower())), 'default')])
else:
    default_text = load_prompt(shared.settings['prompts'][next((k for k in shared.settings['prompts'] if re.match(k.lower(), shared.model_name.lower())), 'default')])
title ='Text generation web UI'

def create_interface():

    gen_events = []
    if shared.args.extensions is not None and len(shared.args.extensions) > 0:
        extensions_module.load_extensions()

    with gr.Blocks(css=ui.css if not shared.is_chat() else ui.css+ui.chat_css, analytics_enabled=False, title=title) as shared.gradio['interface']:
        if shared.is_chat():
            shared.gradio['Chat input'] = gr.State()
            with gr.Tab("Text generation", elem_id="main"):
                shared.gradio['display'] = gr.HTML(value=chat_html_wrapper(shared.history['visible'], shared.settings['name1'], shared.settings['name2'], 'cai-chat'))
                shared.gradio['textbox'] = gr.Textbox(label='Input')
                with gr.Row():
                    shared.gradio['Generate'] = gr.Button('Generate')
                    shared.gradio['Stop'] = gr.Button('Stop', elem_id="stop")
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

                shared.gradio["Chat mode"] = gr.Radio(choices=["cai-chat", "chat", "instruct"], value="cai-chat", label="Mode")
                shared.gradio["Instruction templates"] = gr.Dropdown(choices=get_available_instruction_templates(), label="Instruction template", value="None", visible=False)

            with gr.Tab("Character", elem_id="chat-settings"):
                with gr.Row():
                    with gr.Column(scale=8):
                        shared.gradio['name1'] = gr.Textbox(value=shared.settings['name1'], lines=1, label='Your name')
                        shared.gradio['name2'] = gr.Textbox(value=shared.settings['name2'], lines=1, label='Character\'s name')
                        shared.gradio['greeting'] = gr.Textbox(value=shared.settings['greeting'], lines=4, label='Greeting')
                        shared.gradio['context'] = gr.Textbox(value=shared.settings['context'], lines=4, label='Context')
                        shared.gradio['end_of_turn'] = gr.Textbox(value=shared.settings["end_of_turn"], lines=1, label='End of turn string')
                    with gr.Column(scale=1):
                        shared.gradio['character_picture'] = gr.Image(label='Character picture', type="pil")
                        shared.gradio['your_picture'] = gr.Image(label='Your picture', type="pil", value=Image.open(Path("cache/pfp_me.png")) if Path("cache/pfp_me.png").exists() else None)
                with gr.Row():
                    shared.gradio['character_menu'] = gr.Dropdown(choices=available_characters, value='None', label='Character', elem_id='character-menu')
                    ui.create_refresh_button(shared.gradio['character_menu'], lambda : None, lambda : {'choices': get_available_characters()}, 'refresh-button')

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
                        gr.Markdown("# JSON format")
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown('1. Select the JSON file')
                                shared.gradio['upload_json'] = gr.File(type='binary', file_types=['.json'])
                            with gr.Column():
                                gr.Markdown('2. Select your character\'s profile picture (optional)')
                                shared.gradio['upload_img_bot'] = gr.File(type='binary', file_types=['image'])
                        shared.gradio['Upload character'] = gr.Button(value='Submit')

                        gr.Markdown("# TavernAI PNG format")
                        shared.gradio['upload_img_tavern'] = gr.File(type='binary', file_types=['image'])

            with gr.Tab("Parameters", elem_id="parameters"):
                with gr.Box():
                    gr.Markdown("Chat parameters")
                    with gr.Row():
                        with gr.Column():
                            shared.gradio['max_new_tokens'] = gr.Slider(minimum=shared.settings['max_new_tokens_min'], maximum=shared.settings['max_new_tokens_max'], step=1, label='max_new_tokens', value=shared.settings['max_new_tokens'])
                            shared.gradio['chat_prompt_size_slider'] = gr.Slider(minimum=shared.settings['chat_prompt_size_min'], maximum=shared.settings['chat_prompt_size_max'], step=1, label='Maximum prompt size in tokens', value=shared.settings['chat_prompt_size'])
                        with gr.Column():
                            shared.gradio['chat_generation_attempts'] = gr.Slider(minimum=shared.settings['chat_generation_attempts_min'], maximum=shared.settings['chat_generation_attempts_max'], value=shared.settings['chat_generation_attempts'], step=1, label='Generation attempts (for longer replies)')
                            shared.gradio['stop_at_newline'] = gr.Checkbox(value=shared.settings['stop_at_newline'], label='Stop generating at new line character?')

                create_settings_menus(default_preset)

            shared.input_params = [shared.gradio[k] for k in ['Chat input', 'generate_state', 'name1', 'name2', 'context', 'Chat mode', 'end_of_turn']]

            def set_chat_input(textbox):
                return textbox, ""

            gen_events.append(shared.gradio['Generate'].click(set_chat_input, shared.gradio['textbox'], [shared.gradio['Chat input'], shared.gradio['textbox']], show_progress=False))
            gen_events.append(shared.gradio['Generate'].click(chat.cai_chatbot_wrapper, shared.input_params, shared.gradio['display'], show_progress=shared.args.no_stream))
            gen_events.append(shared.gradio['textbox'].submit(set_chat_input, shared.gradio['textbox'], [shared.gradio['Chat input'], shared.gradio['textbox']], show_progress=False))
            gen_events.append(shared.gradio['textbox'].submit(chat.cai_chatbot_wrapper, shared.input_params, shared.gradio['display'], show_progress=shared.args.no_stream))
            gen_events.append(shared.gradio['Regenerate'].click(chat.regenerate_wrapper, shared.input_params, shared.gradio['display'], show_progress=shared.args.no_stream))
            gen_events.append(shared.gradio['Impersonate'].click(chat.impersonate_wrapper, shared.input_params, shared.gradio['textbox'], show_progress=shared.args.no_stream))
            shared.gradio['Stop'].click(stop_everything_event, [], [], queue=False, cancels=gen_events if shared.args.no_stream else None)

            shared.gradio['Copy last reply'].click(chat.send_last_reply_to_input, [], shared.gradio['textbox'], show_progress=shared.args.no_stream)
            shared.gradio['Replace last reply'].click(chat.replace_last_reply, [shared.gradio[k] for k in ['textbox', 'name1', 'name2', 'Chat mode']], shared.gradio['display'], show_progress=shared.args.no_stream)

            # Clear history with confirmation
            clear_arr = [shared.gradio[k] for k in ['Clear history-confirm', 'Clear history', 'Clear history-cancel']]
            shared.gradio['Clear history'].click(lambda :[gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)], None, clear_arr)
            shared.gradio['Clear history-confirm'].click(lambda :[gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)], None, clear_arr)
            shared.gradio['Clear history-confirm'].click(chat.clear_chat_log, [shared.gradio[k] for k in ['name1', 'name2', 'greeting', 'Chat mode']], shared.gradio['display'])
            shared.gradio['Clear history-cancel'].click(lambda :[gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)], None, clear_arr)
            shared.gradio['Chat mode'].change(lambda x : gr.update(visible= x=='instruct'), shared.gradio['Chat mode'], shared.gradio['Instruction templates'])

            shared.gradio['Remove last'].click(chat.remove_last_message, [shared.gradio[k] for k in ['name1', 'name2', 'Chat mode']], [shared.gradio['display'], shared.gradio['textbox']], show_progress=False)
            shared.gradio['download_button'].click(chat.save_history, inputs=[], outputs=[shared.gradio['download']])
            shared.gradio['Upload character'].click(chat.upload_character, [shared.gradio['upload_json'], shared.gradio['upload_img_bot']], [shared.gradio['character_menu']])

            # Clearing stuff and saving the history
            for i in ['Generate', 'Regenerate', 'Replace last reply']:
                shared.gradio[i].click(lambda x: '', shared.gradio['textbox'], shared.gradio['textbox'], show_progress=False)
                shared.gradio[i].click(lambda : chat.save_history(timestamp=False), [], [], show_progress=False)
            shared.gradio['Clear history-confirm'].click(lambda : chat.save_history(timestamp=False), [], [], show_progress=False)
            shared.gradio['textbox'].submit(lambda x: '', shared.gradio['textbox'], shared.gradio['textbox'], show_progress=False)
            shared.gradio['textbox'].submit(lambda : chat.save_history(timestamp=False), [], [], show_progress=False)

            shared.gradio['character_menu'].change(chat.load_character, [shared.gradio[k] for k in ['character_menu', 'name1', 'name2', 'Chat mode']], [shared.gradio[k] for k in ['name1', 'name2', 'character_picture', 'greeting', 'context', 'end_of_turn', 'display']])
            shared.gradio['Instruction templates'].change(lambda character, name1, name2, mode: chat.load_character(character, name1, name2, mode), [shared.gradio[k] for k in ['Instruction templates', 'name1', 'name2', 'Chat mode']], [shared.gradio[k] for k in ['name1', 'name2', 'character_picture', 'greeting', 'context', 'end_of_turn', 'display']])
            shared.gradio['upload_chat_history'].upload(chat.load_history, [shared.gradio[k] for k in ['upload_chat_history', 'name1', 'name2']], [])
            shared.gradio['upload_img_tavern'].upload(chat.upload_tavern_character, [shared.gradio['upload_img_tavern'], shared.gradio['name1'], shared.gradio['name2']], [shared.gradio['character_menu']])
            shared.gradio['your_picture'].change(chat.upload_your_profile_picture, [shared.gradio[k] for k in ['your_picture', 'name1', 'name2', 'Chat mode']], shared.gradio['display'])

            reload_inputs = [shared.gradio[k] for k in ['name1', 'name2', 'Chat mode']]
            shared.gradio['upload_chat_history'].upload(chat.redraw_html, reload_inputs, [shared.gradio['display']])
            shared.gradio['Stop'].click(chat.redraw_html, reload_inputs, [shared.gradio['display']])
            shared.gradio['Instruction templates'].change(chat.redraw_html, reload_inputs, [shared.gradio['display']])
            shared.gradio['Chat mode'].change(chat.redraw_html, reload_inputs, [shared.gradio['display']])

            shared.gradio['interface'].load(None, None, None, _js=f"() => {{{ui.main_js+ui.chat_js}}}")
            shared.gradio['interface'].load(lambda : chat.load_default_history(shared.settings['name1'], shared.settings['name2']), None, None)
            shared.gradio['interface'].load(chat.redraw_html, reload_inputs, [shared.gradio['display']], show_progress=True)

        elif shared.args.notebook:
            with gr.Tab("Text generation", elem_id="main"):
                with gr.Row():
                    with gr.Column(scale=4):
                        with gr.Tab('Raw'):
                            shared.gradio['textbox'] = gr.Textbox(value=default_text, elem_id="textbox", lines=27)
                        with gr.Tab('Markdown'):
                            shared.gradio['markdown'] = gr.Markdown()
                        with gr.Tab('HTML'):
                            shared.gradio['html'] = gr.HTML()

                        with gr.Row():
                            with gr.Column():
                                with gr.Row():
                                    shared.gradio['Generate'] = gr.Button('Generate')
                                    shared.gradio['Stop'] = gr.Button('Stop')
                            with gr.Column():
                                pass

                    with gr.Column(scale=1):
                        gr.HTML('<div style="padding-bottom: 13px"></div>')
                        shared.gradio['max_new_tokens'] = gr.Slider(minimum=shared.settings['max_new_tokens_min'], maximum=shared.settings['max_new_tokens_max'], step=1, label='max_new_tokens', value=shared.settings['max_new_tokens'])

                        create_prompt_menus()

            with gr.Tab("Parameters", elem_id="parameters"):
                create_settings_menus(default_preset)

            shared.input_params = [shared.gradio[k] for k in ['textbox', 'generate_state']]
            output_params = [shared.gradio[k] for k in ['textbox', 'markdown', 'html']]
            gen_events.append(shared.gradio['Generate'].click(generate_reply, shared.input_params, output_params, show_progress=shared.args.no_stream))
            gen_events.append(shared.gradio['textbox'].submit(generate_reply, shared.input_params, output_params, show_progress=shared.args.no_stream))
            shared.gradio['Stop'].click(stop_everything_event, [], [], queue=False, cancels=gen_events if shared.args.no_stream else None)
            shared.gradio['interface'].load(None, None, None, _js=f"() => {{{ui.main_js}}}")

        else:
            with gr.Tab("Text generation", elem_id="main"):
                with gr.Row():
                    with gr.Column():
                        shared.gradio['textbox'] = gr.Textbox(value=default_text, lines=21, label='Input')
                        shared.gradio['max_new_tokens'] = gr.Slider(minimum=shared.settings['max_new_tokens_min'], maximum=shared.settings['max_new_tokens_max'], step=1, label='max_new_tokens', value=shared.settings['max_new_tokens'])
                        shared.gradio['Generate'] = gr.Button('Generate')
                        with gr.Row():
                            with gr.Column():
                                shared.gradio['Continue'] = gr.Button('Continue')
                            with gr.Column():
                                shared.gradio['Stop'] = gr.Button('Stop')

                        create_prompt_menus()

                    with gr.Column():
                        with gr.Tab('Raw'):
                            shared.gradio['output_textbox'] = gr.Textbox(lines=27, label='Output')
                        with gr.Tab('Markdown'):
                            shared.gradio['markdown'] = gr.Markdown()
                        with gr.Tab('HTML'):
                            shared.gradio['html'] = gr.HTML()

            with gr.Tab("Parameters", elem_id="parameters"):
                create_settings_menus(default_preset)

            shared.input_params = [shared.gradio[k] for k in ['textbox', 'generate_state']]
            output_params = [shared.gradio[k] for k in ['output_textbox', 'markdown', 'html']]
            gen_events.append(shared.gradio['Generate'].click(generate_reply, shared.input_params, output_params, show_progress=shared.args.no_stream))
            gen_events.append(shared.gradio['textbox'].submit(generate_reply, shared.input_params, output_params, show_progress=shared.args.no_stream))
            gen_events.append(shared.gradio['Continue'].click(generate_reply, [shared.gradio['output_textbox']] + shared.input_params[1:], output_params, show_progress=shared.args.no_stream))
            shared.gradio['Stop'].click(stop_everything_event, [], [], queue=False, cancels=gen_events if shared.args.no_stream else None)
            shared.gradio['interface'].load(None, None, None, _js=f"() => {{{ui.main_js}}}")

        with gr.Tab("Model", elem_id="model-tab"):
            create_model_menus()

        with gr.Tab("Training", elem_id="training-tab"):
            training.create_train_interface()

        with gr.Tab("Interface mode", elem_id="interface-mode"):
            modes = ["default", "notebook", "chat", "cai_chat"]
            current_mode = "default"
            for mode in modes[1:]:
                if eval(f"shared.args.{mode}"):
                    current_mode = mode
                    break
            cmd_list = vars(shared.args)
            bool_list = [k for k in cmd_list if type(cmd_list[k]) is bool and k not in modes]
            bool_active = [k for k in bool_list if vars(shared.args)[k]]

            gr.Markdown("*Experimental*")
            shared.gradio['interface_modes_menu'] = gr.Dropdown(choices=modes, value=current_mode, label="Mode")
            shared.gradio['extensions_menu'] = gr.CheckboxGroup(choices=get_available_extensions(), value=shared.args.extensions, label="Available extensions")
            shared.gradio['bool_menu'] = gr.CheckboxGroup(choices=bool_list, value=bool_active, label="Boolean command-line flags")
            shared.gradio['reset_interface'] = gr.Button("Apply and restart the interface", type="primary")

            shared.gradio['reset_interface'].click(set_interface_arguments, [shared.gradio[k] for k in ['interface_modes_menu', 'extensions_menu', 'bool_menu']], None)
            shared.gradio['reset_interface'].click(lambda : None, None, None, _js='() => {document.body.innerHTML=\'<h1 style="font-family:monospace;margin-top:20%;color:lightgray;text-align:center;">Reloading...</h1>\'; setTimeout(function(){location.reload()},2500); return []}')

        if shared.args.extensions is not None:
            extensions_module.create_extensions_block()

        def change_dict_value(d, key, value):
            d[key] = value
            return d

        for k in ['do_sample', 'temperature', 'top_p', 'typical_p', 'repetition_penalty', 'encoder_repetition_penalty', 'top_k', 'min_length', 'no_repeat_ngram_size', 'num_beams', 'penalty_alpha', 'length_penalty', 'early_stopping', 'max_new_tokens', 'seed', 'stop_at_newline', 'chat_prompt_size_slider', 'chat_generation_attempts']:
            if k not in shared.gradio:
                continue
            if type(shared.gradio[k]) in [gr.Checkbox, gr.Number]:
                shared.gradio[k].change(lambda state, value, copy=k: change_dict_value(state, copy, value), inputs=[shared.gradio['generate_state'], shared.gradio[k]], outputs=shared.gradio['generate_state'])
            else:
                shared.gradio[k].release(lambda state, value, copy=k: change_dict_value(state, copy, value), inputs=[shared.gradio['generate_state'], shared.gradio[k]], outputs=shared.gradio['generate_state'])

        if not shared.is_chat():
            api.create_apis()

    # Authentication
    auth = None
    if shared.args.gradio_auth_path is not None:
        gradio_auth_creds = []
        with open(shared.args.gradio_auth_path, 'r', encoding="utf8") as file:
            for line in file.readlines():
                gradio_auth_creds += [x.strip() for x in line.split(',') if x.strip()]
        auth = [tuple(cred.split(':')) for cred in gradio_auth_creds]

    # Launch the interface
    shared.gradio['interface'].queue()
    if shared.args.listen:
        shared.gradio['interface'].launch(prevent_thread_lock=True, share=shared.args.share, server_name='0.0.0.0', server_port=shared.args.listen_port, inbrowser=shared.args.auto_launch, auth=auth)
    else:
        shared.gradio['interface'].launch(prevent_thread_lock=True, share=shared.args.share, server_port=shared.args.listen_port, inbrowser=shared.args.auto_launch, auth=auth)

create_interface()

while True:
    time.sleep(0.5)
    if shared.need_restart:
        shared.need_restart = False
        shared.gradio['interface'].close()
        create_interface()
