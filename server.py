import gc
import io
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import transformers
from PIL import Image
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

import modules.chat as chat
import modules.extensions as extensions_module
import modules.shared as shared
from modules.extensions import extension_state
from modules.extensions import load_extensions
from modules.extensions import update_extensions_parameters
from modules.html_generator import *
from modules.prompt import generate_reply
from modules.ui import *

transformers.logging.set_verbosity_error()

if (shared.args.chat or shared.args.cai_chat) and not shared.args.no_stream:
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
    'chat_prompt_size': 2048,
    'chat_prompt_size_min': 0,
    'chat_prompt_size_max': 2048,
    'preset_pygmalion': 'Pygmalion',
    'name1_pygmalion': 'You',
    'name2_pygmalion': 'Kawaii',
    'context_pygmalion': "Kawaii's persona: Kawaii is a cheerful person who loves to make others smile. She is an optimist who loves to spread happiness and positivity wherever she goes.\n<START>",
    'stop_at_newline_pygmalion': False,
}

if shared.args.settings is not None and Path(shared.args.settings).exists():
    new_settings = json.loads(open(Path(shared.args.settings), 'r').read())
    for item in new_settings:
        settings[item] = new_settings[item]

if shared.args.flexgen:
    from flexgen.flex_opt import (Policy, OptLM, TorchDevice, TorchDisk, TorchMixedDevice, CompressionConfig, Env, Task, get_opt_config)

if shared.args.deepspeed:
    import deepspeed
    from transformers.deepspeed import HfDeepSpeedConfig, is_deepspeed_zero3_enabled
    from modules.deepspeed_parameters import generate_ds_config

    # Distributed setup
    local_rank = shared.args.local_rank if shared.args.local_rank is not None else int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()
    ds_config = generate_ds_config(shared.args.bf16, 1 * world_size, shared.args.nvme_offload_dir)
    dschf = HfDeepSpeedConfig(ds_config) # Keep this object alive for the Transformers integration

if shared.args.picture and (shared.args.cai_chat or shared.args.chat):
    import modules.bot_picture as bot_picture

def load_model(model_name):
    print(f"Loading {model_name}...")
    t0 = time.time()

    # Default settings
    if not (shared.args.cpu or shared.args.load_in_8bit or shared.args.auto_devices or shared.args.disk or shared.args.gpu_memory is not None or shared.args.cpu_memory is not None or shared.args.deepspeed or shared.args.flexgen):
        if any(size in shared.model_name.lower() for size in ('13b', '20b', '30b')):
            model = AutoModelForCausalLM.from_pretrained(Path(f"models/{shared.model_name}"), device_map='auto', load_in_8bit=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(Path(f"models/{shared.model_name}"), low_cpu_mem_usage=True, torch_dtype=torch.bfloat16 if shared.args.bf16 else torch.float16).cuda()

    # FlexGen
    elif shared.args.flexgen:
        gpu = TorchDevice("cuda:0")
        cpu = TorchDevice("cpu")
        disk = TorchDisk(shared.args.disk_cache_dir)
        env = Env(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))

        # Offloading policy
        policy = Policy(1, 1,
                        shared.args.percent[0], shared.args.percent[1],
                        shared.args.percent[2], shared.args.percent[3],
                        shared.args.percent[4], shared.args.percent[5],
                        overlap=True, sep_layer=True, pin_weight=True,
                        cpu_cache_compute=False, attn_sparsity=1.0,
                        compress_weight=shared.args.compress_weight,
                        comp_weight_config=CompressionConfig(
                            num_bits=4, group_size=64,
                            group_dim=0, symmetric=False),
                        compress_cache=False,
                        comp_cache_config=CompressionConfig(
                            num_bits=4, group_size=64,
                            group_dim=2, symmetric=False))

        opt_config = get_opt_config(f"facebook/{shared.model_name}")
        model = OptLM(opt_config, env, "models", policy)
        model.init_all_weights()

    # DeepSpeed ZeRO-3
    elif shared.args.deepspeed:
        model = AutoModelForCausalLM.from_pretrained(Path(f"models/{shared.model_name}"), torch_dtype=torch.bfloat16 if shared.args.bf16 else torch.float16)
        model = deepspeed.initialize(model=model, config_params=ds_config, model_parameters=None, optimizer=None, lr_scheduler=None)[0]
        model.module.eval() # Inference
        print(f"DeepSpeed ZeRO-3 is enabled: {is_deepspeed_zero3_enabled()}")

    # Custom
    else:
        command = "AutoModelForCausalLM.from_pretrained"
        params = ["low_cpu_mem_usage=True"]
        if not shared.args.cpu and not torch.cuda.is_available():
            print("Warning: no GPU has been detected.\nFalling back to CPU mode.\n")
            shared.args.cpu = True

        if shared.args.cpu:
            params.append("low_cpu_mem_usage=True")
            params.append("torch_dtype=torch.float32")
        else:
            params.append("device_map='auto'")
            params.append("load_in_8bit=True" if shared.args.load_in_8bit else "torch_dtype=torch.bfloat16" if shared.args.bf16 else "torch_dtype=torch.float16")

            if shared.args.gpu_memory:
                params.append(f"max_memory={{0: '{shared.args.gpu_memory or '99'}GiB', 'cpu': '{shared.args.cpu_memory or '99'}GiB'}}")
            elif not shared.args.load_in_8bit:
                total_mem = (torch.cuda.get_device_properties(0).total_memory/(1024*1024))
                suggestion = round((total_mem-1000)/1000)*1000
                if total_mem-suggestion < 800:
                    suggestion -= 1000
                suggestion = int(round(suggestion/1000))
                print(f"\033[1;32;1mAuto-assiging --gpu-memory {suggestion} for your GPU to try to prevent out-of-memory errors.\nYou can manually set other values.\033[0;37;0m")
                params.append(f"max_memory={{0: '{suggestion}GiB', 'cpu': '{shared.args.cpu_memory or '99'}GiB'}}")
            if shared.args.disk:
                params.append(f"offload_folder='{shared.args.disk_cache_dir}'")

        command = f"{command}(Path(f'models/{shared.model_name}'), {', '.join(set(params))})"
        model = eval(command)

    # Loading the tokenizer
    if shared.model_name.lower().startswith(('gpt4chan', 'gpt-4chan', '4chan')) and Path(f"models/gpt-j-6B/").exists():
        tokenizer = AutoTokenizer.from_pretrained(Path("models/gpt-j-6B/"))
    else:
        tokenizer = AutoTokenizer.from_pretrained(Path(f"models/{shared.model_name}/"))
    tokenizer.truncation_side = 'left'

    print(f"Loaded the model in {(time.time()-t0):.2f} seconds.")
    return model, tokenizer

def load_soft_prompt(name):
    if name == 'None':
        shared.soft_prompt = False
        shared.soft_prompt_tensor = None
    else:
        with zipfile.ZipFile(Path(f'softprompts/{name}.zip')) as zf:
            zf.extract('tensor.npy')
            zf.extract('meta.json')
            j = json.loads(open('meta.json', 'r').read())
            print(f"\nLoading the softprompt \"{name}\".")
            for field in j:
                if field != 'name':
                    if type(j[field]) is list:
                        print(f"{field}: {', '.join(j[field])}")
                    else:
                        print(f"{field}: {j[field]}")
            print()
            tensor = np.load('tensor.npy')
            Path('tensor.npy').unlink()
            Path('meta.json').unlink()
        tensor = torch.Tensor(tensor).to(device=shared.model.device, dtype=shared.model.dtype)
        tensor = torch.reshape(tensor, (1, tensor.shape[0], tensor.shape[1]))

        shared.soft_prompt = True
        shared.soft_prompt_tensor = tensor

    return name

def upload_soft_prompt(file):
    with zipfile.ZipFile(io.BytesIO(file)) as zf:
        zf.extract('meta.json')
        j = json.loads(open('meta.json', 'r').read())
        name = j['name']
        Path('meta.json').unlink()

    with open(Path(f'softprompts/{name}.zip'), 'wb') as f:
        f.write(file)

    return name

def load_model_wrapper(selected_model):
    if selected_model != shared.model_name:
        shared.model_name = selected_model
        model = shared.tokenizer = None
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

def create_extensions_block():
    extensions_ui_elements = []
    default_values = []
    if not (shared.args.chat or shared.args.cai_chat):
        gr.Markdown('## Extensions parameters')
    for ext in sorted(extension_state, key=lambda x : extension_state[x][1]):
        if extension_state[ext][0] == True:
            params = extensions_module.get_params(ext)
            for param in params:
                _id = f"{ext}-{param}"
                default_value = settings[_id] if _id in settings else params[param]
                default_values.append(default_value)
                if type(params[param]) == str:
                    extensions_ui_elements.append(gr.Textbox(value=default_value, label=f"{ext}-{param}"))
                elif type(params[param]) in [int, float]:
                    extensions_ui_elements.append(gr.Number(value=default_value, label=f"{ext}-{param}"))
                elif type(params[param]) == bool:
                    extensions_ui_elements.append(gr.Checkbox(value=default_value, label=f"{ext}-{param}"))

    update_extensions_parameters(*default_values)
    btn_extensions = gr.Button("Apply")
    btn_extensions.click(update_extensions_parameters, [*extensions_ui_elements], [])

def create_settings_menus():
    generate_params = load_preset_values(settings[f'preset{suffix}'] if not shared.args.flexgen else 'Naive', return_dict=True)

    with gr.Row():
        with gr.Column():
            with gr.Row():
                model_menu = gr.Dropdown(choices=available_models, value=shared.model_name, label='Model')
                create_refresh_button(model_menu, lambda : None, lambda : {"choices": get_available_models()}, "refresh-button")
        with gr.Column():
            with gr.Row():
                preset_menu = gr.Dropdown(choices=available_presets, value=settings[f'preset{suffix}'] if not shared.args.flexgen else 'Naive', label='Generation parameters preset')
                create_refresh_button(preset_menu, lambda : None, lambda : {"choices": get_available_presets()}, "refresh-button")

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
            create_refresh_button(softprompts_menu, lambda : None, lambda : {"choices": get_available_softprompts()}, "refresh-button")

        gr.Markdown('Upload a soft prompt (.zip format):')
        with gr.Row():
            upload_softprompt = gr.File(type='binary', file_types=[".zip"])

    model_menu.change(load_model_wrapper, [model_menu], [model_menu], show_progress=True)
    preset_menu.change(load_preset_values, [preset_menu], [do_sample, temperature, top_p, typical_p, repetition_penalty, top_k, min_length, no_repeat_ngram_size, num_beams, penalty_alpha, length_penalty, early_stopping])
    softprompts_menu.change(load_soft_prompt, [softprompts_menu], [softprompts_menu], show_progress=True)
    upload_softprompt.upload(upload_soft_prompt, [upload_softprompt], [softprompts_menu])
    return preset_menu, do_sample, temperature, top_p, typical_p, repetition_penalty, top_k, min_length, no_repeat_ngram_size, num_beams, penalty_alpha, length_penalty, early_stopping

# Global variables
available_models = get_available_models()
available_presets = get_available_presets()
available_characters = get_available_characters()
extensions_module.available_extensions = get_available_extensions()
available_softprompts = get_available_softprompts()
if shared.args.extensions is not None:
    load_extensions()

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
        for i,model in enumerate(available_models):
            print(f"{i+1}. {model}")
        print(f"\nWhich one do you want to load? 1-{len(available_models)}\n")
        i = int(input())-1
        print()
    shared.model_name = available_models[i]
shared.model, shared.tokenizer = load_model(shared.model_name)
loaded_preset = None

# UI settings
if shared.model_name.lower().startswith(('gpt4chan', 'gpt-4chan', '4chan')):
    default_text = settings['prompt_gpt4chan']
elif re.match('(rosey|chip|joi)_.*_instruct.*', shared.model_name.lower()) is not None:
    default_text = 'User: \n'
else:
    default_text = settings['prompt']
description = f"\n\n# Text generation lab\nGenerate text using Large Language Models.\n"

suffix = '_pygmalion' if 'pygmalion' in shared.model_name.lower() else ''
buttons = {}
gen_events = []

if shared.args.chat or shared.args.cai_chat:

    if Path(f'logs/persistent.json').exists():
        chat.load_history(open(Path(f'logs/persistent.json'), 'rb').read(), settings[f'name1{suffix}'], settings[f'name2{suffix}'])

    with gr.Blocks(css=css+chat_css, analytics_enabled=False) as interface:
        if shared.args.cai_chat:
            display = gr.HTML(value=generate_chat_html(chat.history['visible'], settings[f'name1{suffix}'], settings[f'name2{suffix}'], chat.character))
        else:
            display = gr.Chatbot(value=chat.history['visible'])
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
            name1 = gr.Textbox(value=settings[f'name1{suffix}'], lines=1, label='Your name')
            name2 = gr.Textbox(value=settings[f'name2{suffix}'], lines=1, label='Bot\'s name')
            context = gr.Textbox(value=settings[f'context{suffix}'], lines=2, label='Context')
            with gr.Row():
                character_menu = gr.Dropdown(choices=available_characters, value="None", label='Character')
                create_refresh_button(character_menu, lambda : None, lambda : {"choices": get_available_characters()}, "refresh-button")

            with gr.Row():
                check = gr.Checkbox(value=settings[f'stop_at_newline{suffix}'], label='Stop generating at new line character?')
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
                    max_new_tokens = gr.Slider(minimum=settings['max_new_tokens_min'], maximum=settings['max_new_tokens_max'], step=1, label='max_new_tokens', value=settings['max_new_tokens'])
                with gr.Column():
                    chat_prompt_size_slider = gr.Slider(minimum=settings['chat_prompt_size_min'], maximum=settings['chat_prompt_size_max'], step=1, label='Maximum prompt size in tokens', value=settings['chat_prompt_size'])

            preset_menu, do_sample, temperature, top_p, typical_p, repetition_penalty, top_k, min_length, no_repeat_ngram_size, num_beams, penalty_alpha, length_penalty, early_stopping = create_settings_menus()

        if shared.args.extensions is not None:
            with gr.Tab("Extensions"):
                create_extensions_block()

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
        buttons["Clear history"].click(chat.clear_chat_log, [character_menu, name1, name2], display)
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
        if shared.args.cai_chat:
            upload_chat_history.upload(chat.redraw_html, [name1, name2], [display])
            upload_img_me.upload(chat.redraw_html, [name1, name2], [display])
        else:
            upload_chat_history.upload(lambda : chat.history['visible'], [], [display])
            upload_img_me.upload(lambda : chat.history['visible'], [], [display])

elif shared.args.notebook:
    with gr.Blocks(css=css, analytics_enabled=False) as interface:
        gr.Markdown(description)
        with gr.Tab('Raw'):
            textbox = gr.Textbox(value=default_text, lines=23)
        with gr.Tab('Markdown'):
            markdown = gr.Markdown()
        with gr.Tab('HTML'):
            html = gr.HTML()

        buttons["Generate"] = gr.Button("Generate")
        buttons["Stop"] = gr.Button("Stop")

        max_new_tokens = gr.Slider(minimum=settings['max_new_tokens_min'], maximum=settings['max_new_tokens_max'], step=1, label='max_new_tokens', value=settings['max_new_tokens'])

        preset_menu, do_sample, temperature, top_p, typical_p, repetition_penalty, top_k, min_length, no_repeat_ngram_size, num_beams, penalty_alpha, length_penalty, early_stopping = create_settings_menus()

        if shared.args.extensions is not None:
            create_extensions_block()

        gen_events.append(buttons["Generate"].click(generate_reply, [textbox, max_new_tokens, do_sample, max_new_tokens, temperature, top_p, typical_p, repetition_penalty, top_k, min_length, no_repeat_ngram_size, num_beams, penalty_alpha, length_penalty, early_stopping], [textbox, markdown, html], show_progress=shared.args.no_stream, api_name="textgen"))
        gen_events.append(textbox.submit(generate_reply, [textbox, max_new_tokens, do_sample, max_new_tokens, temperature, top_p, typical_p, repetition_penalty, top_k, min_length, no_repeat_ngram_size, num_beams, penalty_alpha, length_penalty, early_stopping], [textbox, markdown, html], show_progress=shared.args.no_stream))
        buttons["Stop"].click(None, None, None, cancels=gen_events)

else:
    with gr.Blocks(css=css, analytics_enabled=False) as interface:
        gr.Markdown(description)
        with gr.Row():
            with gr.Column():
                textbox = gr.Textbox(value=default_text, lines=15, label='Input')
                max_new_tokens = gr.Slider(minimum=settings['max_new_tokens_min'], maximum=settings['max_new_tokens_max'], step=1, label='max_new_tokens', value=settings['max_new_tokens'])
                buttons["Generate"] = gr.Button("Generate")
                with gr.Row():
                    with gr.Column():
                        buttons["Continue"] = gr.Button("Continue")
                    with gr.Column():
                        buttons["Stop"] = gr.Button("Stop")

                preset_menu, do_sample, temperature, top_p, typical_p, repetition_penalty, top_k, min_length, no_repeat_ngram_size, num_beams, penalty_alpha, length_penalty, early_stopping = create_settings_menus()
                if shared.args.extensions is not None:
                    create_extensions_block()

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
