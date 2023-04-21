import os
import requests
import warnings

os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
os.environ['BITSANDBYTES_NOWELCOME'] = '1'
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

# This is a hack to prevent Gradio from phoning home when it gets imported
def my_get(url, **kwargs):
    print('Gradio HTTP request redirected to localhost :)')
    kwargs.setdefault('allow_redirects', True)
    return requests.api.request('get', 'http://127.0.0.1/', **kwargs)

original_get = requests.get
requests.get = my_get
import gradio as gr
requests.get = original_get

# This fixes LaTeX rendering on some systems
import matplotlib
matplotlib.use('Agg')

import importlib
import io
import json
import math
import os
import re
import sys
import time
import traceback
import zipfile
from datetime import datetime
from pathlib import Path

import psutil
import torch
import yaml
from PIL import Image

import modules.extensions as extensions_module
from modules import api, chat, shared, training, ui
from modules.html_generator import chat_html_wrapper
from modules.LoRA import add_lora_to_model
from modules.models import load_model, load_soft_prompt, unload_model
from modules.text_generation import generate_reply, stop_everything_event

# moderation imports
import lancedb
from sentence_transformers import SentenceTransformer


def get_available_models():
    if shared.args.flexgen:
        return sorted([re.sub('-np$', '', item.name) for item in list(Path(f'{shared.args.model_dir}/').glob('*')) if item.name.endswith('-np')], key=str.lower)
    else:
        return sorted([re.sub('.pth$', '', item.name) for item in list(Path(f'{shared.args.model_dir}/').glob('*')) if not item.name.endswith(('.txt', '-np', '.pt', '.json', '.yaml'))], key=str.lower)


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
    return sorted(set(map(lambda x: x.parts[1], Path('extensions').glob('*/script.py'))), key=str.lower)


def get_available_softprompts():
    return ['None'] + sorted(set((k.stem for k in Path('softprompts').glob('*.zip'))), key=str.lower)


def get_available_loras():
    return sorted([item.name for item in list(Path(shared.args.lora_dir).glob('*')) if not item.name.endswith(('.txt', '-np', '.pt', '.json'))], key=str.lower)


def load_model_wrapper(selected_model):
    try:
        yield f"Loading {selected_model}..."
        shared.model_name = selected_model
        unload_model()
        if selected_model != '':
            shared.model, shared.tokenizer = load_model(shared.model_name)

        yield f"Successfully loaded {selected_model}"
    except:
        yield traceback.format_exc()


def load_lora_wrapper(selected_loras):
    yield ("Applying the following LoRAs to {}:\n\n{}".format(shared.model_name, '\n'.join(selected_loras)))
    add_lora_to_model(selected_loras)
    yield ("Successfuly applied the LoRAs")


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


def connect_lancedb(vector_db_uri = "~/.lancedb",
                   table_name = "jigsaw_old",
                   transformer_model_name = "paraphrase-albert-small-v2"):
    db = lancedb.connect(vector_db_uri)
    lancedb_tbl = db.open_table(table_name)
    transformer_model = SentenceTransformer(transformer_model_name)
    return transformer_model, lancedb_tbl


def download_model_wrapper(repo_id):
    try:
        downloader = importlib.import_module("download-model")

        model = repo_id
        branch = "main"
        check = False

        yield ("Cleaning up the model/branch names")
        model, branch = downloader.sanitize_model_and_branch_names(model, branch)

        yield ("Getting the download links from Hugging Face")
        links, sha256, is_lora = downloader.get_download_links_from_huggingface(model, branch, text_only=False)

        yield ("Getting the output folder")
        output_folder = downloader.get_output_folder(model, branch, is_lora)

        if check:
            yield ("Checking previously downloaded files")
            downloader.check_model_files(model, branch, links, sha256, output_folder)
        else:
            yield (f"Downloading files to {output_folder}")
            downloader.download_model_files(model, branch, links, sha256, output_folder, threads=1)
            yield ("Done!")
    except:
        yield traceback.format_exc()


# Update the command-line arguments based on the interface values
def update_model_parameters(state, initial=False):
    elements = ui.list_model_elements()  # the names of the parameters
    gpu_memories = []

    for i, element in enumerate(elements):
        if element not in state:
            continue

        value = state[element]
        if element.startswith('gpu_memory'):
            gpu_memories.append(value)
            continue

        if initial and vars(shared.args)[element] != vars(shared.args_defaults)[element]:
            continue

        # Setting null defaults
        if element in ['wbits', 'groupsize', 'model_type'] and value == 'None':
            value = vars(shared.args_defaults)[element]
        elif element in ['cpu_memory'] and value == 0:
            value = vars(shared.args_defaults)[element]

        # Making some simple conversions
        if element in ['wbits', 'groupsize', 'pre_layer']:
            value = int(value)
        elif element == 'cpu_memory' and value is not None:
            value = f"{value}MiB"

        setattr(shared.args, element, value)

    found_positive = False
    for i in gpu_memories:
        if i > 0:
            found_positive = True
            break

    if not (initial and vars(shared.args)['gpu_memory'] != vars(shared.args_defaults)['gpu_memory']):
        if found_positive:
            shared.args.gpu_memory = [f"{i}MiB" for i in gpu_memories]
        else:
            shared.args.gpu_memory = None


def get_model_specific_settings(model):
    settings = shared.model_config
    model_settings = {}

    for pat in settings:
        if re.match(pat.lower(), model.lower()):
            for k in settings[pat]:
                model_settings[k] = settings[pat][k]

    return model_settings


def load_model_specific_settings(model, state, return_dict=False):
    model_settings = get_model_specific_settings(model)
    for k in model_settings:
        if k in state:
            state[k] = model_settings[k]

    return state


def save_model_settings(model, state):
    if model == 'None':
        yield ("Not saving the settings because no model is loaded.")
        return

    with Path(f'{shared.args.model_dir}/config-user.yaml') as p:
        if p.exists():
            user_config = yaml.safe_load(open(p, 'r').read())
        else:
            user_config = {}

        if model not in user_config:
            user_config[model] = {}

        for k in ui.list_model_elements():
            user_config[model][k] = state[k]

        with open(p, 'w') as f:
            f.write(yaml.dump(user_config))

        yield (f"Settings for {model} saved to {p}")


def create_model_menus():
    # Finding the default values for the GPU and CPU memories
    total_mem = []
    for i in range(torch.cuda.device_count()):
        total_mem.append(math.floor(torch.cuda.get_device_properties(i).total_memory / (1024 * 1024)))

    default_gpu_mem = []
    if shared.args.gpu_memory is not None and len(shared.args.gpu_memory) > 0:
        for i in shared.args.gpu_memory:
            if 'mib' in i.lower():
                default_gpu_mem.append(int(re.sub('[a-zA-Z ]', '', i)))
            else:
                default_gpu_mem.append(int(re.sub('[a-zA-Z ]', '', i)) * 1000)
    while len(default_gpu_mem) < len(total_mem):
        default_gpu_mem.append(0)

    total_cpu_mem = math.floor(psutil.virtual_memory().total / (1024 * 1024))
    if shared.args.cpu_memory is not None:
        default_cpu_mem = re.sub('[a-zA-Z ]', '', shared.args.cpu_memory)
    else:
        default_cpu_mem = 0

    with gr.Row():
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        shared.gradio['model_menu'] = gr.Dropdown(choices=get_available_models(), value=shared.model_name, label='Model')
                        ui.create_refresh_button(shared.gradio['model_menu'], lambda: None, lambda: {'choices': get_available_models()}, 'refresh-button')

                with gr.Column():
                    with gr.Row():
                        shared.gradio['lora_menu'] = gr.Dropdown(multiselect=True, choices=get_available_loras(), value=shared.lora_names, label='LoRA(s)')
                        ui.create_refresh_button(shared.gradio['lora_menu'], lambda: None, lambda: {'choices': get_available_loras(), 'value': shared.lora_names}, 'refresh-button')

        with gr.Column():
            with gr.Row():
                shared.gradio['lora_menu_apply'] = gr.Button(value='Apply the selected LoRAs')
            with gr.Row():
                unload = gr.Button("Unload the model")
                reload = gr.Button("Reload the model")
                save_settings = gr.Button("Save settings for this model")

    with gr.Row():
        with gr.Column():
            with gr.Box():
                gr.Markdown('Transformers parameters')
                with gr.Row():
                    with gr.Column():
                        for i in range(len(total_mem)):
                            shared.gradio[f'gpu_memory_{i}'] = gr.Slider(label=f"gpu-memory in MiB for device :{i}", maximum=total_mem[i], value=default_gpu_mem[i])
                        shared.gradio['cpu_memory'] = gr.Slider(label="cpu-memory in MiB", maximum=total_cpu_mem, value=default_cpu_mem)

                    with gr.Column():
                        shared.gradio['auto_devices'] = gr.Checkbox(label="auto-devices", value=shared.args.auto_devices)
                        shared.gradio['disk'] = gr.Checkbox(label="disk", value=shared.args.disk)
                        shared.gradio['cpu'] = gr.Checkbox(label="cpu", value=shared.args.cpu)
                        shared.gradio['bf16'] = gr.Checkbox(label="bf16", value=shared.args.bf16)
                        shared.gradio['load_in_8bit'] = gr.Checkbox(label="load-in-8bit", value=shared.args.load_in_8bit)

        with gr.Column():
            with gr.Box():
                gr.Markdown('GPTQ parameters')
                with gr.Row():
                    with gr.Column():
                        shared.gradio['wbits'] = gr.Dropdown(label="wbits", choices=["None", 1, 2, 3, 4, 8], value=shared.args.wbits if shared.args.wbits > 0 else "None")
                        shared.gradio['groupsize'] = gr.Dropdown(label="groupsize", choices=["None", 32, 64, 128], value=shared.args.groupsize if shared.args.groupsize > 0 else "None")

                    with gr.Column():
                        shared.gradio['model_type'] = gr.Dropdown(label="model_type", choices=["None", "llama", "opt", "gptj"], value=shared.args.model_type or "None")
                        shared.gradio['pre_layer'] = gr.Slider(label="pre_layer", minimum=0, maximum=100, value=shared.args.pre_layer)

    with gr.Row():
        with gr.Column():
            shared.gradio['custom_model_menu'] = gr.Textbox(label="Download custom model or LoRA", info="Enter Hugging Face username/model path, e.g: facebook/galactica-125m")
            shared.gradio['download_model_button'] = gr.Button("Download")

        with gr.Column():
            shared.gradio['model_status'] = gr.Markdown('No model is loaded' if shared.model_name == 'None' else 'Ready')

    # In this event handler, the interface state is read and updated
    # with the model defaults (if any), and then the model is loaded
    shared.gradio['model_menu'].change(
        ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
        load_model_specific_settings, [shared.gradio[k] for k in ['model_menu', 'interface_state']], shared.gradio['interface_state']).then(
        ui.apply_interface_values, shared.gradio['interface_state'], [shared.gradio[k] for k in ui.list_interface_input_elements(chat=shared.is_chat())], show_progress=False).then(
        update_model_parameters, shared.gradio['interface_state'], None).then(
        load_model_wrapper, shared.gradio['model_menu'], shared.gradio['model_status'], show_progress=True)

    unload.click(
        unload_model, None, None).then(
        lambda: "Model unloaded", None, shared.gradio['model_status'])

    reload.click(
        unload_model, None, None).then(
        ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
        update_model_parameters, shared.gradio['interface_state'], None).then(
        load_model_wrapper, shared.gradio['model_menu'], shared.gradio['model_status'], show_progress=False)

    save_settings.click(
        ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
        save_model_settings, [shared.gradio[k] for k in ['model_menu', 'interface_state']], shared.gradio['model_status'], show_progress=False)

    shared.gradio['lora_menu_apply'].click(load_lora_wrapper, shared.gradio['lora_menu'], shared.gradio['model_status'], show_progress=False)
    shared.gradio['download_model_button'].click(download_model_wrapper, shared.gradio['custom_model_menu'], shared.gradio['model_status'], show_progress=False)


def create_settings_menus(default_preset):

    generate_params = load_preset_values(default_preset if not shared.args.flexgen else 'Naive', {}, return_dict=True)

    with gr.Row():
        with gr.Column():
            with gr.Row():
                shared.gradio['preset_menu'] = gr.Dropdown(choices=get_available_presets(), value=default_preset if not shared.args.flexgen else 'Naive', label='Generation parameters preset')
                ui.create_refresh_button(shared.gradio['preset_menu'], lambda: None, lambda: {'choices': get_available_presets()}, 'refresh-button')
        with gr.Column():
            shared.gradio['seed'] = gr.Number(value=shared.settings['seed'], label='Seed (-1 for random)')

    with gr.Row():
        with gr.Column():
            with gr.Box():
                gr.Markdown('Custom generation parameters ([click here to view technical documentation](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig))')
                with gr.Row():
                    with gr.Column():
                        shared.gradio['temperature'] = gr.Slider(0.01, 1.99, value=generate_params['temperature'], step=0.01, label='temperature', info='Primary factor to control randomness of outputs. 0 = deterministic (only the most likely token is used). Higher value = more randomness.')
                        shared.gradio['top_p'] = gr.Slider(0.0, 1.0, value=generate_params['top_p'], step=0.01, label='top_p', info='If not set to 1, select tokens with probabilities adding up to less than this number. Higher value = higher range of possible random results.')
                        shared.gradio['top_k'] = gr.Slider(0, 200, value=generate_params['top_k'], step=1, label='top_k', info='Similar to top_p, but select instead only the top_k most likely tokens. Higher value = higher range of possible random results.')
                        shared.gradio['typical_p'] = gr.Slider(0.0, 1.0, value=generate_params['typical_p'], step=0.01, label='typical_p', info='If not set to 1, select only tokens that are at least this much more likely to appear than random tokens, given the prior text.')
                    with gr.Column():
                        shared.gradio['repetition_penalty'] = gr.Slider(1.0, 1.5, value=generate_params['repetition_penalty'], step=0.01, label='repetition_penalty', info='Exponential penalty factor for repeating prior tokens. 1 means no penalty, higher value = less repetition, lower value = more repetition.')
                        shared.gradio['encoder_repetition_penalty'] = gr.Slider(0.8, 1.5, value=generate_params['encoder_repetition_penalty'], step=0.01, label='encoder_repetition_penalty', info='Also known as the "Hallucinations filter". Used to penalize tokens that are *not* in the prior text. Higher value = more likely to stay in context, lower value = more likely to diverge.')
                        shared.gradio['no_repeat_ngram_size'] = gr.Slider(0, 20, step=1, value=generate_params['no_repeat_ngram_size'], label='no_repeat_ngram_size', info='If not set to 0, specifies the length of token sets that are completely blocked from repeating at all. Higher values = blocks larger phrases, lower values = blocks words or letters from repeating. Only 0 or high values are a good idea in most cases.')
                        shared.gradio['min_length'] = gr.Slider(0, 2000, step=1, value=generate_params['min_length'], label='min_length', info='Minimum generation length in tokens.')
                shared.gradio['do_sample'] = gr.Checkbox(value=generate_params['do_sample'], label='do_sample')
        with gr.Column():
            with gr.Box():
                gr.Markdown('Contrastive search')
                shared.gradio['penalty_alpha'] = gr.Slider(0, 5, value=generate_params['penalty_alpha'], label='penalty_alpha')

                gr.Markdown('Beam search (uses a lot of VRAM)')
                with gr.Row():
                    with gr.Column():
                        shared.gradio['num_beams'] = gr.Slider(1, 20, step=1, value=generate_params['num_beams'], label='num_beams')
                        shared.gradio['length_penalty'] = gr.Slider(-5, 5, value=generate_params['length_penalty'], label='length_penalty')
                    with gr.Column():
                        shared.gradio['early_stopping'] = gr.Checkbox(value=generate_params['early_stopping'], label='early_stopping')

            with gr.Box():
                with gr.Row():
                    with gr.Column():
                        shared.gradio['truncation_length'] = gr.Slider(value=shared.settings['truncation_length'], minimum=shared.settings['truncation_length_min'], maximum=shared.settings['truncation_length_max'], step=1, label='Truncate the prompt up to this length', info='The leftmost tokens are removed if the prompt exceeds this length. Most models require this to be at most 2048.')
                        shared.gradio['custom_stopping_strings'] = gr.Textbox(lines=1, value=shared.settings["custom_stopping_strings"] or None, label='Custom stopping strings', info='In addition to the defaults. Written between "" and separated by commas. For instance: "\\nYour Assistant:", "\\nThe assistant:"')
                    with gr.Column():
                        shared.gradio['add_bos_token'] = gr.Checkbox(value=shared.settings['add_bos_token'], label='Add the bos_token to the beginning of prompts', info='Disabling this can make the replies more creative.')
                        shared.gradio['ban_eos_token'] = gr.Checkbox(value=shared.settings['ban_eos_token'], label='Ban the eos_token', info='Forces the model to never end the generation prematurely.')

                        shared.gradio['skip_special_tokens'] = gr.Checkbox(value=shared.settings['skip_special_tokens'], label='Skip special tokens', info='Some specific models need this unset.')

    with gr.Accordion('Soft prompt', open=False):
        with gr.Row():
            shared.gradio['softprompts_menu'] = gr.Dropdown(choices=get_available_softprompts(), value='None', label='Soft prompt')
            ui.create_refresh_button(shared.gradio['softprompts_menu'], lambda: None, lambda: {'choices': get_available_softprompts()}, 'refresh-button')

        gr.Markdown('Upload a soft prompt (.zip format):')
        with gr.Row():
            shared.gradio['upload_softprompt'] = gr.File(type='binary', file_types=['.zip'])

    shared.gradio['preset_menu'].change(load_preset_values, [shared.gradio[k] for k in ['preset_menu', 'interface_state']], [shared.gradio[k] for k in ['interface_state', 'do_sample', 'temperature', 'top_p', 'typical_p', 'repetition_penalty', 'encoder_repetition_penalty', 'top_k', 'min_length', 'no_repeat_ngram_size', 'num_beams', 'penalty_alpha', 'length_penalty', 'early_stopping']])
    shared.gradio['softprompts_menu'].change(load_soft_prompt, shared.gradio['softprompts_menu'], shared.gradio['softprompts_menu'], show_progress=True)
    shared.gradio['upload_softprompt'].upload(upload_soft_prompt, shared.gradio['upload_softprompt'], shared.gradio['softprompts_menu'])


def set_interface_arguments(interface_mode, extensions, bool_active):
    modes = ["default", "notebook", "chat", "cai_chat"]
    cmd_list = vars(shared.args)
    bool_list = [k for k in cmd_list if type(cmd_list[k]) is bool and k not in modes]

    shared.args.extensions = extensions
    for k in modes[1:]:
        setattr(shared.args, k, False)
    if interface_mode != "default":
        setattr(shared.args, interface_mode, True)

    for k in bool_list:
        setattr(shared.args, k, False)
    for k in bool_active:
        setattr(shared.args, k, True)

    shared.need_restart = True


def create_interface():

    # Defining some variables
    gen_events = []
    default_preset = shared.settings['presets'][next((k for k in shared.settings['presets'] if re.match(k.lower(), shared.model_name.lower())), 'default')]
    if len(shared.lora_names) == 1:
        default_text = load_prompt(shared.settings['lora_prompts'][next((k for k in shared.settings['lora_prompts'] if re.match(k.lower(), shared.lora_names[0].lower())), 'default')])
    else:
        default_text = load_prompt(shared.settings['prompts'][next((k for k in shared.settings['prompts'] if re.match(k.lower(), shared.model_name.lower())), 'default')])
    title = 'Text generation web UI'

    # Authentication variables
    auth = None
    if shared.args.gradio_auth_path is not None:
        gradio_auth_creds = []
        with open(shared.args.gradio_auth_path, 'r', encoding="utf8") as file:
            for line in file.readlines():
                gradio_auth_creds += [x.strip() for x in line.split(',') if x.strip()]
        auth = [tuple(cred.split(':')) for cred in gradio_auth_creds]

    # Importing the extension files and executing their setup() functions
    if shared.args.extensions is not None and len(shared.args.extensions) > 0:
        extensions_module.load_extensions()

    # Load models for moderation if moderation is true
    if shared.args.moderation:
        shared.moderation=True
        shared.transformer_model, shared.lancedb_tbl = connect_lancedb(
            vector_db_uri = shared.args.lancedb_uri,
            table_name = shared.args.lancedb_table_name,
            transformer_model_name = shared.args.transformer_model_name)
    else:
        shared.moderation=False

    with gr.Blocks(css=ui.css if not shared.is_chat() else ui.css + ui.chat_css, analytics_enabled=False, title=title, theme=ui.theme) as shared.gradio['interface']:

        # Create chat mode interface
        if shared.is_chat():
            shared.input_elements = ui.list_interface_input_elements(chat=True)
            shared.gradio['interface_state'] = gr.State({k: None for k in shared.input_elements})
            shared.gradio['Chat input'] = gr.State()

            with gr.Tab('Text generation', elem_id='main'):
                shared.gradio['display'] = gr.HTML(value=chat_html_wrapper(shared.history['visible'], shared.settings['name1'], shared.settings['name2'], 'cai-chat'))
                shared.gradio['textbox'] = gr.Textbox(label='Input')
                with gr.Row():
                    shared.gradio['Stop'] = gr.Button('Stop', elem_id='stop')
                    shared.gradio['Generate'] = gr.Button('Generate', elem_id='Generate', variant='primary')
                    shared.gradio['Continue'] = gr.Button('Continue')

                with gr.Row():
                    shared.gradio['Copy last reply'] = gr.Button('Copy last reply')
                    shared.gradio['Regenerate'] = gr.Button('Regenerate')
                    shared.gradio['Replace last reply'] = gr.Button('Replace last reply')

                with gr.Row():
                    shared.gradio['Impersonate'] = gr.Button('Impersonate')
                    shared.gradio['Send dummy message'] = gr.Button('Send dummy message')
                    shared.gradio['Send dummy reply'] = gr.Button('Send dummy reply')

                with gr.Row():
                    shared.gradio['Remove last'] = gr.Button('Remove last')
                    shared.gradio['Clear history'] = gr.Button('Clear history')
                    shared.gradio['Clear history-confirm'] = gr.Button('Confirm', variant='stop', visible=False)
                    shared.gradio['Clear history-cancel'] = gr.Button('Cancel', visible=False)

                shared.gradio['mode'] = gr.Radio(choices=['cai-chat', 'chat', 'instruct'], value=shared.settings['mode'], label='Mode')
                shared.gradio['instruction_template'] = gr.Dropdown(choices=get_available_instruction_templates(), label='Instruction template', value=shared.settings['instruction_template'], visible=shared.settings['mode'] == 'instruct', info='Change this according to the model/LoRA that you are using.')

            with gr.Tab('Character', elem_id='chat-settings'):
                with gr.Row():
                    with gr.Column(scale=8):
                        shared.gradio['name1'] = gr.Textbox(value=shared.settings['name1'], lines=1, label='Your name')
                        shared.gradio['name2'] = gr.Textbox(value=shared.settings['name2'], lines=1, label='Character\'s name')
                        shared.gradio['greeting'] = gr.Textbox(value=shared.settings['greeting'], lines=4, label='Greeting')
                        shared.gradio['context'] = gr.Textbox(value=shared.settings['context'], lines=4, label='Context')
                        shared.gradio['end_of_turn'] = gr.Textbox(value=shared.settings['end_of_turn'], lines=1, label='End of turn string')

                    with gr.Column(scale=1):
                        shared.gradio['character_picture'] = gr.Image(label='Character picture', type='pil')
                        shared.gradio['your_picture'] = gr.Image(label='Your picture', type='pil', value=Image.open(Path('cache/pfp_me.png')) if Path('cache/pfp_me.png').exists() else None)

                with gr.Row():
                    shared.gradio['character_menu'] = gr.Dropdown(choices=get_available_characters(), value='None', label='Character', elem_id='character-menu')
                    ui.create_refresh_button(shared.gradio['character_menu'], lambda: None, lambda: {'choices': get_available_characters()}, 'refresh-button')

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
                        gr.Markdown('# JSON format')
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown('1. Select the JSON file')
                                shared.gradio['upload_json'] = gr.File(type='binary', file_types=['.json'])

                            with gr.Column():
                                gr.Markdown('2. Select your character\'s profile picture (optional)')
                                shared.gradio['upload_img_bot'] = gr.File(type='binary', file_types=['image'])

                        shared.gradio['Upload character'] = gr.Button(value='Submit')
                        gr.Markdown('# TavernAI PNG format')
                        shared.gradio['upload_img_tavern'] = gr.File(type='binary', file_types=['image'])

            with gr.Tab("Parameters", elem_id="parameters"):
                with gr.Box():
                    gr.Markdown("Chat parameters")
                    with gr.Row():
                        with gr.Column():
                            shared.gradio['max_new_tokens'] = gr.Slider(minimum=shared.settings['max_new_tokens_min'], maximum=shared.settings['max_new_tokens_max'], step=1, label='max_new_tokens', value=shared.settings['max_new_tokens'])
                            shared.gradio['chat_prompt_size'] = gr.Slider(minimum=shared.settings['chat_prompt_size_min'], maximum=shared.settings['chat_prompt_size_max'], step=1, label='Maximum prompt size in tokens', value=shared.settings['chat_prompt_size'])

                        with gr.Column():
                            shared.gradio['chat_generation_attempts'] = gr.Slider(minimum=shared.settings['chat_generation_attempts_min'], maximum=shared.settings['chat_generation_attempts_max'], value=shared.settings['chat_generation_attempts'], step=1, label='Generation attempts (for longer replies)')
                            shared.gradio['stop_at_newline'] = gr.Checkbox(value=shared.settings['stop_at_newline'], label='Stop generating at new line character')

                create_settings_menus(default_preset)

        # Create notebook mode interface
        elif shared.args.notebook:
            shared.input_elements = ui.list_interface_input_elements(chat=False)
            shared.gradio['interface_state'] = gr.State({k: None for k in shared.input_elements})
            shared.gradio['last_input'] = gr.State('')
            with gr.Tab("Text generation", elem_id="main"):
                with gr.Row():
                    with gr.Column(scale=4):
                        with gr.Tab('Raw'):
                            shared.gradio['textbox'] = gr.Textbox(value=default_text, elem_classes="textbox", lines=27)

                        with gr.Tab('Markdown'):
                            shared.gradio['markdown'] = gr.Markdown()

                        with gr.Tab('HTML'):
                            shared.gradio['html'] = gr.HTML()

                        with gr.Row():
                            with gr.Column():
                                with gr.Row():
                                    shared.gradio['Generate'] = gr.Button('Generate', variant='primary')
                                    shared.gradio['Stop'] = gr.Button('Stop')
                                    shared.gradio['Undo'] = gr.Button('Undo')
                                    shared.gradio['Regenerate'] = gr.Button('Regenerate')

                            with gr.Column():
                                pass

                    with gr.Column(scale=1):
                        gr.HTML('<div style="padding-bottom: 13px"></div>')
                        shared.gradio['max_new_tokens'] = gr.Slider(minimum=shared.settings['max_new_tokens_min'], maximum=shared.settings['max_new_tokens_max'], step=1, label='max_new_tokens', value=shared.settings['max_new_tokens'])
                        with gr.Row():
                            shared.gradio['prompt_menu'] = gr.Dropdown(choices=get_available_prompts(), value='None', label='Prompt')
                            ui.create_refresh_button(shared.gradio['prompt_menu'], lambda: None, lambda: {'choices': get_available_prompts()}, 'refresh-button')

                        shared.gradio['save_prompt'] = gr.Button('Save prompt')
                        shared.gradio['status'] = gr.Markdown('')

            with gr.Tab("Parameters", elem_id="parameters"):
                create_settings_menus(default_preset)

        # Create default mode interface
        else:
            shared.input_elements = ui.list_interface_input_elements(chat=False)
            shared.gradio['interface_state'] = gr.State({k: None for k in shared.input_elements})
            shared.gradio['last_input'] = gr.State('')
            with gr.Tab("Text generation", elem_id="main"):
                with gr.Row():
                    with gr.Column():
                        shared.gradio['textbox'] = gr.Textbox(value=default_text, elem_classes="textbox_default", lines=27, label='Input')
                        shared.gradio['max_new_tokens'] = gr.Slider(minimum=shared.settings['max_new_tokens_min'], maximum=shared.settings['max_new_tokens_max'], step=1, label='max_new_tokens', value=shared.settings['max_new_tokens'])
                        with gr.Row():
                            shared.gradio['Generate'] = gr.Button('Generate', variant='primary')
                            shared.gradio['Stop'] = gr.Button('Stop')
                            shared.gradio['Continue'] = gr.Button('Continue')
                            shared.gradio['save_prompt'] = gr.Button('Save prompt')

                        with gr.Row():
                            with gr.Column():
                                with gr.Row():
                                    shared.gradio['prompt_menu'] = gr.Dropdown(choices=get_available_prompts(), value='None', label='Prompt')
                                    ui.create_refresh_button(shared.gradio['prompt_menu'], lambda: None, lambda: {'choices': get_available_prompts()}, 'refresh-button')

                            with gr.Column():
                                shared.gradio['status'] = gr.Markdown('')

                    with gr.Column():
                        with gr.Tab('Raw'):
                            shared.gradio['output_textbox'] = gr.Textbox(elem_classes="textbox_default_output", lines=27, label='Output')

                        with gr.Tab('Markdown'):
                            shared.gradio['markdown'] = gr.Markdown()

                        with gr.Tab('HTML'):
                            shared.gradio['html'] = gr.HTML()

            with gr.Tab("Parameters", elem_id="parameters"):
                create_settings_menus(default_preset)

        # Model tab
        with gr.Tab("Model", elem_id="model-tab"):
            create_model_menus()

        # Training tab
        with gr.Tab("Training", elem_id="training-tab"):
            training.create_train_interface()

        # Interface mode tab
        with gr.Tab("Interface mode", elem_id="interface-mode"):
            modes = ["default", "notebook", "chat", "cai_chat"]
            current_mode = "default"
            for mode in modes[1:]:
                if getattr(shared.args, mode):
                    current_mode = mode
                    break
            cmd_list = vars(shared.args)
            bool_list = [k for k in cmd_list if type(cmd_list[k]) is bool and k not in modes + ui.list_model_elements()]
            bool_active = [k for k in bool_list if vars(shared.args)[k]]

            gr.Markdown("*Experimental*")
            shared.gradio['interface_modes_menu'] = gr.Dropdown(choices=modes, value=current_mode, label="Mode")
            shared.gradio['extensions_menu'] = gr.CheckboxGroup(choices=get_available_extensions(), value=shared.args.extensions, label="Available extensions")
            shared.gradio['bool_menu'] = gr.CheckboxGroup(choices=bool_list, value=bool_active, label="Boolean command-line flags")
            shared.gradio['reset_interface'] = gr.Button("Apply and restart the interface")

            # Reset interface event
            shared.gradio['reset_interface'].click(
                set_interface_arguments, [shared.gradio[k] for k in ['interface_modes_menu', 'extensions_menu', 'bool_menu']], None).then(
                lambda: None, None, None, _js='() => {document.body.innerHTML=\'<h1 style="font-family:monospace;margin-top:20%;color:lightgray;text-align:center;">Reloading...</h1>\'; setTimeout(function(){location.reload()},2500); return []}')

        # Extensions block
        if shared.args.extensions is not None:
            extensions_module.create_extensions_block()

        # Create the invisible elements that define the API
        if not shared.is_chat():
            api.create_apis()

        # chat mode event handlers
        if shared.is_chat():
            shared.input_params = [shared.gradio[k] for k in ['Chat input', 'interface_state']]
            clear_arr = [shared.gradio[k] for k in ['Clear history-confirm', 'Clear history', 'Clear history-cancel']]
            reload_inputs = [shared.gradio[k] for k in ['name1', 'name2', 'mode']]

            gen_events.append(shared.gradio['Generate'].click(
                ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
                lambda x: (x, ''), shared.gradio['textbox'], [shared.gradio['Chat input'], shared.gradio['textbox']], show_progress=False).then(
                chat.cai_chatbot_wrapper, shared.input_params, shared.gradio['display'], show_progress=shared.args.no_stream).then(
                chat.save_history, shared.gradio['mode'], None, show_progress=False)
            )

            gen_events.append(shared.gradio['textbox'].submit(
                ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
                lambda x: (x, ''), shared.gradio['textbox'], [shared.gradio['Chat input'], shared.gradio['textbox']], show_progress=False).then(
                chat.cai_chatbot_wrapper, shared.input_params, shared.gradio['display'], show_progress=shared.args.no_stream).then(
                chat.save_history, shared.gradio['mode'], None, show_progress=False)
            )

            gen_events.append(shared.gradio['Regenerate'].click(
                ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
                chat.regenerate_wrapper, shared.input_params, shared.gradio['display'], show_progress=shared.args.no_stream).then(
                chat.save_history, shared.gradio['mode'], None, show_progress=False)
            )

            gen_events.append(shared.gradio['Continue'].click(
                ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
                chat.continue_wrapper, shared.input_params, shared.gradio['display'], show_progress=shared.args.no_stream).then(
                chat.save_history, shared.gradio['mode'], None, show_progress=False)
            )

            gen_events.append(shared.gradio['Impersonate'].click(
                ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
                chat.impersonate_wrapper, shared.input_params, shared.gradio['textbox'], show_progress=shared.args.no_stream)
            )

            shared.gradio['Replace last reply'].click(
                chat.replace_last_reply, [shared.gradio[k] for k in ['textbox', 'name1', 'name2', 'mode']], shared.gradio['display'], show_progress=shared.args.no_stream).then(
                lambda x: '', shared.gradio['textbox'], shared.gradio['textbox'], show_progress=False).then(
                chat.save_history, shared.gradio['mode'], None, show_progress=False)

            shared.gradio['Send dummy message'].click(
                chat.send_dummy_message, [shared.gradio[k] for k in ['textbox', 'name1', 'name2', 'mode']], shared.gradio['display'], show_progress=shared.args.no_stream).then(
                lambda x: '', shared.gradio['textbox'], shared.gradio['textbox'], show_progress=False).then(
                chat.save_history, shared.gradio['mode'], None, show_progress=False)

            shared.gradio['Send dummy reply'].click(
                chat.send_dummy_reply, [shared.gradio[k] for k in ['textbox', 'name1', 'name2', 'mode']], shared.gradio['display'], show_progress=shared.args.no_stream).then(
                lambda x: '', shared.gradio['textbox'], shared.gradio['textbox'], show_progress=False).then(
                chat.save_history, shared.gradio['mode'], None, show_progress=False)

            shared.gradio['Clear history-confirm'].click(
                lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)], None, clear_arr).then(
                chat.clear_chat_log, [shared.gradio[k] for k in ['name1', 'name2', 'greeting', 'mode']], shared.gradio['display']).then(
                chat.save_history, shared.gradio['mode'], None, show_progress=False)

            shared.gradio['Stop'].click(
                stop_everything_event, None, None, queue=False, cancels=gen_events if shared.args.no_stream else None).then(
                chat.redraw_html, reload_inputs, shared.gradio['display'])

            shared.gradio['mode'].change(
                lambda x: gr.update(visible=x == 'instruct'), shared.gradio['mode'], shared.gradio['instruction_template']).then(
                lambda x: gr.update(interactive=x != 'instruct'), shared.gradio['mode'], shared.gradio['character_menu']).then(
                chat.redraw_html, reload_inputs, shared.gradio['display'])

            shared.gradio['instruction_template'].change(
                chat.load_character, [shared.gradio[k] for k in ['instruction_template', 'name1', 'name2', 'mode']], [shared.gradio[k] for k in ['name1', 'name2', 'character_picture', 'greeting', 'context', 'end_of_turn', 'display']]).then(
                chat.redraw_html, reload_inputs, shared.gradio['display'])

            shared.gradio['upload_chat_history'].upload(
                chat.load_history, [shared.gradio[k] for k in ['upload_chat_history', 'name1', 'name2']], None).then(
                chat.redraw_html, reload_inputs, shared.gradio['display'])

            shared.gradio['Copy last reply'].click(chat.send_last_reply_to_input, None, shared.gradio['textbox'], show_progress=shared.args.no_stream)
            shared.gradio['Clear history'].click(lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)], None, clear_arr)
            shared.gradio['Clear history-cancel'].click(lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)], None, clear_arr)
            shared.gradio['Remove last'].click(chat.remove_last_message, [shared.gradio[k] for k in ['name1', 'name2', 'mode']], [shared.gradio['display'], shared.gradio['textbox']], show_progress=False)
            shared.gradio['download_button'].click(lambda x: chat.save_history(x, timestamp=True), shared.gradio['mode'], shared.gradio['download'])
            shared.gradio['Upload character'].click(chat.upload_character, [shared.gradio['upload_json'], shared.gradio['upload_img_bot']], [shared.gradio['character_menu']])
            shared.gradio['character_menu'].change(chat.load_character, [shared.gradio[k] for k in ['character_menu', 'name1', 'name2', 'mode']], [shared.gradio[k] for k in ['name1', 'name2', 'character_picture', 'greeting', 'context', 'end_of_turn', 'display']])
            shared.gradio['upload_img_tavern'].upload(chat.upload_tavern_character, [shared.gradio['upload_img_tavern'], shared.gradio['name1'], shared.gradio['name2']], [shared.gradio['character_menu']])
            shared.gradio['your_picture'].change(chat.upload_your_profile_picture, [shared.gradio[k] for k in ['your_picture', 'name1', 'name2', 'mode']], shared.gradio['display'])

            shared.gradio['interface'].load(None, None, None, _js=f"() => {{{ui.main_js+ui.chat_js}}}")
            shared.gradio['interface'].load(chat.load_character, [shared.gradio[k] for k in ['instruction_template', 'name1', 'name2', 'mode']], [shared.gradio[k] for k in ['name1', 'name2', 'character_picture', 'greeting', 'context', 'end_of_turn', 'display']])
            shared.gradio['interface'].load(chat.load_default_history, [shared.gradio[k] for k in ['name1', 'name2']], None)
            shared.gradio['interface'].load(chat.redraw_html, reload_inputs, shared.gradio['display'], show_progress=True)

        # notebook/default modes event handlers
        else:
            shared.input_params = [shared.gradio[k] for k in ['textbox', 'interface_state']]

            if shared.args.notebook:
                output_params = [shared.gradio[k] for k in ['textbox', 'markdown', 'html']]
            else:
                output_params = [shared.gradio[k] for k in ['output_textbox', 'markdown', 'html']]

            gen_events.append(shared.gradio['Generate'].click(
                lambda x: x, shared.gradio['textbox'], shared.gradio['last_input']).then(
                ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
                generate_reply, shared.input_params, output_params, show_progress=shared.args.no_stream)  # .then(
                # None, None, None, _js="() => {element = document.getElementsByTagName('textarea')[0]; element.scrollTop = element.scrollHeight}")
            )

            gen_events.append(shared.gradio['textbox'].submit(
                lambda x: x, shared.gradio['textbox'], shared.gradio['last_input']).then(
                ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
                generate_reply, shared.input_params, output_params, show_progress=shared.args.no_stream)  # .then(
                # None, None, None, _js="() => {element = document.getElementsByTagName('textarea')[0]; element.scrollTop = element.scrollHeight}")
            )

            if shared.args.notebook:
                shared.gradio['Undo'].click(lambda x: x, shared.gradio['last_input'], shared.gradio['textbox'], show_progress=False)
                gen_events.append(shared.gradio['Regenerate'].click(
                    lambda x: x, shared.gradio['last_input'], shared.gradio['textbox'], show_progress=False).then(
                    ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
                    generate_reply, shared.input_params, output_params, show_progress=shared.args.no_stream)  # .then(
                    # None, None, None, _js="() => {element = document.getElementsByTagName('textarea')[0]; element.scrollTop = element.scrollHeight}")
                )
            else:
                gen_events.append(shared.gradio['Continue'].click(
                    ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
                    generate_reply, [shared.gradio['output_textbox']] + shared.input_params[1:], output_params, show_progress=shared.args.no_stream)  # .then(
                    # None, None, None, _js="() => {element = document.getElementsByTagName('textarea')[1]; element.scrollTop = element.scrollHeight}")
                )

            shared.gradio['Stop'].click(stop_everything_event, None, None, queue=False, cancels=gen_events if shared.args.no_stream else None)
            shared.gradio['prompt_menu'].change(load_prompt, [shared.gradio['prompt_menu']], [shared.gradio['textbox']], show_progress=False)
            shared.gradio['save_prompt'].click(save_prompt, [shared.gradio['textbox']], [shared.gradio['status']], show_progress=False)
            shared.gradio['interface'].load(None, None, None, _js=f"() => {{{ui.main_js}}}")

    # Launch the interface
    shared.gradio['interface'].queue()
    if shared.args.listen:
        shared.gradio['interface'].launch(prevent_thread_lock=True, share=shared.args.share, server_name=shared.args.listen_host or '0.0.0.0', server_port=shared.args.listen_port, inbrowser=shared.args.auto_launch, auth=auth)
    else:
        shared.gradio['interface'].launch(prevent_thread_lock=True, share=shared.args.share, server_port=shared.args.listen_port, inbrowser=shared.args.auto_launch, auth=auth)


if __name__ == "__main__":

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

    available_models = get_available_models()

    # Model defined through --model
    if shared.args.model is not None:
        shared.model_name = shared.args.model

    # Only one model is available
    elif len(available_models) == 1:
        shared.model_name = available_models[0]

    # Select the model from a command-line menu
    elif shared.args.model_menu:
        if len(available_models) == 0:
            print('No models are available! Please download at least one.')
            sys.exit(0)
        else:
            print('The following models are available:\n')
            for i, model in enumerate(available_models):
                print(f'{i+1}. {model}')
            print(f'\nWhich one do you want to load? 1-{len(available_models)}\n')
            i = int(input()) - 1
            print()
        shared.model_name = available_models[i]

    # If any model has been selected, load it
    if shared.model_name != 'None':

        model_settings = get_model_specific_settings(shared.model_name)
        shared.settings.update(model_settings)  # hijacking the interface defaults
        update_model_parameters(model_settings, initial=True)  # hijacking the command-line arguments

        # Load the model
        shared.model, shared.tokenizer = load_model(shared.model_name)
        if shared.args.lora:
            add_lora_to_model([shared.args.lora])

    # Launch the web UI
    create_interface()
    while True:
        time.sleep(0.5)
        if shared.need_restart:
            shared.need_restart = False
            shared.gradio['interface'].close()
            create_interface()
