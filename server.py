import os
import warnings

from modules.logging_colors import logger
from modules.block_requests import RequestBlocker

os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
os.environ['BITSANDBYTES_NOWELCOME'] = '1'
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

with RequestBlocker():
    import gradio as gr

import matplotlib
matplotlib.use('Agg')  # This fixes LaTeX rendering on some systems

import importlib
import json
import math
import os
import re
import sys
import time
import traceback
from functools import partial
from pathlib import Path
from threading import Lock

import psutil
import torch
import yaml
from PIL import Image

import modules.extensions as extensions_module
from modules import chat, loaders, presets, shared, training, ui, utils
from modules.extensions import apply_extensions
from modules.github import clone_or_pull_repository
from modules.html_generator import chat_html_wrapper
from modules.LoRA import add_lora_to_model
from modules.models import load_model, unload_model
from modules.models_settings import (
    apply_model_settings_to_state,
    get_model_settings_from_yamls,
    save_model_settings,
    update_model_parameters
)
from modules.text_generation import (
    generate_reply_wrapper,
    get_encoded_length,
    stop_everything_event
)


def load_model_wrapper(selected_model, loader, autoload=False):
    if not autoload:
        yield f"The settings for {selected_model} have been updated.\nClick on \"Load the model\" to load it."
        return

    if selected_model == 'None':
        yield "No model selected"
    else:
        try:
            yield f"Loading {selected_model}..."
            shared.model_name = selected_model
            unload_model()
            if selected_model != '':
                shared.model, shared.tokenizer = load_model(shared.model_name, loader)

            if shared.model is not None:
                yield f"Successfully loaded {selected_model}"
            else:
                yield f"Failed to load {selected_model}."
        except:
            exc = traceback.format_exc()
            logger.error('Failed to load the model.')
            print(exc)
            yield exc


def load_lora_wrapper(selected_loras):
    yield ("Applying the following LoRAs to {}:\n\n{}".format(shared.model_name, '\n'.join(selected_loras)))
    add_lora_to_model(selected_loras)
    yield ("Successfuly applied the LoRAs")


def load_prompt(fname):
    if fname in ['None', '']:
        return ''
    elif fname.startswith('Instruct-'):
        fname = re.sub('^Instruct-', '', fname)
        file_path = Path(f'characters/instruction-following/{fname}.yaml')
        if not file_path.exists():
            return ''

        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            output = ''
            if 'context' in data:
                output += data['context']

            replacements = {
                '<|user|>': data['user'],
                '<|bot|>': data['bot'],
                '<|user-message|>': 'Input',
            }

            output += utils.replace_all(data['turn_template'].split('<|bot-message|>')[0], replacements)
            return output.rstrip(' ')
    else:
        file_path = Path(f'prompts/{fname}.txt')
        if not file_path.exists():
            return ''

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            if text[-1] == '\n':
                text = text[:-1]

            return text


def count_tokens(text):
    try:
        tokens = get_encoded_length(text)
        return f'{tokens} tokens in the input.'
    except:
        return 'Couldn\'t count the number of tokens. Is a tokenizer loaded?'


def download_model_wrapper(repo_id, progress=gr.Progress()):
    try:
        downloader_module = importlib.import_module("download-model")
        downloader = downloader_module.ModelDownloader()
        repo_id_parts = repo_id.split(":")
        model = repo_id_parts[0] if len(repo_id_parts) > 0 else repo_id
        branch = repo_id_parts[1] if len(repo_id_parts) > 1 else "main"
        check = False

        progress(0.0)
        yield ("Cleaning up the model/branch names")
        model, branch = downloader.sanitize_model_and_branch_names(model, branch)

        yield ("Getting the download links from Hugging Face")
        links, sha256, is_lora = downloader.get_download_links_from_huggingface(model, branch, text_only=False)

        yield ("Getting the output folder")
        output_folder = downloader.get_output_folder(model, branch, is_lora)

        if check:
            progress(0.5)
            yield ("Checking previously downloaded files")
            downloader.check_model_files(model, branch, links, sha256, output_folder)
            progress(1.0)
        else:
            yield (f"Downloading files to {output_folder}")
            downloader.download_model_files(model, branch, links, sha256, output_folder, progress_bar=progress, threads=1)
            yield ("Done!")
    except:
        progress(1.0)
        yield traceback.format_exc()


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
                        shared.gradio['model_menu'] = gr.Dropdown(choices=utils.get_available_models(), value=shared.model_name, label='Model', elem_classes='slim-dropdown')
                        ui.create_refresh_button(shared.gradio['model_menu'], lambda: None, lambda: {'choices': utils.get_available_models()}, 'refresh-button')
                        load = gr.Button("Load", visible=not shared.settings['autoload_model'], elem_classes='refresh-button')
                        unload = gr.Button("Unload", elem_classes='refresh-button')
                        reload = gr.Button("Reload", elem_classes='refresh-button')
                        save_settings = gr.Button("Save settings", elem_classes='refresh-button')

                with gr.Column():
                    with gr.Row():
                        shared.gradio['lora_menu'] = gr.Dropdown(multiselect=True, choices=utils.get_available_loras(), value=shared.lora_names, label='LoRA(s)', elem_classes='slim-dropdown')
                        ui.create_refresh_button(shared.gradio['lora_menu'], lambda: None, lambda: {'choices': utils.get_available_loras(), 'value': shared.lora_names}, 'refresh-button')
                        shared.gradio['lora_menu_apply'] = gr.Button(value='Apply LoRAs', elem_classes='refresh-button')

    with gr.Row():
        with gr.Column():
            shared.gradio['loader'] = gr.Dropdown(label="Model loader", choices=["Transformers", "AutoGPTQ", "GPTQ-for-LLaMa", "ExLlama", "ExLlama_HF", "llama.cpp"], value=None)
            with gr.Box():
                with gr.Row():
                    with gr.Column():
                        for i in range(len(total_mem)):
                            shared.gradio[f'gpu_memory_{i}'] = gr.Slider(label=f"gpu-memory in MiB for device :{i}", maximum=total_mem[i], value=default_gpu_mem[i])

                        shared.gradio['cpu_memory'] = gr.Slider(label="cpu-memory in MiB", maximum=total_cpu_mem, value=default_cpu_mem)
                        shared.gradio['transformers_info'] = gr.Markdown('load-in-4bit params:')
                        shared.gradio['compute_dtype'] = gr.Dropdown(label="compute_dtype", choices=["bfloat16", "float16", "float32"], value=shared.args.compute_dtype)
                        shared.gradio['quant_type'] = gr.Dropdown(label="quant_type", choices=["nf4", "fp4"], value=shared.args.quant_type)
                        shared.gradio['threads'] = gr.Slider(label="threads", minimum=0, step=1, maximum=32, value=shared.args.threads)
                        shared.gradio['n_batch'] = gr.Slider(label="n_batch", minimum=1, maximum=2048, value=shared.args.n_batch)
                        shared.gradio['n_gpu_layers'] = gr.Slider(label="n-gpu-layers", minimum=0, maximum=128, value=shared.args.n_gpu_layers)
                        shared.gradio['n_ctx'] = gr.Slider(minimum=0, maximum=16384, step=256, label="n_ctx", value=shared.args.n_ctx)
                        shared.gradio['wbits'] = gr.Dropdown(label="wbits", choices=["None", 1, 2, 3, 4, 8], value=shared.args.wbits if shared.args.wbits > 0 else "None")
                        shared.gradio['groupsize'] = gr.Dropdown(label="groupsize", choices=["None", 32, 64, 128, 1024], value=shared.args.groupsize if shared.args.groupsize > 0 else "None")
                        shared.gradio['model_type'] = gr.Dropdown(label="model_type", choices=["None", "llama", "opt", "gptj"], value=shared.args.model_type or "None")
                        shared.gradio['pre_layer'] = gr.Slider(label="pre_layer", minimum=0, maximum=100, value=shared.args.pre_layer[0] if shared.args.pre_layer is not None else 0)
                        shared.gradio['autogptq_info'] = gr.Markdown('On some systems, AutoGPTQ can be 2x slower than GPTQ-for-LLaMa. You can manually select the GPTQ-for-LLaMa loader above.')
                        shared.gradio['gpu_split'] = gr.Textbox(label='gpu-split', info='Comma-separated list of VRAM (in GB) to use per GPU. Example: 20,7,7')
                        shared.gradio['max_seq_len'] = gr.Slider(label='max_seq_len', minimum=2048, maximum=16384, step=256, info='Maximum sequence length.', value=shared.args.max_seq_len)
                        shared.gradio['compress_pos_emb'] = gr.Slider(label='compress_pos_emb', minimum=1, maximum=8, step=1, info='Positional embeddings compression factor. Should typically be set to max_seq_len / 2048.', value=shared.args.compress_pos_emb)

                    with gr.Column():
                        shared.gradio['triton'] = gr.Checkbox(label="triton", value=shared.args.triton)
                        shared.gradio['no_inject_fused_attention'] = gr.Checkbox(label="no_inject_fused_attention", value=shared.args.no_inject_fused_attention, info='Disable fused attention. Fused attention improves inference performance but uses more VRAM. Disable if running low on VRAM.')
                        shared.gradio['no_inject_fused_mlp'] = gr.Checkbox(label="no_inject_fused_mlp", value=shared.args.no_inject_fused_mlp, info='Affects Triton only. Disable fused MLP. Fused MLP improves performance but uses more VRAM. Disable if running low on VRAM.')
                        shared.gradio['no_use_cuda_fp16'] = gr.Checkbox(label="no_use_cuda_fp16", value=shared.args.no_use_cuda_fp16, info='This can make models faster on some systems.')
                        shared.gradio['desc_act'] = gr.Checkbox(label="desc_act", value=shared.args.desc_act, info='\'desc_act\', \'wbits\', and \'groupsize\' are used for old models without a quantize_config.json.')
                        shared.gradio['cpu'] = gr.Checkbox(label="cpu", value=shared.args.cpu)
                        shared.gradio['load_in_8bit'] = gr.Checkbox(label="load-in-8bit", value=shared.args.load_in_8bit)
                        shared.gradio['bf16'] = gr.Checkbox(label="bf16", value=shared.args.bf16)
                        shared.gradio['auto_devices'] = gr.Checkbox(label="auto-devices", value=shared.args.auto_devices)
                        shared.gradio['disk'] = gr.Checkbox(label="disk", value=shared.args.disk)
                        shared.gradio['load_in_4bit'] = gr.Checkbox(label="load-in-4bit", value=shared.args.load_in_4bit)
                        shared.gradio['use_double_quant'] = gr.Checkbox(label="use_double_quant", value=shared.args.use_double_quant)
                        shared.gradio['no_mmap'] = gr.Checkbox(label="no-mmap", value=shared.args.no_mmap)
                        shared.gradio['mlock'] = gr.Checkbox(label="mlock", value=shared.args.mlock)
                        shared.gradio['llama_cpp_seed'] = gr.Number(label='Seed (0 for random)', value=shared.args.llama_cpp_seed)
                        shared.gradio['trust_remote_code'] = gr.Checkbox(label="trust-remote-code", value=shared.args.trust_remote_code, info='Make sure to inspect the .py files inside the model folder before loading it with this option enabled.')
                        shared.gradio['gptq_for_llama_info'] = gr.Markdown('GPTQ-for-LLaMa is currently 2x faster than AutoGPTQ on some systems. It is installed by default with the one-click installers. Otherwise, it has to be installed manually following the instructions here: [instructions](https://github.com/oobabooga/text-generation-webui/blob/main/docs/GPTQ-models-(4-bit-mode).md#installation-1).')
                        shared.gradio['exllama_info'] = gr.Markdown('For more information, consult the [docs](https://github.com/oobabooga/text-generation-webui/blob/main/docs/ExLlama.md).')
                        shared.gradio['exllama_HF_info'] = gr.Markdown('ExLlama_HF is a wrapper that lets you use ExLlama like a Transformers model, which means it can use the Transformers samplers. It\'s a bit slower than the regular ExLlama.')

        with gr.Column():
            with gr.Row():
                shared.gradio['autoload_model'] = gr.Checkbox(value=shared.settings['autoload_model'], label='Autoload the model', info='Whether to load the model as soon as it is selected in the Model dropdown.')

            shared.gradio['custom_model_menu'] = gr.Textbox(label="Download custom model or LoRA", info="Enter the Hugging Face username/model path, for instance: facebook/galactica-125m. To specify a branch, add it at the end after a \":\" character like this: facebook/galactica-125m:main")
            shared.gradio['download_model_button'] = gr.Button("Download")

            with gr.Row():
                shared.gradio['model_status'] = gr.Markdown('No model is loaded' if shared.model_name == 'None' else 'Ready')

    shared.gradio['loader'].change(loaders.make_loader_params_visible, shared.gradio['loader'], [shared.gradio[k] for k in loaders.get_all_params()])

    # In this event handler, the interface state is read and updated
    # with the model defaults (if any), and then the model is loaded
    # unless "autoload_model" is unchecked
    shared.gradio['model_menu'].change(
        ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
        apply_model_settings_to_state, [shared.gradio[k] for k in ['model_menu', 'interface_state']], shared.gradio['interface_state']).then(
        ui.apply_interface_values, shared.gradio['interface_state'], [shared.gradio[k] for k in ui.list_interface_input_elements(chat=shared.is_chat())], show_progress=False).then(
        update_model_parameters, shared.gradio['interface_state'], None).then(
        load_model_wrapper, [shared.gradio[k] for k in ['model_menu', 'loader', 'autoload_model']], shared.gradio['model_status'], show_progress=False)

    load.click(
        ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
        update_model_parameters, shared.gradio['interface_state'], None).then(
        partial(load_model_wrapper, autoload=True), [shared.gradio[k] for k in ['model_menu', 'loader']], shared.gradio['model_status'], show_progress=False)

    unload.click(
        unload_model, None, None).then(
        lambda: "Model unloaded", None, shared.gradio['model_status'])

    reload.click(
        unload_model, None, None).then(
        ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
        update_model_parameters, shared.gradio['interface_state'], None).then(
        partial(load_model_wrapper, autoload=True), [shared.gradio[k] for k in ['model_menu', 'loader']], shared.gradio['model_status'], show_progress=False)

    save_settings.click(
        ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
        save_model_settings, [shared.gradio[k] for k in ['model_menu', 'interface_state']], shared.gradio['model_status'], show_progress=False)

    shared.gradio['lora_menu_apply'].click(load_lora_wrapper, shared.gradio['lora_menu'], shared.gradio['model_status'], show_progress=False)
    shared.gradio['download_model_button'].click(download_model_wrapper, shared.gradio['custom_model_menu'], shared.gradio['model_status'], show_progress=True)
    shared.gradio['autoload_model'].change(lambda x: gr.update(visible=not x), shared.gradio['autoload_model'], load)


def create_chat_settings_menus():
    if not shared.is_chat():
        return

    with gr.Box():
        gr.Markdown("Chat parameters")
        with gr.Row():
            with gr.Column():
                shared.gradio['max_new_tokens'] = gr.Slider(minimum=shared.settings['max_new_tokens_min'], maximum=shared.settings['max_new_tokens_max'], step=1, label='max_new_tokens', value=shared.settings['max_new_tokens'])
                shared.gradio['chat_generation_attempts'] = gr.Slider(minimum=shared.settings['chat_generation_attempts_min'], maximum=shared.settings['chat_generation_attempts_max'], value=shared.settings['chat_generation_attempts'], step=1, label='Generation attempts (for longer replies)', info='New generations will be called until either this number is reached or no new content is generated between two iterations.')

            with gr.Column():
                shared.gradio['stop_at_newline'] = gr.Checkbox(value=shared.settings['stop_at_newline'], label='Stop generating at new line character')


def create_settings_menus(default_preset):
    generate_params = presets.load_preset(default_preset)
    with gr.Row():
        with gr.Column():
            with gr.Row():
                shared.gradio['preset_menu'] = gr.Dropdown(choices=utils.get_available_presets(), value=default_preset if not shared.args.flexgen else 'Naive', label='Generation parameters preset', elem_classes='slim-dropdown')
                ui.create_refresh_button(shared.gradio['preset_menu'], lambda: None, lambda: {'choices': utils.get_available_presets()}, 'refresh-button')
                shared.gradio['save_preset'] = gr.Button('💾', elem_classes='refresh-button')
                shared.gradio['delete_preset'] = gr.Button('🗑️', elem_classes='refresh-button')

        with gr.Column():
            shared.gradio['seed'] = gr.Number(value=shared.settings['seed'], label='Seed (-1 for random)')

    with gr.Row():
        with gr.Column():
            with gr.Box():
                gr.Markdown('Main parameters')
                with gr.Row():
                    with gr.Column():
                        shared.gradio['temperature'] = gr.Slider(0.01, 1.99, value=generate_params['temperature'], step=0.01, label='temperature', info='Primary factor to control randomness of outputs. 0 = deterministic (only the most likely token is used). Higher value = more randomness.')
                        shared.gradio['top_p'] = gr.Slider(0.0, 1.0, value=generate_params['top_p'], step=0.01, label='top_p', info='If not set to 1, select tokens with probabilities adding up to less than this number. Higher value = higher range of possible random results.')
                        shared.gradio['top_k'] = gr.Slider(0, 200, value=generate_params['top_k'], step=1, label='top_k', info='Similar to top_p, but select instead only the top_k most likely tokens. Higher value = higher range of possible random results.')
                        shared.gradio['typical_p'] = gr.Slider(0.0, 1.0, value=generate_params['typical_p'], step=0.01, label='typical_p', info='If not set to 1, select only tokens that are at least this much more likely to appear than random tokens, given the prior text.')
                        shared.gradio['epsilon_cutoff'] = gr.Slider(0, 9, value=generate_params['epsilon_cutoff'], step=0.01, label='epsilon_cutoff', info='In units of 1e-4; a reasonable value is 3. This sets a probability floor below which tokens are excluded from being sampled. Should be used with top_p, top_k, and eta_cutoff set to 0.')
                        shared.gradio['eta_cutoff'] = gr.Slider(0, 20, value=generate_params['eta_cutoff'], step=0.01, label='eta_cutoff', info='In units of 1e-4; a reasonable value is 3. Should be used with top_p, top_k, and epsilon_cutoff set to 0.')

                    with gr.Column():
                        shared.gradio['repetition_penalty'] = gr.Slider(1.0, 1.5, value=generate_params['repetition_penalty'], step=0.01, label='repetition_penalty', info='Exponential penalty factor for repeating prior tokens. 1 means no penalty, higher value = less repetition, lower value = more repetition.')
                        shared.gradio['encoder_repetition_penalty'] = gr.Slider(0.8, 1.5, value=generate_params['encoder_repetition_penalty'], step=0.01, label='encoder_repetition_penalty', info='Also known as the "Hallucinations filter". Used to penalize tokens that are *not* in the prior text. Higher value = more likely to stay in context, lower value = more likely to diverge.')
                        shared.gradio['no_repeat_ngram_size'] = gr.Slider(0, 20, step=1, value=generate_params['no_repeat_ngram_size'], label='no_repeat_ngram_size', info='If not set to 0, specifies the length of token sets that are completely blocked from repeating at all. Higher values = blocks larger phrases, lower values = blocks words or letters from repeating. Only 0 or high values are a good idea in most cases.')
                        shared.gradio['min_length'] = gr.Slider(0, 2000, step=1, value=generate_params['min_length'], label='min_length', info='Minimum generation length in tokens.')
                        shared.gradio['tfs'] = gr.Slider(0.0, 1.0, value=generate_params['tfs'], step=0.01, label='tfs')
                        shared.gradio['top_a'] = gr.Slider(0.0, 1.0, value=generate_params['top_a'], step=0.01, label='top_a')
                        shared.gradio['do_sample'] = gr.Checkbox(value=generate_params['do_sample'], label='do_sample')

            gr.Markdown('[Click here for more information.](https://github.com/oobabooga/text-generation-webui/blob/main/docs/Generation-parameters.md)')

        with gr.Column():
            create_chat_settings_menus()
            with gr.Box():
                with gr.Row():
                    with gr.Column():
                        gr.Markdown('Contrastive search')
                        shared.gradio['penalty_alpha'] = gr.Slider(0, 5, value=generate_params['penalty_alpha'], label='penalty_alpha', info='Contrastive Search is enabled by setting this to greater than zero and unchecking "do_sample". It should be used with a low value of top_k, for instance, top_k = 4.')

                        gr.Markdown('Beam search')
                        shared.gradio['num_beams'] = gr.Slider(1, 20, step=1, value=generate_params['num_beams'], label='num_beams')
                        shared.gradio['length_penalty'] = gr.Slider(-5, 5, value=generate_params['length_penalty'], label='length_penalty')
                        shared.gradio['early_stopping'] = gr.Checkbox(value=generate_params['early_stopping'], label='early_stopping')

                    with gr.Column():
                        gr.Markdown('Mirostat (mode=1 is only for llama.cpp)')
                        shared.gradio['mirostat_mode'] = gr.Slider(0, 2, step=1, value=generate_params['mirostat_mode'], label='mirostat_mode')
                        shared.gradio['mirostat_tau'] = gr.Slider(0, 10, step=0.01, value=generate_params['mirostat_tau'], label='mirostat_tau')
                        shared.gradio['mirostat_eta'] = gr.Slider(0, 1, step=0.01, value=generate_params['mirostat_eta'], label='mirostat_eta')

            with gr.Box():
                with gr.Row():
                    with gr.Column():
                        shared.gradio['truncation_length'] = gr.Slider(value=shared.settings['truncation_length'], minimum=shared.settings['truncation_length_min'], maximum=shared.settings['truncation_length_max'], step=256, label='Truncate the prompt up to this length', info='The leftmost tokens are removed if the prompt exceeds this length. Most models require this to be at most 2048.')
                        shared.gradio['custom_stopping_strings'] = gr.Textbox(lines=1, value=shared.settings["custom_stopping_strings"] or None, label='Custom stopping strings', info='In addition to the defaults. Written between "" and separated by commas. For instance: "\\nYour Assistant:", "\\nThe assistant:"')
                    with gr.Column():
                        shared.gradio['ban_eos_token'] = gr.Checkbox(value=shared.settings['ban_eos_token'], label='Ban the eos_token', info='Forces the model to never end the generation prematurely.')
                        shared.gradio['add_bos_token'] = gr.Checkbox(value=shared.settings['add_bos_token'], label='Add the bos_token to the beginning of prompts', info='Disabling this can make the replies more creative.')

                        shared.gradio['skip_special_tokens'] = gr.Checkbox(value=shared.settings['skip_special_tokens'], label='Skip special tokens', info='Some specific models need this unset.')
                        shared.gradio['stream'] = gr.Checkbox(value=not shared.args.no_stream, label='Activate text streaming')

    shared.gradio['preset_menu'].change(presets.load_preset_for_ui, [shared.gradio[k] for k in ['preset_menu', 'interface_state']], [shared.gradio[k] for k in ['interface_state', 'do_sample', 'temperature', 'top_p', 'typical_p', 'epsilon_cutoff', 'eta_cutoff', 'repetition_penalty', 'encoder_repetition_penalty', 'top_k', 'min_length', 'no_repeat_ngram_size', 'num_beams', 'penalty_alpha', 'length_penalty', 'early_stopping', 'mirostat_mode', 'mirostat_tau', 'mirostat_eta', 'tfs', 'top_a']])


def create_file_saving_menus():

    # Text file saver
    with gr.Box(visible=False, elem_classes='file-saver') as shared.gradio['file_saver']:
        shared.gradio['save_filename'] = gr.Textbox(lines=1, label='File name')
        shared.gradio['save_root'] = gr.Textbox(lines=1, label='File folder', info='For reference. Unchangeable.', interactive=False)
        shared.gradio['save_contents'] = gr.Textbox(lines=10, label='File contents')
        with gr.Row():
            shared.gradio['save_confirm'] = gr.Button('Save', elem_classes="small-button")
            shared.gradio['save_cancel'] = gr.Button('Cancel', elem_classes="small-button")

    # Text file deleter
    with gr.Box(visible=False, elem_classes='file-saver') as shared.gradio['file_deleter']:
        shared.gradio['delete_filename'] = gr.Textbox(lines=1, label='File name')
        shared.gradio['delete_root'] = gr.Textbox(lines=1, label='File folder', info='For reference. Unchangeable.', interactive=False)
        with gr.Row():
            shared.gradio['delete_confirm'] = gr.Button('Delete', elem_classes="small-button", variant='stop')
            shared.gradio['delete_cancel'] = gr.Button('Cancel', elem_classes="small-button")

    # Character saver/deleter
    if shared.is_chat():
        with gr.Box(visible=False, elem_classes='file-saver') as shared.gradio['character_saver']:
            shared.gradio['save_character_filename'] = gr.Textbox(lines=1, label='File name', info='The character will be saved to your characters/ folder with this base filename.')
            with gr.Row():
                shared.gradio['save_character_confirm'] = gr.Button('Save', elem_classes="small-button")
                shared.gradio['save_character_cancel'] = gr.Button('Cancel', elem_classes="small-button")

        with gr.Box(visible=False, elem_classes='file-saver') as shared.gradio['character_deleter']:
            gr.Markdown('Confirm the character deletion?')
            with gr.Row():
                shared.gradio['delete_character_confirm'] = gr.Button('Delete', elem_classes="small-button", variant='stop')
                shared.gradio['delete_character_cancel'] = gr.Button('Cancel', elem_classes="small-button")


def create_file_saving_event_handlers():
    shared.gradio['save_confirm'].click(
        lambda x, y, z: utils.save_file(x + y, z), [shared.gradio[k] for k in ['save_root', 'save_filename', 'save_contents']], None).then(
        lambda: gr.update(visible=False), None, shared.gradio['file_saver'])

    shared.gradio['delete_confirm'].click(
        lambda x, y: utils.delete_file(x + y), [shared.gradio[k] for k in ['delete_root', 'delete_filename']], None).then(
        lambda: gr.update(visible=False), None, shared.gradio['file_deleter'])

    shared.gradio['delete_cancel'].click(lambda: gr.update(visible=False), None, shared.gradio['file_deleter'])
    shared.gradio['save_cancel'].click(lambda: gr.update(visible=False), None, shared.gradio['file_saver'])
    if shared.is_chat():
        shared.gradio['save_character_confirm'].click(
            chat.save_character, [shared.gradio[k] for k in ['name2', 'greeting', 'context', 'character_picture', 'save_character_filename']], None).then(
            lambda: gr.update(visible=False), None, shared.gradio['character_saver'])

        shared.gradio['delete_character_confirm'].click(
            chat.delete_character, shared.gradio['character_menu'], None).then(
            lambda: gr.update(visible=False), None, shared.gradio['character_deleter']).then(
            lambda: gr.update(choices=utils.get_available_characters()), outputs=shared.gradio['character_menu'])

        shared.gradio['save_character_cancel'].click(lambda: gr.update(visible=False), None, shared.gradio['character_saver'])
        shared.gradio['delete_character_cancel'].click(lambda: gr.update(visible=False), None, shared.gradio['character_deleter'])

    shared.gradio['save_preset'].click(
        ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
        presets.generate_preset_yaml, shared.gradio['interface_state'], shared.gradio['save_contents']).then(
        lambda: 'presets/', None, shared.gradio['save_root']).then(
        lambda: 'My Preset.yaml', None, shared.gradio['save_filename']).then(
        lambda: gr.update(visible=True), None, shared.gradio['file_saver'])

    shared.gradio['delete_preset'].click(
        lambda x: f'{x}.yaml', shared.gradio['preset_menu'], shared.gradio['delete_filename']).then(
        lambda: 'presets/', None, shared.gradio['delete_root']).then(
        lambda: gr.update(visible=True), None, shared.gradio['file_deleter'])


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
    default_preset = shared.settings['preset']
    default_text = load_prompt(shared.settings['prompt'])
    title = 'Text generation web UI'

    # Authentication variables
    auth = None
    gradio_auth_creds = []
    if shared.args.gradio_auth:
        gradio_auth_creds += [x.strip() for x in shared.args.gradio_auth.strip('"').replace('\n', '').split(',') if x.strip()]
    if shared.args.gradio_auth_path is not None:
        with open(shared.args.gradio_auth_path, 'r', encoding="utf8") as file:
            for line in file.readlines():
                gradio_auth_creds += [x.strip() for x in line.split(',') if x.strip()]
    if gradio_auth_creds:
        auth = [tuple(cred.split(':')) for cred in gradio_auth_creds]

    # Importing the extension files and executing their setup() functions
    if shared.args.extensions is not None and len(shared.args.extensions) > 0:
        extensions_module.load_extensions()

    # css/js strings
    css = ui.css if not shared.is_chat() else ui.css + ui.chat_css
    js = ui.main_js if not shared.is_chat() else ui.main_js + ui.chat_js
    css += apply_extensions('css')
    js += apply_extensions('js')

    with gr.Blocks(css=css, analytics_enabled=False, title=title, theme=ui.theme) as shared.gradio['interface']:
        if Path("notification.mp3").exists():
            shared.gradio['audio_notification'] = gr.Audio(interactive=False, value="notification.mp3", elem_id="audio_notification", visible=False)
            audio_notification_js = "document.querySelector('#audio_notification audio')?.play();"
        else:
            audio_notification_js = ""

        # Floating menus for saving/deleting files
        create_file_saving_menus()

        # Create chat mode interface
        if shared.is_chat():
            shared.input_elements = ui.list_interface_input_elements(chat=True)
            shared.gradio['interface_state'] = gr.State({k: None for k in shared.input_elements})
            shared.gradio['Chat input'] = gr.State()
            shared.gradio['dummy'] = gr.State()

            with gr.Tab('Text generation', elem_id='main'):
                shared.gradio['display'] = gr.HTML(value=chat_html_wrapper(shared.history['visible'], shared.settings['name1'], shared.settings['name2'], 'chat', 'cai-chat'))
                shared.gradio['textbox'] = gr.Textbox(label='Input')
                with gr.Row():
                    shared.gradio['Stop'] = gr.Button('Stop', elem_id='stop')
                    shared.gradio['Generate'] = gr.Button('Generate', elem_id='Generate', variant='primary')
                    shared.gradio['Continue'] = gr.Button('Continue')

                with gr.Row():
                    shared.gradio['Impersonate'] = gr.Button('Impersonate')
                    shared.gradio['Regenerate'] = gr.Button('Regenerate')
                    shared.gradio['Remove last'] = gr.Button('Remove last')

                with gr.Row():
                    shared.gradio['Copy last reply'] = gr.Button('Copy last reply')
                    shared.gradio['Replace last reply'] = gr.Button('Replace last reply')
                    shared.gradio['Send dummy message'] = gr.Button('Send dummy message')
                    shared.gradio['Send dummy reply'] = gr.Button('Send dummy reply')

                with gr.Row():
                    shared.gradio['Clear history'] = gr.Button('Clear history')
                    shared.gradio['Clear history-confirm'] = gr.Button('Confirm', variant='stop', visible=False)
                    shared.gradio['Clear history-cancel'] = gr.Button('Cancel', visible=False)

                with gr.Row():
                    shared.gradio['start_with'] = gr.Textbox(label='Start reply with', placeholder='Sure thing!', value=shared.settings['start_with'])

                with gr.Row():
                    shared.gradio['mode'] = gr.Radio(choices=['chat', 'chat-instruct', 'instruct'], value=shared.settings['mode'] if shared.settings['mode'] in ['chat', 'instruct', 'chat-instruct'] else 'chat', label='Mode', info='Defines how the chat prompt is generated. In instruct and chat-instruct modes, the instruction template selected under "Chat settings" must match the current model.')
                    shared.gradio['chat_style'] = gr.Dropdown(choices=utils.get_available_chat_styles(), label='Chat style', value=shared.settings['chat_style'], visible=shared.settings['mode'] != 'instruct')

            with gr.Tab('Chat settings', elem_id='chat-settings'):

                with gr.Tab("Character"):
                    with gr.Row():
                        with gr.Column(scale=8):
                            with gr.Row():
                                shared.gradio['character_menu'] = gr.Dropdown(choices=utils.get_available_characters(), label='Character', elem_id='character-menu', info='Used in chat and chat-instruct modes.', elem_classes='slim-dropdown')
                                ui.create_refresh_button(shared.gradio['character_menu'], lambda: None, lambda: {'choices': utils.get_available_characters()}, 'refresh-button')
                                shared.gradio['save_character'] = gr.Button('💾', elem_classes='refresh-button')
                                shared.gradio['delete_character'] = gr.Button('🗑️', elem_classes='refresh-button')

                            shared.gradio['name1'] = gr.Textbox(value=shared.settings['name1'], lines=1, label='Your name')
                            shared.gradio['name2'] = gr.Textbox(value=shared.settings['name2'], lines=1, label='Character\'s name')
                            shared.gradio['context'] = gr.Textbox(value=shared.settings['context'], lines=4, label='Context')
                            shared.gradio['greeting'] = gr.Textbox(value=shared.settings['greeting'], lines=4, label='Greeting')

                        with gr.Column(scale=1):
                            shared.gradio['character_picture'] = gr.Image(label='Character picture', type='pil')
                            shared.gradio['your_picture'] = gr.Image(label='Your picture', type='pil', value=Image.open(Path('cache/pfp_me.png')) if Path('cache/pfp_me.png').exists() else None)

                with gr.Tab("Instruction template"):
                    with gr.Row():
                        with gr.Row():
                            shared.gradio['instruction_template'] = gr.Dropdown(choices=utils.get_available_instruction_templates(), label='Instruction template', value='None', info='Change this according to the model/LoRA that you are using. Used in instruct and chat-instruct modes.', elem_classes='slim-dropdown')
                            ui.create_refresh_button(shared.gradio['instruction_template'], lambda: None, lambda: {'choices': utils.get_available_instruction_templates()}, 'refresh-button')
                            shared.gradio['save_template'] = gr.Button('💾', elem_classes='refresh-button')
                            shared.gradio['delete_template'] = gr.Button('🗑️ ', elem_classes='refresh-button')

                    shared.gradio['name1_instruct'] = gr.Textbox(value='', lines=2, label='User string')
                    shared.gradio['name2_instruct'] = gr.Textbox(value='', lines=1, label='Bot string')
                    shared.gradio['context_instruct'] = gr.Textbox(value='', lines=4, label='Context')
                    shared.gradio['turn_template'] = gr.Textbox(value=shared.settings['turn_template'], lines=1, label='Turn template', info='Used to precisely define the placement of spaces and new line characters in instruction prompts.')
                    with gr.Row():
                        shared.gradio['chat-instruct_command'] = gr.Textbox(value=shared.settings['chat-instruct_command'], lines=4, label='Command for chat-instruct mode', info='<|character|> gets replaced by the bot name, and <|prompt|> gets replaced by the regular chat prompt.')

                with gr.Tab('Chat history'):
                    with gr.Row():
                        with gr.Column():
                            shared.gradio['download'] = gr.File(label="Download")
                            shared.gradio['download_button'] = gr.Button(value='Refresh')

                        with gr.Column():
                            shared.gradio['upload_chat_history'] = gr.File(type='binary', file_types=['.json', '.txt'], label="Upload")

                with gr.Tab('Upload character'):
                    with gr.Tab('JSON'):
                        with gr.Row():
                            shared.gradio['upload_json'] = gr.File(type='binary', file_types=['.json'], label='JSON File')
                            shared.gradio['upload_img_bot'] = gr.Image(type='pil', label='Profile Picture (optional)')

                        shared.gradio['Upload character'] = gr.Button(value='Submit', interactive=False)

                    with gr.Tab('TavernAI'):
                        shared.gradio['upload_img_tavern'] = gr.File(type='binary', file_types=['image'], label='TavernAI PNG File')
                        shared.gradio['Upload tavern character'] = gr.Button(value='Submit', interactive=False)

            with gr.Tab("Parameters", elem_id="parameters"):
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
                            shared.gradio['markdown_render'] = gr.Button('Render')
                            shared.gradio['markdown'] = gr.Markdown()

                        with gr.Tab('HTML'):
                            shared.gradio['html'] = gr.HTML()

                        with gr.Row():
                            shared.gradio['Generate'] = gr.Button('Generate', variant='primary', elem_classes="small-button")
                            shared.gradio['Stop'] = gr.Button('Stop', elem_classes="small-button")
                            shared.gradio['Undo'] = gr.Button('Undo', elem_classes="small-button")
                            shared.gradio['Regenerate'] = gr.Button('Regenerate', elem_classes="small-button")

                    with gr.Column(scale=1):
                        gr.HTML('<div style="padding-bottom: 13px"></div>')
                        shared.gradio['max_new_tokens'] = gr.Slider(minimum=shared.settings['max_new_tokens_min'], maximum=shared.settings['max_new_tokens_max'], step=1, label='max_new_tokens', value=shared.settings['max_new_tokens'])
                        with gr.Row():
                            shared.gradio['prompt_menu'] = gr.Dropdown(choices=utils.get_available_prompts(), value='None', label='Prompt', elem_classes='slim-dropdown')
                            ui.create_refresh_button(shared.gradio['prompt_menu'], lambda: None, lambda: {'choices': utils.get_available_prompts()}, ['refresh-button', 'refresh-button-small'])
                            shared.gradio['save_prompt'] = gr.Button('💾', elem_classes=['refresh-button', 'refresh-button-small'])
                            shared.gradio['delete_prompt'] = gr.Button('🗑️', elem_classes=['refresh-button', 'refresh-button-small'])

                        shared.gradio['count_tokens'] = gr.Button('Count tokens')
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
                            shared.gradio['count_tokens'] = gr.Button('Count tokens')

                        with gr.Row():
                            shared.gradio['prompt_menu'] = gr.Dropdown(choices=utils.get_available_prompts(), value='None', label='Prompt', elem_classes='slim-dropdown')
                            ui.create_refresh_button(shared.gradio['prompt_menu'], lambda: None, lambda: {'choices': utils.get_available_prompts()}, 'refresh-button')
                            shared.gradio['save_prompt'] = gr.Button('💾', elem_classes='refresh-button')
                            shared.gradio['delete_prompt'] = gr.Button('🗑️', elem_classes='refresh-button')

                        shared.gradio['status'] = gr.Markdown('')

                    with gr.Column():
                        with gr.Tab('Raw'):
                            shared.gradio['output_textbox'] = gr.Textbox(elem_classes="textbox_default_output", lines=27, label='Output')

                        with gr.Tab('Markdown'):
                            shared.gradio['markdown_render'] = gr.Button('Render')
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
            modes = ["default", "notebook", "chat"]
            current_mode = "default"
            for mode in modes[1:]:
                if getattr(shared.args, mode):
                    current_mode = mode
                    break

            cmd_list = vars(shared.args)
            bool_list = sorted([k for k in cmd_list if type(cmd_list[k]) is bool and k not in modes + ui.list_model_elements()])
            bool_active = [k for k in bool_list if vars(shared.args)[k]]

            with gr.Row():
                shared.gradio['interface_modes_menu'] = gr.Dropdown(choices=modes, value=current_mode, label="Mode")
                shared.gradio['reset_interface'] = gr.Button("Apply and restart the interface", elem_classes="small-button")
                shared.gradio['toggle_dark_mode'] = gr.Button('Toggle dark/light mode', elem_classes="small-button")

            with gr.Row():
                with gr.Column():
                    shared.gradio['extensions_menu'] = gr.CheckboxGroup(choices=utils.get_available_extensions(), value=shared.args.extensions, label="Available extensions", info='Note that some of these extensions may require manually installing Python requirements through the command: pip install -r extensions/extension_name/requirements.txt', elem_classes='checkboxgroup-table')

                with gr.Column():
                    shared.gradio['bool_menu'] = gr.CheckboxGroup(choices=bool_list, value=bool_active, label="Boolean command-line flags", elem_classes='checkboxgroup-table')

            with gr.Row():
                extension_name = gr.Textbox(lines=1, label='Install or update an extension', info='Enter the GitHub URL below. For a list of extensions, see: https://github.com/oobabooga/text-generation-webui-extensions ⚠️  WARNING ⚠️ : extensions can execute arbitrary code. Make sure to inspect their source code before activating them.')
                extension_install = gr.Button('Install or update', elem_classes="small-button")

            extension_status = gr.Markdown()

            extension_install.click(
                clone_or_pull_repository, extension_name, extension_status, show_progress=False).then(
                lambda: gr.update(choices=utils.get_available_extensions(), value=shared.args.extensions), outputs=shared.gradio['extensions_menu'])

            # Reset interface event
            shared.gradio['reset_interface'].click(
                set_interface_arguments, [shared.gradio[k] for k in ['interface_modes_menu', 'extensions_menu', 'bool_menu']], None).then(
                lambda: None, None, None, _js='() => {document.body.innerHTML=\'<h1 style="font-family:monospace;margin-top:20%;color:lightgray;text-align:center;">Reloading...</h1>\'; setTimeout(function(){location.reload()},2500); return []}')

            shared.gradio['toggle_dark_mode'].click(lambda: None, None, None, _js='() => {document.getElementsByTagName("body")[0].classList.toggle("dark")}')

        # chat mode event handlers
        if shared.is_chat():
            shared.input_params = [shared.gradio[k] for k in ['Chat input', 'start_with', 'interface_state']]
            clear_arr = [shared.gradio[k] for k in ['Clear history-confirm', 'Clear history', 'Clear history-cancel']]
            shared.reload_inputs = [shared.gradio[k] for k in ['name1', 'name2', 'mode', 'chat_style']]

            gen_events.append(shared.gradio['Generate'].click(
                ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
                lambda x: (x, ''), shared.gradio['textbox'], [shared.gradio['Chat input'], shared.gradio['textbox']], show_progress=False).then(
                chat.generate_chat_reply_wrapper, shared.input_params, shared.gradio['display'], show_progress=False).then(
                chat.save_history, shared.gradio['mode'], None, show_progress=False).then(
                lambda: None, None, None, _js=f"() => {{{audio_notification_js}}}")
            )

            gen_events.append(shared.gradio['textbox'].submit(
                ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
                lambda x: (x, ''), shared.gradio['textbox'], [shared.gradio['Chat input'], shared.gradio['textbox']], show_progress=False).then(
                chat.generate_chat_reply_wrapper, shared.input_params, shared.gradio['display'], show_progress=False).then(
                chat.save_history, shared.gradio['mode'], None, show_progress=False).then(
                lambda: None, None, None, _js=f"() => {{{audio_notification_js}}}")
            )

            gen_events.append(shared.gradio['Regenerate'].click(
                ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
                partial(chat.generate_chat_reply_wrapper, regenerate=True), shared.input_params, shared.gradio['display'], show_progress=False).then(
                chat.save_history, shared.gradio['mode'], None, show_progress=False).then(
                lambda: None, None, None, _js=f"() => {{{audio_notification_js}}}")
            )

            gen_events.append(shared.gradio['Continue'].click(
                ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
                partial(chat.generate_chat_reply_wrapper, _continue=True), shared.input_params, shared.gradio['display'], show_progress=False).then(
                chat.save_history, shared.gradio['mode'], None, show_progress=False).then(
                lambda: None, None, None, _js=f"() => {{{audio_notification_js}}}")
            )

            gen_events.append(shared.gradio['Impersonate'].click(
                ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
                lambda x: x, shared.gradio['textbox'], shared.gradio['Chat input'], show_progress=False).then(
                chat.impersonate_wrapper, shared.input_params, shared.gradio['textbox'], show_progress=False).then(
                lambda: None, None, None, _js=f"() => {{{audio_notification_js}}}")
            )

            shared.gradio['Replace last reply'].click(
                chat.replace_last_reply, shared.gradio['textbox'], None).then(
                lambda: '', None, shared.gradio['textbox'], show_progress=False).then(
                chat.save_history, shared.gradio['mode'], None, show_progress=False).then(
                chat.redraw_html, shared.reload_inputs, shared.gradio['display'])

            shared.gradio['Send dummy message'].click(
                chat.send_dummy_message, shared.gradio['textbox'], None).then(
                lambda: '', None, shared.gradio['textbox'], show_progress=False).then(
                chat.save_history, shared.gradio['mode'], None, show_progress=False).then(
                chat.redraw_html, shared.reload_inputs, shared.gradio['display'])

            shared.gradio['Send dummy reply'].click(
                chat.send_dummy_reply, shared.gradio['textbox'], None).then(
                lambda: '', None, shared.gradio['textbox'], show_progress=False).then(
                chat.save_history, shared.gradio['mode'], None, show_progress=False).then(
                chat.redraw_html, shared.reload_inputs, shared.gradio['display'])

            shared.gradio['Clear history-confirm'].click(
                lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)], None, clear_arr).then(
                chat.clear_chat_log, [shared.gradio[k] for k in ['greeting', 'mode']], None).then(
                chat.save_history, shared.gradio['mode'], None, show_progress=False).then(
                chat.redraw_html, shared.reload_inputs, shared.gradio['display'])

            shared.gradio['Stop'].click(
                stop_everything_event, None, None, queue=False, cancels=gen_events if shared.args.no_stream else None).then(
                chat.redraw_html, shared.reload_inputs, shared.gradio['display'])

            shared.gradio['mode'].change(
                lambda x: gr.update(visible=x != 'instruct'), shared.gradio['mode'], shared.gradio['chat_style'], show_progress=False).then(
                chat.redraw_html, shared.reload_inputs, shared.gradio['display'])

            shared.gradio['chat_style'].change(chat.redraw_html, shared.reload_inputs, shared.gradio['display'])
            shared.gradio['instruction_template'].change(
                partial(chat.load_character, instruct=True), [shared.gradio[k] for k in ['instruction_template', 'name1_instruct', 'name2_instruct']], [shared.gradio[k] for k in ['name1_instruct', 'name2_instruct', 'dummy', 'dummy', 'context_instruct', 'turn_template']])

            shared.gradio['upload_chat_history'].upload(
                chat.load_history, [shared.gradio[k] for k in ['upload_chat_history', 'name1', 'name2']], None).then(
                chat.redraw_html, shared.reload_inputs, shared.gradio['display'])

            shared.gradio['Copy last reply'].click(chat.send_last_reply_to_input, None, shared.gradio['textbox'], show_progress=False)
            shared.gradio['Clear history'].click(lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)], None, clear_arr)
            shared.gradio['Clear history-cancel'].click(lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)], None, clear_arr)
            shared.gradio['Remove last'].click(
                chat.remove_last_message, None, shared.gradio['textbox'], show_progress=False).then(
                chat.save_history, shared.gradio['mode'], None, show_progress=False).then(
                chat.redraw_html, shared.reload_inputs, shared.gradio['display'])

            # Save/delete a character
            shared.gradio['save_character'].click(
                lambda x: x, shared.gradio['name2'], shared.gradio['save_character_filename']).then(
                lambda: gr.update(visible=True), None, shared.gradio['character_saver'])

            shared.gradio['delete_character'].click(lambda: gr.update(visible=True), None, shared.gradio['character_deleter'])

            shared.gradio['save_template'].click(
                lambda: 'My Template.yaml', None, shared.gradio['save_filename']).then(
                lambda: 'characters/instruction-following/', None, shared.gradio['save_root']).then(
                chat.generate_instruction_template_yaml, [shared.gradio[k] for k in ['name1_instruct', 'name2_instruct', 'context_instruct', 'turn_template']], shared.gradio['save_contents']).then(
                lambda: gr.update(visible=True), None, shared.gradio['file_saver'])

            shared.gradio['delete_template'].click(
                lambda x: f'{x}.yaml', shared.gradio['instruction_template'], shared.gradio['delete_filename']).then(
                lambda: 'characters/instruction-following/', None, shared.gradio['delete_root']).then(
                lambda: gr.update(visible=True), None, shared.gradio['file_deleter'])

            shared.gradio['download_button'].click(lambda x: chat.save_history(x, timestamp=True, user_request=True), shared.gradio['mode'], shared.gradio['download'])
            shared.gradio['Upload character'].click(chat.upload_character, [shared.gradio['upload_json'], shared.gradio['upload_img_bot']], [shared.gradio['character_menu']])
            shared.gradio['upload_json'].upload(lambda: gr.update(interactive=True), None, [shared.gradio['Upload character']])
            shared.gradio['upload_json'].clear(lambda: gr.update(interactive=False), None, [shared.gradio['Upload character']])

            shared.gradio['character_menu'].change(
                partial(chat.load_character, instruct=False), [shared.gradio[k] for k in ['character_menu', 'name1', 'name2']], [shared.gradio[k] for k in ['name1', 'name2', 'character_picture', 'greeting', 'context', 'dummy']]).then(
                chat.redraw_html, shared.reload_inputs, shared.gradio['display'])

            shared.gradio['Upload tavern character'].click(chat.upload_tavern_character, [shared.gradio['upload_img_tavern'], shared.gradio['name1'], shared.gradio['name2']], [shared.gradio['character_menu']])
            shared.gradio['upload_img_tavern'].upload(lambda: gr.update(interactive=True), None, [shared.gradio['Upload tavern character']])
            shared.gradio['upload_img_tavern'].clear(lambda: gr.update(interactive=False), None, [shared.gradio['Upload tavern character']])
            shared.gradio['your_picture'].change(
                chat.upload_your_profile_picture, shared.gradio['your_picture'], None).then(
                partial(chat.redraw_html, reset_cache=True), shared.reload_inputs, shared.gradio['display'])

        # notebook/default modes event handlers
        else:
            shared.input_params = [shared.gradio[k] for k in ['textbox', 'interface_state']]
            if shared.args.notebook:
                output_params = [shared.gradio[k] for k in ['textbox', 'html']]
            else:
                output_params = [shared.gradio[k] for k in ['output_textbox', 'html']]

            gen_events.append(shared.gradio['Generate'].click(
                lambda x: x, shared.gradio['textbox'], shared.gradio['last_input']).then(
                ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
                generate_reply_wrapper, shared.input_params, output_params, show_progress=False).then(
                lambda: None, None, None, _js=f"() => {{{audio_notification_js}}}")
                # lambda: None, None, None, _js="() => {element = document.getElementsByTagName('textarea')[0]; element.scrollTop = element.scrollHeight}")
            )

            gen_events.append(shared.gradio['textbox'].submit(
                lambda x: x, shared.gradio['textbox'], shared.gradio['last_input']).then(
                ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
                generate_reply_wrapper, shared.input_params, output_params, show_progress=False).then(
                lambda: None, None, None, _js=f"() => {{{audio_notification_js}}}")
                # lambda: None, None, None, _js="() => {element = document.getElementsByTagName('textarea')[0]; element.scrollTop = element.scrollHeight}")
            )

            if shared.args.notebook:
                shared.gradio['Undo'].click(lambda x: x, shared.gradio['last_input'], shared.gradio['textbox'], show_progress=False)
                shared.gradio['markdown_render'].click(lambda x: x, shared.gradio['textbox'], shared.gradio['markdown'], queue=False)
                gen_events.append(shared.gradio['Regenerate'].click(
                    lambda x: x, shared.gradio['last_input'], shared.gradio['textbox'], show_progress=False).then(
                    ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
                    generate_reply_wrapper, shared.input_params, output_params, show_progress=False).then(
                    lambda: None, None, None, _js=f"() => {{{audio_notification_js}}}")
                    # lambda: None, None, None, _js="() => {element = document.getElementsByTagName('textarea')[0]; element.scrollTop = element.scrollHeight}")
                )
            else:
                shared.gradio['markdown_render'].click(lambda x: x, shared.gradio['output_textbox'], shared.gradio['markdown'], queue=False)
                gen_events.append(shared.gradio['Continue'].click(
                    ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
                    generate_reply_wrapper, [shared.gradio['output_textbox']] + shared.input_params[1:], output_params, show_progress=False).then(
                    lambda: None, None, None, _js=f"() => {{{audio_notification_js}}}")
                    # lambda: None, None, None, _js="() => {element = document.getElementsByTagName('textarea')[1]; element.scrollTop = element.scrollHeight}")
                )

            shared.gradio['Stop'].click(stop_everything_event, None, None, queue=False, cancels=gen_events if shared.args.no_stream else None)
            shared.gradio['prompt_menu'].change(load_prompt, shared.gradio['prompt_menu'], shared.gradio['textbox'], show_progress=False)
            shared.gradio['save_prompt'].click(
                lambda x: x, shared.gradio['textbox'], shared.gradio['save_contents']).then(
                lambda: 'prompts/', None, shared.gradio['save_root']).then(
                lambda: utils.current_time() + '.txt', None, shared.gradio['save_filename']).then(
                lambda: gr.update(visible=True), None, shared.gradio['file_saver'])

            shared.gradio['delete_prompt'].click(
                lambda: 'prompts/', None, shared.gradio['delete_root']).then(
                lambda x: x + '.txt', shared.gradio['prompt_menu'], shared.gradio['delete_filename']).then(
                lambda: gr.update(visible=True), None, shared.gradio['file_deleter'])

            shared.gradio['count_tokens'].click(count_tokens, shared.gradio['textbox'], shared.gradio['status'], show_progress=False)

        create_file_saving_event_handlers()
        shared.gradio['interface'].load(lambda: None, None, None, _js=f"() => {{{js}}}")
        if shared.settings['dark_theme']:
            shared.gradio['interface'].load(lambda: None, None, None, _js="() => document.getElementsByTagName('body')[0].classList.add('dark')")

        shared.gradio['interface'].load(partial(ui.apply_interface_values, {}, use_persistent=True), None, [shared.gradio[k] for k in ui.list_interface_input_elements(chat=shared.is_chat())], show_progress=False)

        # Extensions tabs
        extensions_module.create_extensions_tabs()

        # Extensions block
        extensions_module.create_extensions_block()

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
    elif Path('settings.yaml').exists():
        settings_file = Path('settings.yaml')
    elif Path('settings.json').exists():
        settings_file = Path('settings.json')

    if settings_file is not None:
        logger.info(f"Loading settings from {settings_file}...")
        file_contents = open(settings_file, 'r', encoding='utf-8').read()
        new_settings = json.loads(file_contents) if settings_file.suffix == "json" else yaml.safe_load(file_contents)
        for item in new_settings:
            shared.settings[item] = new_settings[item]

    # Set default model settings based on settings file
    shared.model_config['.*'] = {
        'wbits': 'None',
        'model_type': 'None',
        'groupsize': 'None',
        'pre_layer': 0,
        'mode': shared.settings['mode'],
        'skip_special_tokens': shared.settings['skip_special_tokens'],
        'custom_stopping_strings': shared.settings['custom_stopping_strings'],
        'truncation_length': shared.settings['truncation_length'],
    }

    shared.model_config.move_to_end('.*', last=False)  # Move to the beginning

    # Default extensions
    extensions_module.available_extensions = utils.get_available_extensions()
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

    available_models = utils.get_available_models()

    # Model defined through --model
    if shared.args.model is not None:
        shared.model_name = shared.args.model

    # Only one model is available
    elif len(available_models) == 1:
        shared.model_name = available_models[0]

    # Select the model from a command-line menu
    elif shared.args.model_menu:
        if len(available_models) == 0:
            logger.error('No models are available! Please download at least one.')
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
        model_settings = get_model_settings_from_yamls(shared.model_name)
        shared.settings.update(model_settings)  # hijacking the interface defaults
        update_model_parameters(model_settings, initial=True)  # hijacking the command-line arguments

        # Load the model
        shared.model, shared.tokenizer = load_model(shared.model_name)
        if shared.args.lora:
            add_lora_to_model(shared.args.lora)

    # Force a character to be loaded
    if shared.is_chat():
        shared.persistent_interface_state.update({
            'mode': shared.settings['mode'],
            'character_menu': shared.args.character or shared.settings['character'],
            'instruction_template': shared.settings['instruction_template']
        })

    shared.persistent_interface_state.update({
        'loader': shared.args.loader or 'Transformers',
    })

    shared.generation_lock = Lock()
    # Launch the web UI
    create_interface()
    while True:
        time.sleep(0.5)
        if shared.need_restart:
            shared.need_restart = False
            time.sleep(0.5)
            shared.gradio['interface'].close()
            time.sleep(0.5)
            create_interface()
