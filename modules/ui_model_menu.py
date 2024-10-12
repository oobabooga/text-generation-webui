import importlib
import math
import re
import traceback
from functools import partial
from pathlib import Path

import gradio as gr
import psutil
import torch
from transformers import is_torch_npu_available, is_torch_xpu_available

from modules import loaders, shared, ui, utils
from modules.logging_colors import logger
from modules.LoRA import add_lora_to_model
from modules.models import load_model, unload_model
from modules.models_settings import (
    apply_model_settings_to_state,
    get_model_metadata,
    save_instruction_template,
    save_model_settings,
    update_model_parameters
)
from modules.utils import gradio


def create_ui():
    mu = shared.args.multi_user

    # Finding the default values for the GPU and CPU memories
    total_mem = []
    if is_torch_xpu_available():
        for i in range(torch.xpu.device_count()):
            total_mem.append(math.floor(torch.xpu.get_device_properties(i).total_memory / (1024 * 1024)))
    elif is_torch_npu_available():
        for i in range(torch.npu.device_count()):
            total_mem.append(math.floor(torch.npu.get_device_properties(i).total_memory / (1024 * 1024)))
    else:
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

    with gr.Tab("Model", elem_id="model-tab"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            shared.gradio['model_menu'] = gr.Dropdown(choices=utils.get_available_models(), value=lambda: shared.model_name, label='Model', elem_classes='slim-dropdown', interactive=not mu)
                            ui.create_refresh_button(shared.gradio['model_menu'], lambda: None, lambda: {'choices': utils.get_available_models()}, 'refresh-button', interactive=not mu)
                            shared.gradio['load_model'] = gr.Button("Load", visible=not shared.settings['autoload_model'], elem_classes='refresh-button', interactive=not mu)
                            shared.gradio['unload_model'] = gr.Button("Unload", elem_classes='refresh-button', interactive=not mu)
                            shared.gradio['save_model_settings'] = gr.Button("Save settings", elem_classes='refresh-button', interactive=not mu)

                    with gr.Column():
                        with gr.Row():
                            shared.gradio['lora_menu'] = gr.Dropdown(multiselect=True, choices=utils.get_available_loras(), value=shared.lora_names, label='LoRA(s)', elem_classes='slim-dropdown', interactive=not mu)
                            ui.create_refresh_button(shared.gradio['lora_menu'], lambda: None, lambda: {'choices': utils.get_available_loras(), 'value': shared.lora_names}, 'refresh-button', interactive=not mu)
                            shared.gradio['lora_menu_apply'] = gr.Button(value='Apply LoRAs', elem_classes='refresh-button', interactive=not mu)

        with gr.Row():
            with gr.Column():
                shared.gradio['loader'] = gr.Dropdown(label="Model loader", choices=loaders.loaders_and_params.keys(), value=None)
                with gr.Blocks():
                    with gr.Row():
                        with gr.Column():
                            with gr.Blocks():
                                for i in range(len(total_mem)):
                                    shared.gradio[f'gpu_memory_{i}'] = gr.Slider(label=f"gpu-memory in MiB for device :{i}", maximum=total_mem[i], value=default_gpu_mem[i])

                                shared.gradio['cpu_memory'] = gr.Slider(label="cpu-memory in MiB", maximum=total_cpu_mem, value=default_cpu_mem)

                            with gr.Blocks():
                                shared.gradio['transformers_info'] = gr.Markdown('load-in-4bit params:')
                                shared.gradio['compute_dtype'] = gr.Dropdown(label="compute_dtype", choices=["bfloat16", "float16", "float32"], value=shared.args.compute_dtype)
                                shared.gradio['quant_type'] = gr.Dropdown(label="quant_type", choices=["nf4", "fp4"], value=shared.args.quant_type)

                            shared.gradio['hqq_backend'] = gr.Dropdown(label="hqq_backend", choices=["PYTORCH", "PYTORCH_COMPILE", "ATEN"], value=shared.args.hqq_backend)
                            shared.gradio['n_gpu_layers'] = gr.Slider(label="n-gpu-layers", minimum=0, maximum=256, value=shared.args.n_gpu_layers, info='Must be set to more than 0 for your GPU to be used.')
                            shared.gradio['n_ctx'] = gr.Number(label="n_ctx", precision=0, step=256, value=shared.args.n_ctx, info='Context length. Try lowering this if you run out of memory while loading the model.')
                            shared.gradio['tensor_split'] = gr.Textbox(label='tensor_split', info='List of proportions to split the model across multiple GPUs. Example: 60,40')
                            shared.gradio['n_batch'] = gr.Slider(label="n_batch", minimum=1, maximum=2048, step=1, value=shared.args.n_batch)
                            shared.gradio['threads'] = gr.Slider(label="threads", minimum=0, step=1, maximum=256, value=shared.args.threads)
                            shared.gradio['threads_batch'] = gr.Slider(label="threads_batch", minimum=0, step=1, maximum=256, value=shared.args.threads_batch)
                            shared.gradio['wbits'] = gr.Dropdown(label="wbits", choices=["None", 1, 2, 3, 4, 8], value=shared.args.wbits if shared.args.wbits > 0 else "None")
                            shared.gradio['groupsize'] = gr.Dropdown(label="groupsize", choices=["None", 32, 64, 128, 1024], value=shared.args.groupsize if shared.args.groupsize > 0 else "None")
                            shared.gradio['gpu_split'] = gr.Textbox(label='gpu-split', info='Comma-separated list of VRAM (in GB) to use per GPU. Example: 20,7,7')
                            shared.gradio['max_seq_len'] = gr.Number(label='max_seq_len', precision=0, step=256, value=shared.args.max_seq_len, info='Context length. Try lowering this if you run out of memory while loading the model.')
                            with gr.Blocks():
                                shared.gradio['alpha_value'] = gr.Number(label='alpha_value', value=shared.args.alpha_value, precision=2, info='Positional embeddings alpha factor for NTK RoPE scaling. Recommended values (NTKv1): 1.75 for 1.5x context, 2.5 for 2x context. Use either this or compress_pos_emb, not both.')
                                shared.gradio['rope_freq_base'] = gr.Number(label='rope_freq_base', value=shared.args.rope_freq_base, precision=0, info='Positional embeddings frequency base for NTK RoPE scaling. Related to alpha_value by rope_freq_base = 10000 * alpha_value ^ (64 / 63). 0 = from model.')
                                shared.gradio['compress_pos_emb'] = gr.Number(label='compress_pos_emb', value=shared.args.compress_pos_emb, precision=2, info='Positional embeddings compression factor. Should be set to (context length) / (model\'s original context length). Equal to 1/rope_freq_scale.')

                            shared.gradio['autogptq_info'] = gr.Markdown('ExLlamav2_HF is recommended over AutoGPTQ for models derived from Llama.')

                        with gr.Column():
                            shared.gradio['load_in_8bit'] = gr.Checkbox(label="load-in-8bit", value=shared.args.load_in_8bit)
                            shared.gradio['load_in_4bit'] = gr.Checkbox(label="load-in-4bit", value=shared.args.load_in_4bit)
                            shared.gradio['use_double_quant'] = gr.Checkbox(label="use_double_quant", value=shared.args.use_double_quant)
                            shared.gradio['use_flash_attention_2'] = gr.Checkbox(label="use_flash_attention_2", value=shared.args.use_flash_attention_2, info='Set use_flash_attention_2=True while loading the model.')
                            shared.gradio['use_eager_attention'] = gr.Checkbox(label="use_eager_attention", value=shared.args.use_eager_attention, info='Set attn_implementation= eager while loading the model.')
                            shared.gradio['flash_attn'] = gr.Checkbox(label="flash_attn", value=shared.args.flash_attn, info='Use flash-attention.')
                            shared.gradio['auto_devices'] = gr.Checkbox(label="auto-devices", value=shared.args.auto_devices)
                            shared.gradio['tensorcores'] = gr.Checkbox(label="tensorcores", value=shared.args.tensorcores, info='NVIDIA only: use llama-cpp-python compiled with tensor cores support. This may increase performance on newer cards.')
                            shared.gradio['cache_8bit'] = gr.Checkbox(label="cache_8bit", value=shared.args.cache_8bit, info='Use 8-bit cache to save VRAM.')
                            shared.gradio['cache_4bit'] = gr.Checkbox(label="cache_4bit", value=shared.args.cache_4bit, info='Use Q4 cache to save VRAM.')
                            shared.gradio['streaming_llm'] = gr.Checkbox(label="streaming_llm", value=shared.args.streaming_llm, info='(experimental) Activate StreamingLLM to avoid re-evaluating the entire prompt when old messages are removed.')
                            shared.gradio['attention_sink_size'] = gr.Number(label="attention_sink_size", value=shared.args.attention_sink_size, precision=0, info='StreamingLLM: number of sink tokens. Only used if the trimmed prompt doesn\'t share a prefix with the old prompt.')
                            shared.gradio['cpu'] = gr.Checkbox(label="cpu", value=shared.args.cpu, info='llama.cpp: Use llama-cpp-python compiled without GPU acceleration. Transformers: use PyTorch in CPU mode.')
                            shared.gradio['row_split'] = gr.Checkbox(label="row_split", value=shared.args.row_split, info='Split the model by rows across GPUs. This may improve multi-gpu performance.')
                            shared.gradio['no_offload_kqv'] = gr.Checkbox(label="no_offload_kqv", value=shared.args.no_offload_kqv, info='Do not offload the  K, Q, V to the GPU. This saves VRAM but reduces the performance.')
                            shared.gradio['no_mul_mat_q'] = gr.Checkbox(label="no_mul_mat_q", value=shared.args.no_mul_mat_q, info='Disable the mulmat kernels.')
                            shared.gradio['triton'] = gr.Checkbox(label="triton", value=shared.args.triton)
                            shared.gradio['no_inject_fused_mlp'] = gr.Checkbox(label="no_inject_fused_mlp", value=shared.args.no_inject_fused_mlp, info='Affects Triton only. Disable fused MLP. Fused MLP improves performance but uses more VRAM. Disable if running low on VRAM.')
                            shared.gradio['no_use_cuda_fp16'] = gr.Checkbox(label="no_use_cuda_fp16", value=shared.args.no_use_cuda_fp16, info='This can make models faster on some systems.')
                            shared.gradio['desc_act'] = gr.Checkbox(label="desc_act", value=shared.args.desc_act, info='\'desc_act\', \'wbits\', and \'groupsize\' are used for old models without a quantize_config.json.')
                            shared.gradio['no_mmap'] = gr.Checkbox(label="no-mmap", value=shared.args.no_mmap)
                            shared.gradio['mlock'] = gr.Checkbox(label="mlock", value=shared.args.mlock)
                            shared.gradio['numa'] = gr.Checkbox(label="numa", value=shared.args.numa, info='NUMA support can help on some systems with non-uniform memory access.')
                            shared.gradio['disk'] = gr.Checkbox(label="disk", value=shared.args.disk)
                            shared.gradio['bf16'] = gr.Checkbox(label="bf16", value=shared.args.bf16)
                            shared.gradio['autosplit'] = gr.Checkbox(label="autosplit", value=shared.args.autosplit, info='Automatically split the model tensors across the available GPUs.')
                            shared.gradio['enable_tp'] = gr.Checkbox(label="enable_tp", value=shared.args.enable_tp, info='Enable Tensor Parallelism (TP).')
                            shared.gradio['no_flash_attn'] = gr.Checkbox(label="no_flash_attn", value=shared.args.no_flash_attn)
                            shared.gradio['no_xformers'] = gr.Checkbox(label="no_xformers", value=shared.args.no_xformers)
                            shared.gradio['no_sdpa'] = gr.Checkbox(label="no_sdpa", value=shared.args.no_sdpa)
                            shared.gradio['cfg_cache'] = gr.Checkbox(label="cfg-cache", value=shared.args.cfg_cache, info='Necessary to use CFG with this loader.')
                            shared.gradio['cpp_runner'] = gr.Checkbox(label="cpp-runner", value=shared.args.cpp_runner, info='Enable inference with ModelRunnerCpp, which is faster than the default ModelRunner.')
                            shared.gradio['num_experts_per_token'] = gr.Number(label="Number of experts per token", value=shared.args.num_experts_per_token, info='Only applies to MoE models like Mixtral.')
                            with gr.Blocks():
                                shared.gradio['trust_remote_code'] = gr.Checkbox(label="trust-remote-code", value=shared.args.trust_remote_code, info='Set trust_remote_code=True while loading the tokenizer/model. To enable this option, start the web UI with the --trust-remote-code flag.', interactive=shared.args.trust_remote_code)
                                shared.gradio['no_use_fast'] = gr.Checkbox(label="no_use_fast", value=shared.args.no_use_fast, info='Set use_fast=False while loading the tokenizer.')
                                shared.gradio['logits_all'] = gr.Checkbox(label="logits_all", value=shared.args.logits_all, info='Needs to be set for perplexity evaluation to work with this loader. Otherwise, ignore it, as it makes prompt processing slower.')

                            shared.gradio['disable_exllama'] = gr.Checkbox(label="disable_exllama", value=shared.args.disable_exllama, info='Disable ExLlama kernel for GPTQ models.')
                            shared.gradio['disable_exllamav2'] = gr.Checkbox(label="disable_exllamav2", value=shared.args.disable_exllamav2, info='Disable ExLlamav2 kernel for GPTQ models.')
                            shared.gradio['exllamav2_info'] = gr.Markdown("ExLlamav2_HF is recommended over ExLlamav2 for better integration with extensions and more consistent sampling behavior across loaders.")
                            shared.gradio['llamacpp_HF_info'] = gr.Markdown("llamacpp_HF loads llama.cpp as a Transformers model. To use it, you need to place your GGUF in a subfolder of models/ with the necessary tokenizer files.\n\nYou can use the \"llamacpp_HF creator\" menu to do that automatically.")
                            shared.gradio['tensorrt_llm_info'] = gr.Markdown('* TensorRT-LLM has to be installed manually in a separate Python 3.10 environment at the moment. For a guide, consult the description of [this PR](https://github.com/oobabooga/text-generation-webui/pull/5715). \n\n* `max_seq_len` is only used when `cpp-runner` is checked.\n\n* `cpp_runner` does not support streaming at the moment.')

            with gr.Column():
                with gr.Row():
                    shared.gradio['autoload_model'] = gr.Checkbox(value=shared.settings['autoload_model'], label='Autoload the model', info='Whether to load the model as soon as it is selected in the Model dropdown.', interactive=not mu)

                with gr.Tab("Download"):
                    shared.gradio['custom_model_menu'] = gr.Textbox(label="Download model or LoRA", info="Enter the Hugging Face username/model path, for instance: facebook/galactica-125m. To specify a branch, add it at the end after a \":\" character like this: facebook/galactica-125m:main. To download a single file, enter its name in the second box.", interactive=not mu)
                    shared.gradio['download_specific_file'] = gr.Textbox(placeholder="File name (for GGUF models)", show_label=False, max_lines=1, interactive=not mu)
                    with gr.Row():
                        shared.gradio['download_model_button'] = gr.Button("Download", variant='primary', interactive=not mu)
                        shared.gradio['get_file_list'] = gr.Button("Get file list", interactive=not mu)

                with gr.Tab("llamacpp_HF creator"):
                    with gr.Row():
                        shared.gradio['gguf_menu'] = gr.Dropdown(choices=utils.get_available_ggufs(), value=lambda: shared.model_name, label='Choose your GGUF', elem_classes='slim-dropdown', interactive=not mu)
                        ui.create_refresh_button(shared.gradio['gguf_menu'], lambda: None, lambda: {'choices': utils.get_available_ggufs()}, 'refresh-button', interactive=not mu)

                    shared.gradio['unquantized_url'] = gr.Textbox(label="Enter the URL for the original (unquantized) model", info="Example: https://huggingface.co/lmsys/vicuna-13b-v1.5", max_lines=1)
                    shared.gradio['create_llamacpp_hf_button'] = gr.Button("Submit", variant="primary", interactive=not mu)
                    gr.Markdown("This will move your gguf file into a subfolder of `models` along with the necessary tokenizer files.")

                with gr.Tab("Customize instruction template"):
                    with gr.Row():
                        shared.gradio['customized_template'] = gr.Dropdown(choices=utils.get_available_instruction_templates(), value='None', label='Select the desired instruction template', elem_classes='slim-dropdown')
                        ui.create_refresh_button(shared.gradio['customized_template'], lambda: None, lambda: {'choices': utils.get_available_instruction_templates()}, 'refresh-button', interactive=not mu)

                    shared.gradio['customized_template_submit'] = gr.Button("Submit", variant="primary", interactive=not mu)
                    gr.Markdown("This allows you to set a customized template for the model currently selected in the \"Model loader\" menu. Whenever the model gets loaded, this template will be used in place of the template specified in the model's medatada, which sometimes is wrong.")

                with gr.Row():
                    shared.gradio['model_status'] = gr.Markdown('No model is loaded' if shared.model_name == 'None' else 'Ready')


def create_event_handlers():
    shared.gradio['loader'].change(loaders.make_loader_params_visible, gradio('loader'), gradio(loaders.get_all_params()), show_progress=False)

    # In this event handler, the interface state is read and updated
    # with the model defaults (if any), and then the model is loaded
    # unless "autoload_model" is unchecked
    shared.gradio['model_menu'].change(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        handle_load_model_event_initial, gradio('model_menu', 'interface_state'), gradio(ui.list_interface_input_elements()) + gradio('interface_state'), show_progress=False).then(
        load_model_wrapper, gradio('model_menu', 'loader', 'autoload_model'), gradio('model_status'), show_progress=False).success(
        handle_load_model_event_final, gradio('truncation_length', 'loader', 'interface_state'), gradio('truncation_length', 'filter_by_loader'), show_progress=False)

    shared.gradio['load_model'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        update_model_parameters, gradio('interface_state'), None).then(
        partial(load_model_wrapper, autoload=True), gradio('model_menu', 'loader'), gradio('model_status'), show_progress=False).success(
        handle_load_model_event_final, gradio('truncation_length', 'loader', 'interface_state'), gradio('truncation_length', 'filter_by_loader'), show_progress=False)

    shared.gradio['unload_model'].click(handle_unload_model_click, None, gradio('model_status'), show_progress=False)
    shared.gradio['save_model_settings'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        save_model_settings, gradio('model_menu', 'interface_state'), gradio('model_status'), show_progress=False)

    shared.gradio['lora_menu_apply'].click(load_lora_wrapper, gradio('lora_menu'), gradio('model_status'), show_progress=False)
    shared.gradio['download_model_button'].click(download_model_wrapper, gradio('custom_model_menu', 'download_specific_file'), gradio('model_status'), show_progress=True)
    shared.gradio['get_file_list'].click(partial(download_model_wrapper, return_links=True), gradio('custom_model_menu', 'download_specific_file'), gradio('model_status'), show_progress=True)
    shared.gradio['autoload_model'].change(lambda x: gr.update(visible=not x), gradio('autoload_model'), gradio('load_model'))
    shared.gradio['create_llamacpp_hf_button'].click(create_llamacpp_hf, gradio('gguf_menu', 'unquantized_url'), gradio('model_status'), show_progress=True)
    shared.gradio['customized_template_submit'].click(save_instruction_template, gradio('model_menu', 'customized_template'), gradio('model_status'), show_progress=True)


def load_model_wrapper(selected_model, loader, autoload=False):
    if not autoload:
        yield f"The settings for `{selected_model}` have been updated.\n\nClick on \"Load\" to load it."
        return

    if selected_model == 'None':
        yield "No model selected"
    else:
        try:
            yield f"Loading `{selected_model}`..."
            unload_model()
            if selected_model != '':
                shared.model, shared.tokenizer = load_model(selected_model, loader)

            if shared.model is not None:
                output = f"Successfully loaded `{selected_model}`."

                settings = get_model_metadata(selected_model)
                if 'instruction_template' in settings:
                    output += '\n\nIt seems to be an instruction-following model with template "{}". In the chat tab, instruct or chat-instruct modes should be used.'.format(settings['instruction_template'])

                yield output
            else:
                yield f"Failed to load `{selected_model}`."
        except:
            exc = traceback.format_exc()
            logger.error('Failed to load the model.')
            print(exc)
            yield exc.replace('\n', '\n\n')


def load_lora_wrapper(selected_loras):
    yield ("Applying the following LoRAs to {}:\n\n{}".format(shared.model_name, '\n'.join(selected_loras)))
    add_lora_to_model(selected_loras)
    yield ("Successfuly applied the LoRAs")


def download_model_wrapper(repo_id, specific_file, progress=gr.Progress(), return_links=False, check=False):
    try:
        if repo_id == "":
            yield ("Please enter a model path")
            return

        downloader = importlib.import_module("download-model").ModelDownloader()

        progress(0.0)
        model, branch = downloader.sanitize_model_and_branch_names(repo_id, None)

        yield ("Getting the download links from Hugging Face")
        links, sha256, is_lora, is_llamacpp = downloader.get_download_links_from_huggingface(model, branch, text_only=False, specific_file=specific_file)
        if return_links:
            output = "```\n"
            for link in links:
                output += f"{Path(link).name}" + "\n"

            output += "```"
            yield output
            return

        yield ("Getting the output folder")
        output_folder = downloader.get_output_folder(
            model,
            branch,
            is_lora,
            is_llamacpp=is_llamacpp,
            model_dir=shared.args.model_dir if shared.args.model_dir != shared.args_defaults.model_dir else None
        )

        if output_folder == Path("models"):
            output_folder = Path(shared.args.model_dir)
        elif output_folder == Path("loras"):
            output_folder = Path(shared.args.lora_dir)

        if check:
            progress(0.5)

            yield ("Checking previously downloaded files")
            downloader.check_model_files(model, branch, links, sha256, output_folder)
            progress(1.0)
        else:
            yield (f"Downloading file{'s' if len(links) > 1 else ''} to `{output_folder}`")
            downloader.download_model_files(model, branch, links, sha256, output_folder, progress_bar=progress, threads=4, is_llamacpp=is_llamacpp)

            yield (f"Model successfully saved to `{output_folder}/`.")
    except:
        progress(1.0)
        yield traceback.format_exc().replace('\n', '\n\n')


def create_llamacpp_hf(gguf_name, unquantized_url, progress=gr.Progress()):
    try:
        downloader = importlib.import_module("download-model").ModelDownloader()

        progress(0.0)
        model, branch = downloader.sanitize_model_and_branch_names(unquantized_url, None)

        yield ("Getting the tokenizer files links from Hugging Face")
        links, sha256, is_lora, is_llamacpp = downloader.get_download_links_from_huggingface(model, branch, text_only=True)
        output_folder = Path(shared.args.model_dir) / (re.sub(r'(?i)\.gguf$', '', gguf_name) + "-HF")

        yield (f"Downloading tokenizer to `{output_folder}`")
        downloader.download_model_files(model, branch, links, sha256, output_folder, progress_bar=progress, threads=4, is_llamacpp=False)

        # Move the GGUF
        (Path(shared.args.model_dir) / gguf_name).rename(output_folder / gguf_name)

        yield (f"Model saved to `{output_folder}/`.\n\nYou can now load it using llamacpp_HF.")
    except:
        progress(1.0)
        yield traceback.format_exc().replace('\n', '\n\n')


def update_truncation_length(current_length, state):
    if 'loader' in state:
        if state['loader'].lower().startswith('exllama'):
            return state['max_seq_len']
        elif state['loader'] in ['llama.cpp', 'llamacpp_HF']:
            return state['n_ctx']

    return current_length


def handle_load_model_event_initial(model, state):
    state = apply_model_settings_to_state(model, state)
    output = ui.apply_interface_values(state)
    update_model_parameters(state)
    return output + [state]


def handle_load_model_event_final(truncation_length, loader, state):
    truncation_length = update_truncation_length(truncation_length, state)
    return [truncation_length, loader]


def handle_unload_model_click():
    unload_model()
    return "Model unloaded"
