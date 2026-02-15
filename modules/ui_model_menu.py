import importlib
import math
import queue
import threading
import traceback
from functools import partial
from pathlib import Path

import gradio as gr

from modules import loaders, shared, ui, utils
from modules.logging_colors import logger
from modules.LoRA import add_lora_to_model
from modules.models import load_model, unload_model
from modules.models_settings import (
    apply_model_settings_to_state,
    get_model_metadata,
    save_instruction_template,
    save_model_settings,
    update_gpu_layers_and_vram,
    update_model_parameters
)
from modules.utils import gradio
from modules.i18n import t


def create_ui():
    mu = shared.args.multi_user

    with gr.Tab(t("Model"), elem_id="model-tab"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    shared.gradio['model_menu'] = gr.Dropdown(choices=utils.get_available_models(), value=lambda: shared.model_name, label=t('Model'), elem_classes='slim-dropdown', interactive=not mu)
                    ui.create_refresh_button(shared.gradio['model_menu'], lambda: None, lambda: {'choices': utils.get_available_models()}, 'refresh-button', interactive=not mu)
                    shared.gradio['load_model'] = gr.Button(t("Load"), elem_classes='refresh-button', interactive=not mu)
                    shared.gradio['unload_model'] = gr.Button(t("Unload"), elem_classes='refresh-button', interactive=not mu)
                    shared.gradio['save_model_settings'] = gr.Button(t("Save settings"), elem_classes='refresh-button', interactive=not mu)

                shared.gradio['loader'] = gr.Dropdown(label=t("Model loader"), choices=loaders.loaders_and_params.keys() if not shared.args.portable else ['llama.cpp'], value=None)
                with gr.Blocks():
                    gr.Markdown(t("## Main options"))
                    with gr.Row():
                        with gr.Column():
                            shared.gradio['gpu_layers'] = gr.Slider(label=t("gpu-layers"), minimum=-1, maximum=get_initial_gpu_layers_max(), step=1, value=shared.args.gpu_layers, info=t('Number of layers to offload to the GPU. -1 = auto.'))
                            shared.gradio['ctx_size'] = gr.Slider(label=t('ctx-size'), minimum=0, maximum=1048576, step=1024, value=shared.args.ctx_size, info=t('Context length. 0 = auto for llama.cpp (requires gpu-layers=-1), 8192 for other loaders. Common values: 4096, 8192, 16384, 32768, 65536, 131072.'))
                            shared.gradio['gpu_split'] = gr.Textbox(label=t('gpu-split'), info=t('Comma-separated list of VRAM (in GB) to use per GPU. Example: 20,7,7'))
                            shared.gradio['attn_implementation'] = gr.Dropdown(label=t("attn-implementation"), choices=['sdpa', 'eager', 'flash_attention_2'], value=shared.args.attn_implementation, info=t('Attention implementation.'))
                            shared.gradio['cache_type'] = gr.Dropdown(label=t("cache-type"), choices=['fp16', 'q8_0', 'q4_0', 'fp8', 'q8', 'q7', 'q6', 'q5', 'q4', 'q3', 'q2'], value=shared.args.cache_type, allow_custom_value=True, info=t('Valid options: llama.cpp - fp16, q8_0, q4_0; ExLlamaV3 - fp16, q2 to q8. For ExLlamaV3, you can type custom combinations for separate k/v bits (e.g. q4_q8).'))
                            shared.gradio['fit_target'] = gr.Textbox(label=t('fit-target'), value=shared.args.fit_target, info=t('Target VRAM margin per device for auto GPU layers (MiB). Comma-separated list for multiple devices.'))
                            shared.gradio['tp_backend'] = gr.Dropdown(label=t("tp-backend"), choices=['native', 'nccl'], value=shared.args.tp_backend, info=t('The backend for tensor parallelism.'))

                        with gr.Column():
                            shared.gradio['vram_info'] = gr.HTML(value=get_initial_vram_info())
                            if not shared.args.portable:
                                shared.gradio['ik'] = gr.Checkbox(label=t("ik"), value=shared.args.ik, info=t('Use ik_llama.cpp instead of upstream llama.cpp.'))

                            shared.gradio['cpu_moe'] = gr.Checkbox(label=t("cpu-moe"), value=shared.args.cpu_moe, info=t('Move the experts to the CPU. Saves VRAM on MoE models.'))
                            shared.gradio['streaming_llm'] = gr.Checkbox(label=t("streaming-llm"), value=shared.args.streaming_llm, info=t('Activate StreamingLLM to avoid re-evaluating the entire prompt when old messages are removed.'))
                            shared.gradio['load_in_8bit'] = gr.Checkbox(label="load-in-8bit", value=shared.args.load_in_8bit)
                            shared.gradio['load_in_4bit'] = gr.Checkbox(label="load-in-4bit", value=shared.args.load_in_4bit)
                            shared.gradio['use_double_quant'] = gr.Checkbox(label="use_double_quant", value=shared.args.use_double_quant, info=t('Used by load-in-4bit.'))
                            shared.gradio['enable_tp'] = gr.Checkbox(label="enable_tp", value=shared.args.enable_tp, info=t('Enable tensor parallelism (TP).'))
                            shared.gradio['tensorrt_llm_info'] = gr.Markdown(
                                t('* TensorRT-LLM has to be installed manually: `pip install tensorrt_llm==1.1.0 --extra-index-url https://pypi.nvidia.com`.\n\n')
                                + t('* You can load either a pre-built TensorRT engine or a regular HF model. ')
                                + t('HF models will be compiled to a TensorRT engine automatically on each load (this can take a while).')
                            )

                            # Multimodal
                            with gr.Accordion(t("Multimodal (vision)"), open=False, elem_classes='tgw-accordion') as shared.gradio['mmproj_accordion']:
                                with gr.Row():
                                    shared.gradio['mmproj'] = gr.Dropdown(label=t("mmproj file"), choices=utils.get_available_mmproj(), value=lambda: shared.args.mmproj or 'None', elem_classes='slim-dropdown', info=f"{t('Select a file that matches your model. Must be placed in')} {shared.user_data_dir}/mmproj/", interactive=not mu)
                                    ui.create_refresh_button(shared.gradio['mmproj'], lambda: None, lambda: {'choices': utils.get_available_mmproj()}, 'refresh-button', interactive=not mu)

                            # Speculative decoding
                            with gr.Accordion(t("Speculative decoding"), open=False, elem_classes='tgw-accordion') as shared.gradio['speculative_decoding_accordion']:
                                shared.gradio['draft_max'] = gr.Number(label=t("draft-max"), precision=0, step=1, value=shared.args.draft_max, info=t('Maximum number of tokens to draft for speculative decoding. Recommended: 4 for draft model, 64 for n-gram.'))

                                gr.Markdown(t('#### Draft model'))
                                with gr.Row():
                                    shared.gradio['model_draft'] = gr.Dropdown(label=t("model-draft"), choices=['None'] + utils.get_available_models(), value=lambda: shared.args.model_draft, elem_classes='slim-dropdown', info=t('Draft model. Must share the same vocabulary as the main model.'), interactive=not mu)
                                    ui.create_refresh_button(shared.gradio['model_draft'], lambda: None, lambda: {'choices': ['None'] + utils.get_available_models()}, 'refresh-button', interactive=not mu)

                                shared.gradio['gpu_layers_draft'] = gr.Slider(label=t("gpu-layers-draft"), minimum=0, maximum=256, value=shared.args.gpu_layers_draft, info=t('Number of layers to offload to the GPU for the draft model.'))
                                shared.gradio['device_draft'] = gr.Textbox(label=t("device-draft"), value=shared.args.device_draft, info=t('Comma-separated list of devices to use for offloading the draft model. Example: CUDA0,CUDA1'))
                                shared.gradio['ctx_size_draft'] = gr.Number(label=t("ctx-size-draft"), precision=0, step=256, value=shared.args.ctx_size_draft, info=t('Size of the prompt context for the draft model. If 0, uses the same as the main model.'))

                                shared.gradio['ngram_header'] = gr.Markdown(t('#### N-gram (draftless)'))
                                shared.gradio['spec_type'] = gr.Dropdown(label=t("spec-type"), choices=['none', 'ngram-mod', 'ngram-simple', 'ngram-map-k', 'ngram-map-k4v', 'ngram-cache'], value=shared.args.spec_type, info=t('Draftless speculative decoding type. Recommended: ngram-mod.'))
                                shared.gradio['spec_ngram_size_n'] = gr.Number(label=t("spec-ngram-size-n"), precision=0, step=1, value=shared.args.spec_ngram_size_n, info=t('N-gram lookup size for speculative decoding.'), visible=shared.args.spec_type != 'none')
                                shared.gradio['spec_ngram_size_m'] = gr.Number(label=t("spec-ngram-size-m"), precision=0, step=1, value=shared.args.spec_ngram_size_m, info=t('Draft n-gram size for speculative decoding.'), visible=shared.args.spec_type != 'none')
                                shared.gradio['spec_ngram_min_hits'] = gr.Number(label=t("spec-ngram-min-hits"), precision=0, step=1, value=shared.args.spec_ngram_min_hits, info=t('Minimum n-gram hits for ngram-map speculative decoding.'), visible=shared.args.spec_type != 'none')

                    gr.Markdown(t("## Other options"))
                    with gr.Accordion(t("See more options"), open=False, elem_classes='tgw-accordion'):
                        with gr.Row():
                            with gr.Column():
                                shared.gradio['parallel'] = gr.Slider(label=t("parallel"), minimum=1, step=1, maximum=64, value=shared.args.parallel, info=t('Number of parallel request slots for the API. The context size is divided equally among slots. For example, to have 4 slots with 8192 context each, set ctx_size to 32768.'))
                                shared.gradio['threads'] = gr.Slider(label=t("threads"), minimum=0, step=1, maximum=256, value=shared.args.threads)
                                shared.gradio['threads_batch'] = gr.Slider(label=t("threads_batch"), minimum=0, step=1, maximum=256, value=shared.args.threads_batch)
                                shared.gradio['batch_size'] = gr.Slider(label=t("batch_size"), minimum=1, maximum=4096, step=1, value=shared.args.batch_size)
                                shared.gradio['ubatch_size'] = gr.Slider(label=t("ubatch_size"), minimum=1, maximum=4096, step=1, value=shared.args.ubatch_size)
                                shared.gradio['tensor_split'] = gr.Textbox(label=t('tensor_split'), info=t('List of proportions to split the model across multiple GPUs. Example: 60,40'))
                                shared.gradio['extra_flags'] = gr.Textbox(label=t('extra-flags'), info=t('Extra flags to pass to llama-server. Example: --jinja --rpc 192.168.1.100:50052'), value=shared.args.extra_flags)
                                shared.gradio['cpu_memory'] = gr.Number(label=t("Maximum CPU memory in GiB. Use this for CPU offloading."), value=shared.args.cpu_memory)
                                shared.gradio['alpha_value'] = gr.Number(label='alpha_value', value=shared.args.alpha_value, precision=2, info=t('Positional embeddings alpha factor for NTK RoPE scaling. Recommended values (NTKv1): 1.75 for 1.5x context, 2.5 for 2x context. Use either this or compress_pos_emb, not both.'))
                                shared.gradio['rope_freq_base'] = gr.Number(label=t('rope_freq_base'), value=shared.args.rope_freq_base, precision=0, info=t('Positional embeddings frequency base for NTK RoPE scaling. Related to alpha_value by rope_freq_base = 10000 * alpha_value ^ (64 / 63). 0 = from model.'))
                                shared.gradio['compress_pos_emb'] = gr.Number(label=t('compress_pos_emb'), value=shared.args.compress_pos_emb, precision=2, info=t("Positional embeddings compression factor. Should be set to (context length) / (model's original context length). Equal to 1/rope_freq_scale."))
                                shared.gradio['compute_dtype'] = gr.Dropdown(label=t("compute_dtype"), choices=["bfloat16", "float16", "float32"], value=shared.args.compute_dtype, info=t('Used by load-in-4bit.'))
                                shared.gradio['quant_type'] = gr.Dropdown(label=t("quant_type"), choices=["nf4", "fp4"], value=shared.args.quant_type, info=t('Used by load-in-4bit.'))
                                shared.gradio['num_experts_per_token'] = gr.Number(label=t("Number of experts per token"), value=shared.args.num_experts_per_token, info=t('Only applies to MoE models like Mixtral.'))

                            with gr.Column():
                                shared.gradio['cpu'] = gr.Checkbox(label=t("cpu"), value=shared.args.cpu, info=t('Use PyTorch in CPU mode.'))
                                shared.gradio['disk'] = gr.Checkbox(label=t("disk"), value=shared.args.disk)
                                shared.gradio['row_split'] = gr.Checkbox(label=t("row_split"), value=shared.args.row_split, info=t('Split the model by rows across GPUs. This may improve multi-gpu performance.'))
                                shared.gradio['no_kv_offload'] = gr.Checkbox(label=t("no_kv_offload"), value=shared.args.no_kv_offload, info=t('Do not offload the K, Q, V to the GPU. This saves VRAM but reduces performance.'))
                                shared.gradio['no_mmap'] = gr.Checkbox(label=t("no-mmap"), value=shared.args.no_mmap)
                                shared.gradio['mlock'] = gr.Checkbox(label=t("mlock"), value=shared.args.mlock)
                                shared.gradio['numa'] = gr.Checkbox(label=t("numa"), value=shared.args.numa, info=t('NUMA support can help on some systems with non-uniform memory access.'))
                                shared.gradio['bf16'] = gr.Checkbox(label=t("bf16"), value=shared.args.bf16)
                                shared.gradio['no_flash_attn'] = gr.Checkbox(label=t("no_flash_attn"), value=shared.args.no_flash_attn)
                                shared.gradio['no_xformers'] = gr.Checkbox(label=t("no_xformers"), value=shared.args.no_xformers)
                                shared.gradio['no_sdpa'] = gr.Checkbox(label=t("no_sdpa"), value=shared.args.no_sdpa)
                                shared.gradio['cfg_cache'] = gr.Checkbox(label=t("cfg-cache"), value=shared.args.cfg_cache, info=t('Necessary to use CFG with this loader.'))
                                shared.gradio['no_use_fast'] = gr.Checkbox(label="no_use_fast", value=shared.args.no_use_fast, info=t('Set use_fast=False while loading the tokenizer.'))
                                if not shared.args.portable:
                                    with gr.Row():
                                        shared.gradio['lora_menu'] = gr.Dropdown(multiselect=True, choices=utils.get_available_loras(), value=shared.lora_names, label=t('LoRA(s)'), elem_classes='slim-dropdown', interactive=not mu)
                                        ui.create_refresh_button(shared.gradio['lora_menu'], lambda: None, lambda: {'choices': utils.get_available_loras(), 'value': shared.lora_names}, 'refresh-button', interactive=not mu)
                                        shared.gradio['lora_menu_apply'] = gr.Button(value=t('Apply LoRAs'), elem_classes='refresh-button', interactive=not mu)

            with gr.Column():
                with gr.Tab(t("Download")):
                    shared.gradio['custom_model_menu'] = gr.Textbox(label=t("Download model or LoRA"), info=t("Enter the Hugging Face username/model path, for instance: facebook/galactica-125m. To specify a branch, add it at the end after a \":\" character like this: facebook/galactica-125m:main. To download a single file, enter its name in the second box."), interactive=not mu)
                    shared.gradio['download_specific_file'] = gr.Textbox(placeholder=t("File name (for GGUF models)"), show_label=False, max_lines=1, interactive=not mu)
                    with gr.Row():
                        shared.gradio['download_model_button'] = gr.Button(t("Download"), variant='primary', interactive=not mu)
                        shared.gradio['get_file_list'] = gr.Button(t("Get file list"), interactive=not mu)

                with gr.Tab(t("Customize instruction template")):
                    with gr.Row():
                        shared.gradio['customized_template'] = gr.Dropdown(choices=utils.get_available_instruction_templates(), value='None', label=t('Select the desired instruction template'), elem_classes='slim-dropdown')
                        ui.create_refresh_button(shared.gradio['customized_template'], lambda: None, lambda: {'choices': utils.get_available_instruction_templates()}, 'refresh-button', interactive=not mu)

                    shared.gradio['customized_template_submit'] = gr.Button(t("Submit"), variant="primary", interactive=not mu)
                    gr.Markdown(t("This allows you to set a customized template for the model currently selected in the \"Model loader\" menu. Whenever the model gets loaded, this template will be used in place of the template specified in the model's metadata, which sometimes is wrong."))

                with gr.Row():
                    shared.gradio['model_status'] = gr.Markdown(t('No model is loaded') if shared.model_name == 'None' else t('Ready'))


def create_event_handlers():
    mu = shared.args.multi_user
    if mu:
        return

    shared.gradio['loader'].change(loaders.make_loader_params_visible, gradio('loader'), gradio(loaders.get_all_params()), show_progress=False)

    # In this event handler, the interface state is read and updated
    # with the model defaults (if any), and then the model is loaded
    shared.gradio['model_menu'].change(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        handle_load_model_event_initial, gradio('model_menu', 'interface_state'), gradio(ui.list_interface_input_elements()) + gradio('interface_state') + gradio('vram_info'), show_progress=False).then(
        partial(load_model_wrapper, autoload=False), gradio('model_menu', 'loader'), gradio('model_status'), show_progress=True).success(
        handle_load_model_event_final, gradio('truncation_length', 'loader', 'interface_state'), gradio('truncation_length', 'filter_by_loader'), show_progress=False)

    shared.gradio['load_model'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        update_model_parameters, gradio('interface_state'), None).then(
        partial(load_model_wrapper, autoload=True), gradio('model_menu', 'loader'), gradio('model_status'), show_progress=True).success(
        handle_load_model_event_final, gradio('truncation_length', 'loader', 'interface_state'), gradio('truncation_length', 'filter_by_loader'), show_progress=False)

    shared.gradio['unload_model'].click(handle_unload_model_click, None, gradio('model_status'), show_progress=False).then(
        update_gpu_layers_and_vram, gradio('loader', 'model_menu', 'gpu_layers', 'ctx_size', 'cache_type'), gradio('vram_info'), show_progress=False)

    shared.gradio['save_model_settings'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        save_model_settings, gradio('model_menu', 'interface_state'), gradio('model_status'), show_progress=False)

    # For ctx_size and cache_type - update VRAM display
    for param in ['ctx_size', 'cache_type']:
        shared.gradio[param].change(
            update_gpu_layers_and_vram,
            gradio('loader', 'model_menu', 'gpu_layers', 'ctx_size', 'cache_type'),
            gradio('vram_info'), show_progress=False)

    # For manual gpu_layers changes - only update VRAM
    shared.gradio['gpu_layers'].change(
        update_gpu_layers_and_vram,
        gradio('loader', 'model_menu', 'gpu_layers', 'ctx_size', 'cache_type'),
        gradio('vram_info'), show_progress=False)

    if not shared.args.portable:
        shared.gradio['lora_menu_apply'].click(load_lora_wrapper, gradio('lora_menu'), gradio('model_status'), show_progress=False)

    shared.gradio['spec_type'].change(
        lambda x: [gr.update(visible=x != 'none')] * 3,
        gradio('spec_type'),
        gradio('spec_ngram_size_n', 'spec_ngram_size_m', 'spec_ngram_min_hits'),
        show_progress=False
    )

    shared.gradio['download_model_button'].click(download_model_wrapper, gradio('custom_model_menu', 'download_specific_file'), gradio('model_status'), show_progress=True)
    shared.gradio['get_file_list'].click(partial(download_model_wrapper, return_links=True), gradio('custom_model_menu', 'download_specific_file'), gradio('model_status'), show_progress=True)
    shared.gradio['customized_template_submit'].click(save_instruction_template, gradio('model_menu', 'customized_template'), gradio('model_status'), show_progress=True)


def load_model_wrapper(selected_model, loader, autoload=False):
    try:
        settings = get_model_metadata(selected_model)
    except FileNotFoundError:
        exc = traceback.format_exc()
        yield exc.replace('\n', '\n\n')
        return

    if not autoload:
        yield "### {}\n\n- Settings updated: Click \"Load\" to load the model\n- Max sequence length: {}".format(selected_model, settings['truncation_length_info'])
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
                yield f"Successfully loaded `{selected_model}`."
            else:
                yield f"Failed to load `{selected_model}`."
        except Exception:
            logger.exception('Failed to load the model.')
            yield traceback.format_exc().replace('\n', '\n\n')


def load_lora_wrapper(selected_loras):
    yield ("Applying the following LoRAs to {}:\n\n{}".format(shared.model_name, '\n'.join(selected_loras)))
    add_lora_to_model(selected_loras)
    yield ("Successfully applied the LoRAs")


def download_model_wrapper(repo_id, specific_file, progress=gr.Progress(), return_links=False, check=False):
    downloader_module = importlib.import_module("download-model")
    downloader = downloader_module.ModelDownloader()
    update_queue = queue.Queue()

    try:
        # Handle direct GGUF URLs
        if repo_id.startswith("https://") and ("huggingface.co" in repo_id) and (repo_id.endswith(".gguf") or repo_id.endswith(".gguf?download=true")):
            try:
                path = repo_id.split("huggingface.co/")[1]
                parts = path.split("/")
                if len(parts) >= 2:
                    extracted_repo_id = f"{parts[0]}/{parts[1]}"
                    filename = repo_id.split("/")[-1].replace("?download=true", "")
                    repo_id = extracted_repo_id
                    specific_file = filename
            except Exception as e:
                yield f"Error parsing GGUF URL: {e}"
                progress(0.0)
                return

        if not repo_id:
            yield t("Please enter a model path.")
            progress(0.0)
            return

        repo_id = repo_id.strip()
        specific_file = specific_file.strip()

        progress(0.0, "Preparing download...")

        model, branch = downloader.sanitize_model_and_branch_names(repo_id, None)
        yield "Getting download links from Hugging Face..."
        links, sha256, is_lora, is_llamacpp, file_sizes = downloader.get_download_links_from_huggingface(model, branch, text_only=False, specific_file=specific_file)

        if not links:
            yield "No files found to download for the given model/criteria."
            progress(0.0)
            return

        # Check for multiple GGUF files
        gguf_files = [link for link in links if link.lower().endswith('.gguf')]
        if len(gguf_files) > 1 and not specific_file:
            # Sort by size in ascending order
            gguf_data = []
            for i, link in enumerate(links):
                if link.lower().endswith('.gguf'):
                    file_size = file_sizes[i]
                    gguf_data.append((file_size, link))

            gguf_data.sort(key=lambda x: x[0])

            output = "Multiple GGUF files found. Please copy one of the following filenames to the 'File name' field above:\n\n```\n"
            for file_size, link in gguf_data:
                size_str = format_file_size(file_size)
                output += f"{size_str} - {Path(link).name}\n"

            output += "```"
            yield output
            return

        if return_links:
            # Sort by size in ascending order
            file_data = list(zip(file_sizes, links))
            file_data.sort(key=lambda x: x[0])

            output = "```\n"
            for file_size, link in file_data:
                size_str = format_file_size(file_size)
                output += f"{size_str} - {Path(link).name}\n"

            output += "```"
            yield output
            return

        yield "Determining output folder..."
        output_folder = downloader.get_output_folder(
            model, branch, is_lora, is_llamacpp=is_llamacpp,
            model_dir=shared.args.model_dir if shared.args.model_dir != shared.args_defaults.model_dir else None
        )

        if output_folder == shared.user_data_dir / "models":
            output_folder = Path(shared.args.model_dir)
        elif output_folder == shared.user_data_dir / "loras":
            output_folder = Path(shared.args.lora_dir)

        if check:
            yield "Checking previously downloaded files..."
            progress(0.5, "Verifying files...")
            downloader.check_model_files(model, branch, links, sha256, output_folder)
            progress(1.0, "Verification complete.")
            yield "File check complete."
            return

        yield ""
        progress(0.0, "Download starting...")

        def downloader_thread_target():
            try:
                downloader.download_model_files(
                    model, branch, links, sha256, output_folder,
                    progress_queue=update_queue,
                    threads=4,
                    is_llamacpp=is_llamacpp,
                    specific_file=specific_file
                )
                update_queue.put(("COMPLETED", f"Model successfully saved to `{output_folder}/`."))
            except Exception:
                tb_str = traceback.format_exc().replace('\n', '\n\n')
                update_queue.put(("ERROR", tb_str))

        download_thread = threading.Thread(target=downloader_thread_target)
        download_thread.start()

        while True:
            try:
                message = update_queue.get(timeout=0.2)
                if not isinstance(message, tuple) or len(message) != 2:
                    continue

                msg_identifier, data = message

                if msg_identifier == "COMPLETED":
                    progress(1.0, "Download complete!")
                    yield data
                    break
                elif msg_identifier == "ERROR":
                    progress(0.0, "Error occurred")
                    yield data
                    break
                elif isinstance(msg_identifier, float):
                    progress_value = msg_identifier
                    description_str = data
                    progress(progress_value, f"Downloading: {description_str}")

            except queue.Empty:
                if not download_thread.is_alive():
                    yield "Download process finished."
                    break

        download_thread.join()

    except Exception:
        progress(0.0)
        tb_str = traceback.format_exc().replace('\n', '\n\n')
        yield tb_str


def update_truncation_length(current_length, state):
    if 'loader' in state:
        if state['loader'].lower().startswith('exllama') or state['loader'] == 'llama.cpp':
            if state['ctx_size'] > 0:
                return state['ctx_size']

            # ctx_size == 0 means auto: use the actual value from the server
            return shared.settings['truncation_length']

    return current_length


def get_initial_vram_info():
    if shared.model_name != 'None' and shared.args.loader == 'llama.cpp':
        return update_gpu_layers_and_vram(
            shared.args.loader,
            shared.model_name,
            shared.args.gpu_layers,
            shared.args.ctx_size,
            shared.args.cache_type,
        )

    return "<div id=\"vram-info\"'>Estimated VRAM to load the model:</div>"


def get_initial_gpu_layers_max():
    if shared.model_name != 'None' and shared.args.loader == 'llama.cpp':
        model_settings = get_model_metadata(shared.model_name)
        return model_settings.get('max_gpu_layers', 256)

    return 256


def handle_load_model_event_initial(model, state):
    state = apply_model_settings_to_state(model, state)
    output = ui.apply_interface_values(state)
    update_model_parameters(state)  # This updates the command-line flags

    vram_info = state.get('vram_info', "<div id=\"vram-info\"'>Estimated VRAM to load the model:</div>")
    return output + [state] + [vram_info]


def handle_load_model_event_final(truncation_length, loader, state):
    truncation_length = update_truncation_length(truncation_length, state)
    return [truncation_length, loader]


def handle_unload_model_click():
    unload_model()
    return "Model unloaded"


def format_file_size(size_bytes):
    """Convert bytes to human readable format with 2 decimal places for GB and above"""
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = size_bytes / p

    if i >= 3:  # GB or TB
        return f"{s:.2f} {size_names[i]}"
    else:
        return f"{s:.1f} {size_names[i]}"
