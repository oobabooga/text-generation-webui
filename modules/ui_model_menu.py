import importlib
import math
import re
import traceback
from functools import partial
from pathlib import Path

import gradio as gr
import psutil
import torch
from transformers import is_torch_xpu_available

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

    with gr.Tab("模型", elem_id="model-tab"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            shared.gradio['model_menu'] = gr.Dropdown(choices=utils.get_available_models(), value=lambda: shared.model_name, label='模型', elem_classes='slim-dropdown', interactive=not mu)
                            ui.create_refresh_button(shared.gradio['model_menu'], lambda: None, lambda: {'choices': utils.get_available_models()}, '刷新按钮', interactive=not mu)
                            shared.gradio['load_model'] = gr.Button("加载", visible=not shared.settings['autoload_model'], elem_classes='刷新按钮', interactive=not mu)
                            shared.gradio['unload_model'] = gr.Button("卸载", elem_classes='刷新按钮', interactive=not mu)
                            shared.gradio['reload_model'] = gr.Button("重载", elem_classes='刷新按钮', interactive=not mu)
                            shared.gradio['save_model_settings'] = gr.Button("保存设置", elem_classes='刷新按钮', interactive=not mu)

                    with gr.Column():
                        with gr.Row():
                            shared.gradio['lora_menu'] = gr.Dropdown(multiselect=True, choices=utils.get_available_loras(), value=shared.lora_names, label='LoRA(s)', elem_classes='slim-dropdown', interactive=not mu)
                            ui.create_refresh_button(shared.gradio['lora_menu'], lambda: None, lambda: {'choices': utils.get_available_loras(), 'value': shared.lora_names}, '刷新按钮', interactive=not mu)
                            shared.gradio['lora_menu_apply'] = gr.Button(value='应用LoRAs', elem_classes='刷新按钮', interactive=not mu)

        with gr.Row():
            with gr.Column():
                shared.gradio['loader'] = gr.Dropdown(label="模型加载器", choices=loaders.loaders_and_params.keys(), value=None)
                with gr.Box():
                    with gr.Row():
                        with gr.Column():
                            with gr.Blocks():
                                for i in range(len(total_mem)):
                                    shared.gradio[f'gpu_memory_{i}'] = gr.Slider(label=f"GPU内存（MiB）设备：{i}", maximum=total_mem[i], value=default_gpu_mem[i])

                                shared.gradio['cpu_memory'] = gr.Slider(label="CPU内存（MiB）", maximum=total_cpu_mem, value=default_cpu_mem)

                            with gr.Blocks():
                                shared.gradio['transformers_info'] = gr.Markdown('加载4比特参数：')
                                shared.gradio['compute_dtype'] = gr.Dropdown(label="计算数据类型", choices=["bfloat16", "float16", "float32"], value=shared.args.compute_dtype)
                                shared.gradio['quant_type'] = gr.Dropdown(label="量化类型", choices=["nf4", "fp4"], value=shared.args.quant_type)

                            shared.gradio['hqq_backend'] = gr.Dropdown(label="hqq后端", choices=["PYTORCH", "PYTORCH_COMPILE", "ATEN"], value=shared.args.hqq_backend)
                            shared.gradio['n_gpu_layers'] = gr.Slider(label="GPU层数", minimum=0, maximum=256, value=shared.args.n_gpu_layers)
                            shared.gradio['n_ctx'] = gr.Slider(minimum=0, maximum=shared.settings['truncation_length_max'], step=256, label="n_ctx", value=shared.args.n_ctx, info='上下文长度。如果在加载模型时内存不足，请尝试降低此值。')
                            shared.gradio['tensor_split'] = gr.Textbox(label='张量分割', info='将模型分割到多个GPU的比例列表。示例：18,17')
                            shared.gradio['n_batch'] = gr.Slider(label="批处理大小", minimum=1, maximum=2048, step=1, value=shared.args.n_batch)
                            shared.gradio['threads'] = gr.Slider(label="线程", minimum=0, step=1, maximum=32, value=shared.args.threads)
                            shared.gradio['threads_batch'] = gr.Slider(label="批处理线程", minimum=0, step=1, maximum=32, value=shared.args.threads_batch)
                            shared.gradio['wbits'] = gr.Dropdown(label="权重位", choices=["None", 1, 2, 3, 4, 8], value=shared.args.wbits if shared.args.wbits > 0 else "None")
                            shared.gradio['groupsize'] = gr.Dropdown(label="组大小", choices=["None", 32, 64, 128, 1024], value=shared.args.groupsize if shared.args.groupsize > 0 else "None")
                            shared.gradio['model_type'] = gr.Dropdown(label="模型类型", choices=["None"], value=shared.args.model_type or "None")
                            shared.gradio['pre_layer'] = gr.Slider(label="预处理层", minimum=0, maximum=100, value=shared.args.pre_layer[0] if shared.args.pre_layer is not None else 0)
                            shared.gradio['gpu_split'] = gr.Textbox(label='GPU分割', info='以逗号分隔的每个GPU使用的VRAM（以GB为单位）列表。示例：20,7,7')
                            shared.gradio['max_seq_len'] = gr.Slider(label='最大序列长度', minimum=0, maximum=shared.settings['truncation_length_max'], step=256, info='上下文长度。如果在加载模型时内存不足，请尝试降低此值。', value=shared.args.max_seq_len)
                            with gr.Blocks():
                                shared.gradio['alpha_value'] = gr.Slider(label='alpha值', minimum=1, maximum=8, step=0.05, info='NTK RoPE缩放的位置嵌入alpha因子。推荐值（NTKv1）：1.5倍上下文长度用1.75，2倍上下文长度用2.5。使用此项或压缩位置嵌入，不要同时使用。', value=shared.args.alpha_value)
                                shared.gradio['rope_freq_base'] = gr.Slider(label='rope频率基数', minimum=0, maximum=1000000, step=1000, info='如果大于0，将代替alpha值使用。这两者之间的关系是rope_freq_base = 10000 * alpha值 ^ (64 / 63)', value=shared.args.rope_freq_base)
                                shared.gradio['compress_pos_emb'] = gr.Slider(label='压缩位置嵌入', minimum=1, maximum=8, step=1, info='位置嵌入的压缩因子。应设置为（上下文长度）/（模型原始上下文长度）。等于1/rope_freq_scale。', value=shared.args.compress_pos_emb)

                            shared.gradio['autogptq_info'] = gr.Markdown('推荐使用ExLlamav2_HF而非AutoGPTQ，适用于从Llama衍生的模型。')
                            shared.gradio['quipsharp_info'] = gr.Markdown('QuIP#目前需要手动安装。')

                        with gr.Column():
                            shared.gradio['load_in_8bit'] = gr.Checkbox(label="加载8比特", value=shared.args.load_in_8bit)
                            shared.gradio['load_in_4bit'] = gr.Checkbox(label="加载4比特", value=shared.args.load_in_4bit)
                            shared.gradio['use_double_quant'] = gr.Checkbox(label="使用双重量化", value=shared.args.use_double_quant)
                            shared.gradio['use_flash_attention_2'] = gr.Checkbox(label="使用flash_attention_2", value=shared.args.use_flash_attention_2, info='加载模型时设置use_flash_attention_2=True。')
                            shared.gradio['auto_devices'] = gr.Checkbox(label="自动分配设备", value=shared.args.auto_devices)
                            shared.gradio['tensorcores'] = gr.Checkbox(label="张量核心", value=shared.args.tensorcores, info='仅限NVIDIA：使用支持张量核心的llama-cpp-python编译。这可以提高RTX卡的性能。')
                            shared.gradio['streaming_llm'] = gr.Checkbox(label="streaming_llm", value=shared.args.streaming_llm, info='（实验性功能）激活StreamingLLM以避免在删除旧消息时重新评估整个提示词。')
                            shared.gradio['attention_sink_size'] = gr.Number(label="attention_sink_size", value=shared.args.attention_sink_size, info='StreamingLLM：sink token的数量。仅在修剪后的提示词不与旧提示词前缀相同时使用。')
                            shared.gradio['cpu'] = gr.Checkbox(label="CPU", value=shared.args.cpu, info='llama.cpp：使用没有GPU加速的llama-cpp-python编译。Transformers：使用PyTorch的CPU模式。')
                            shared.gradio['row_split'] = gr.Checkbox(label="行分割", value=shared.args.row_split, info='在GPU之间按行分割模型。这可能会提高多GPU性能。')
                            shared.gradio['no_offload_kqv'] = gr.Checkbox(label="不卸载KQV", value=shared.args.no_offload_kqv, info='不要将K、Q、V卸载到GPU。这可以节省VRAM，但会降低性能。')
                            shared.gradio['no_mul_mat_q'] = gr.Checkbox(label="禁用mul_mat_q", value=shared.args.no_mul_mat_q, info='禁用mulmat内核。')
                            shared.gradio['triton'] = gr.Checkbox(label="Triton", value=shared.args.triton)
                            shared.gradio['no_inject_fused_attention'] = gr.Checkbox(label="不注入融合注意力", value=shared.args.no_inject_fused_attention, info='禁用融合注意力。融合注意力可以提高推理性能，但会使用更多的VRAM。融合AutoAWQ的层。如果VRAM不足，请禁用。')
                            shared.gradio['no_inject_fused_mlp'] = gr.Checkbox(label="不注入融合MLP", value=shared.args.no_inject_fused_mlp, info='仅影响Triton。禁用融合MLP。融合MLP可以提高性能，但会使用更多的VRAM。如果VRAM不足，请禁用。')
                            shared.gradio['no_use_cuda_fp16'] = gr.Checkbox(label="不使用cuda_fp16", value=shared.args.no_use_cuda_fp16, info='在某些系统上，这可以使模型更快。')
                            shared.gradio['desc_act'] = gr.Checkbox(label="描述激活", value=shared.args.desc_act, info='\'描述激活\'、\'权重位\'和\'组大小\'用于没有quantize_config.json的旧模型。')
                            shared.gradio['no_mmap'] = gr.Checkbox(label="不使用内存映射", value=shared.args.no_mmap)
                            shared.gradio['mlock'] = gr.Checkbox(label="内存锁定", value=shared.args.mlock)
                            shared.gradio['numa'] = gr.Checkbox(label="NUMA", value=shared.args.numa, info='NUMA支持可以在具有非统一内存访问的系统上提供帮助。')
                            shared.gradio['disk'] = gr.Checkbox(label="磁盘", value=shared.args.disk)
                            shared.gradio['bf16'] = gr.Checkbox(label="bf16", value=shared.args.bf16)
                            shared.gradio['cache_8bit'] = gr.Checkbox(label="8比特缓存", value=shared.args.cache_8bit, info='使用8比特缓存以节省VRAM。')
                            shared.gradio['cache_4bit'] = gr.Checkbox(label="4比特缓存", value=shared.args.cache_4bit, info='使用Q4缓存以节省VRAM。')
                            shared.gradio['autosplit'] = gr.Checkbox(label="自动分割", value=shared.args.autosplit, info='自动在可用的GPU之间分割模型张量。')
                            shared.gradio['no_flash_attn'] = gr.Checkbox(label="不使用flash_attention", value=shared.args.no_flash_attn, info='强制不使用flash-attention。')
                            shared.gradio['cfg_cache'] = gr.Checkbox(label="CFG缓存", value=shared.args.cfg_cache, info='使用此加载器时，使用CFG是必需的。')
                            shared.gradio['num_experts_per_token'] = gr.Number(label="每个标记的专家数量", value=shared.args.num_experts_per_token, info='仅适用于像Mixtral这样的MoE模型。')
                            with gr.Blocks():
                                shared.gradio['trust_remote_code'] = gr.Checkbox(label="信任远程代码(trust-remote-code)", value=shared.args.trust_remote_code, info='加载分词器/模型时设置trust_remote_code=True。要启用此选项，请使用--trust-remote-code参数启动Web UI。', interactive=shared.args.trust_remote_code)
                                shared.gradio['no_use_fast'] = gr.Checkbox(label="不使用快速模式", value=shared.args.no_use_fast, info='加载分词器时设置use_fast=False。')
                                shared.gradio['logits_all'] = gr.Checkbox(label="全部逻辑", value=shared.args.logits_all, info='使用此加载器进行困惑度评估时需要设置。否则，请忽略它，因为它会使提示处理速度变慢。')

                            shared.gradio['disable_exllama'] = gr.Checkbox(label="禁用ExLlama", value=shared.args.disable_exllama, info='对于GPTQ模型，禁用ExLlama内核。')
                            shared.gradio['disable_exllamav2'] = gr.Checkbox(label="禁用ExLlamav2", value=shared.args.disable_exllamav2, info='对于GPTQ模型，禁用ExLlamav2内核。')
                            shared.gradio['gptq_for_llama_info'] = gr.Markdown('用于与旧GPU兼容的传统加载器。如果支持，推荐使用ExLlamav2_HF或AutoGPTQ适用于GPTQ模型。')
                            shared.gradio['exllamav2_info'] = gr.Markdown("相比于ExLlamav2，推荐使用ExLlamav2_HF，因为它与扩展有更好的集成，并且在加载器之间提供了更一致的采样行为。")
                            shared.gradio['llamacpp_HF_info'] = gr.Markdown("llamacpp_HF将llama.cpp作为Transformers模型加载。要使用它，您需要将GGUF放在models/的子文件夹中，并提供必要的分词器文件。\n\n您可以使用'llamacpp_HF创建器'菜单自动完成。")

            with gr.Column():
                with gr.Row():
                    shared.gradio['autoload_model'] = gr.Checkbox(value=shared.settings['autoload_model'], label='自动加载模型', info='选择模型下拉菜单中的模型后是否立即加载模型。', interactive=not mu)

                with gr.Tab("下载"):
                    shared.gradio['custom_model_menu'] = gr.Textbox(label="下载模型或LoRA", info="输入Hugging Face用户名/模型路径，例如：facebook/galactica-125m。要指定分支，在最后加上\":\"字符，像这样：facebook/galactica-125m:main。要下载单个文件，请在第二个框中输入其名称。", interactive=not mu)
                    shared.gradio['download_specific_file'] = gr.Textbox(placeholder="文件名（适用于GGUF模型）", show_label=False, max_lines=1, interactive=not mu)
                    with gr.Row():
                        shared.gradio['download_model_button'] = gr.Button("下载", variant='primary', interactive=not mu)
                        shared.gradio['get_file_list'] = gr.Button("获取文件列表", interactive=not mu)

                with gr.Tab("llamacpp_HF创建器"):
                    with gr.Row():
                        shared.gradio['gguf_menu'] = gr.Dropdown(choices=utils.get_available_ggufs(), value=lambda: shared.model_name, label='选择你的GGUF', elem_classes='slim-dropdown', interactive=not mu)
                        ui.create_refresh_button(shared.gradio['gguf_menu'], lambda: None, lambda: {'choices': utils.get_available_ggufs()}, 'refresh-button', interactive=not mu)

                    shared.gradio['unquantized_url'] = gr.Textbox(label="输入原始（未量化）模型的URL", info="示例：https://hf-mirror.com/lmsys/vicuna-13b-v1.5", max_lines=1)
                    shared.gradio['create_llamacpp_hf_button'] = gr.Button("提交", variant="primary", interactive=not mu)
                    gr.Markdown("这将把你的gguf文件移动到`models`的子文件夹中，并附带必要的分词器文件。")

                with gr.Tab("自定义指令模板"):
                    with gr.Row():
                        shared.gradio['customized_template'] = gr.Dropdown(choices=utils.get_available_instruction_templates(), value='None', label='选择所需的指令模板', elem_classes='slim-dropdown')
                        ui.create_refresh_button(shared.gradio['customized_template'], lambda: None, lambda: {'choices': utils.get_available_instruction_templates()}, 'refresh-button', interactive=not mu)

                    shared.gradio['customized_template_submit'] = gr.Button("提交", variant="primary", interactive=not mu)
                    gr.Markdown("这允许你为\"模型加载器\"菜单中当前选中的模型设置一个自定义模板。每当加载模型时，都会使用此模板代替模型元数据中指定的模板，有时后者可能是错误的。")

                with gr.Row():
                    shared.gradio['model_status'] = gr.Markdown('没有加载模型' if shared.model_name == 'None' else '准备就绪')


def create_event_handlers():
    shared.gradio['loader'].change(
        loaders.make_loader_params_visible, gradio('loader'), gradio(loaders.get_all_params())).then(
        lambda value: gr.update(choices=loaders.get_model_types(value)), gradio('loader'), gradio('model_type'))

    # In this event handler, the interface state is read and updated
    # with the model defaults (if any), and then the model is loaded
    # unless "autoload_model" is unchecked
    shared.gradio['model_menu'].change(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        apply_model_settings_to_state, gradio('model_menu', 'interface_state'), gradio('interface_state')).then(
        ui.apply_interface_values, gradio('interface_state'), gradio(ui.list_interface_input_elements()), show_progress=False).then(
        update_model_parameters, gradio('interface_state'), None).then(
        load_model_wrapper, gradio('model_menu', 'loader', 'autoload_model'), gradio('model_status'), show_progress=False).success(
        update_truncation_length, gradio('truncation_length', 'interface_state'), gradio('truncation_length')).then(
        lambda x: x, gradio('loader'), gradio('filter_by_loader'))

    shared.gradio['load_model'].click(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        update_model_parameters, gradio('interface_state'), None).then(
        partial(load_model_wrapper, autoload=True), gradio('model_menu', 'loader'), gradio('model_status'), show_progress=False).success(
        update_truncation_length, gradio('truncation_length', 'interface_state'), gradio('truncation_length')).then(
        lambda x: x, gradio('loader'), gradio('filter_by_loader'))

    shared.gradio['reload_model'].click(
        unload_model, None, None).then(
        ui.gather_interface_values, gradio(shared.input_elements), gradio('interface_state')).then(
        update_model_parameters, gradio('interface_state'), None).then(
        partial(load_model_wrapper, autoload=True), gradio('model_menu', 'loader'), gradio('model_status'), show_progress=False).success(
        update_truncation_length, gradio('truncation_length', 'interface_state'), gradio('truncation_length')).then(
        lambda x: x, gradio('loader'), gradio('filter_by_loader'))

    shared.gradio['unload_model'].click(
        unload_model, None, None).then(
        lambda: "Model unloaded", None, gradio('model_status'))

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
        yield f"已更新`{selected_model}`的设置。\n\n点击“加载”来加载模型。"
        return

    if selected_model == 'None':
        yield "未选择模型"
    else:
        try:
            yield f"正在加载`{selected_model}`..."
            unload_model()
            if selected_model != '':
                shared.model, shared.tokenizer = load_model(selected_model, loader)

            if shared.model is not None:
                output = f"成功加载`{selected_model}`。"

                settings = get_model_metadata(selected_model)
                if 'instruction_template' in settings:
                    output += '\n\n这似乎是一个有指令模板 "{}" 的指令跟随模型。在聊天标签页中，应使用指令或聊天指令模式。'.format(settings['instruction_template'])

                yield output
            else:
                yield f"加载`{selected_model}`失败。"
        except:
            exc = traceback.format_exc()
            logger.error('加载模型失败。')
            print(exc)
            yield exc.replace('\n', '\n\n')


def load_lora_wrapper(selected_loras):
    yield ("将以下LoRAs应用于{}:\n\n{}".format(shared.model_name, '\n'.join(selected_loras)))
    add_lora_to_model(selected_loras)
    yield ("成功应用了LoRAs")


def download_model_wrapper(repo_id, specific_file, progress=gr.Progress(), return_links=False, check=False):
    try:
        downloader = importlib.import_module("download-model").ModelDownloader()

        progress(0.0)
        model, branch = downloader.sanitize_model_and_branch_names(repo_id, None)

        yield ("从HF Mirror获取下载链接")
        links, sha256, is_lora, is_llamacpp = downloader.get_download_links_from_huggingface(model, branch, text_only=False, specific_file=specific_file)
        if return_links:
            output = "```\n"
            for link in links:
                output += f"{Path(link).name}" + "\n"

            output += "```"
            yield output
            return

        yield ("获取输出文件夹")
        output_folder = downloader.get_output_folder(model, branch, is_lora, is_llamacpp=is_llamacpp)
        if check:
            progress(0.5)

            yield ("检查之前下载的文件")
            downloader.check_model_files(model, branch, links, sha256, output_folder)
            progress(1.0)
        else:
            yield (f"下载文件{'们' if len(links) > 1 else ''}到`{output_folder}`")
            downloader.download_model_files(model, branch, links, sha256, output_folder, progress_bar=progress, threads=4, is_llamacpp=is_llamacpp)

            yield (f"模型成功保存到`{output_folder}/`。")
    except:
        progress(1.0)
        yield traceback.format_exc().replace('\n', '\n\n')


def create_llamacpp_hf(gguf_name, unquantized_url, progress=gr.Progress()):
    try:
        downloader = importlib.import_module("download-model").ModelDownloader()

        progress(0.0)
        model, branch = downloader.sanitize_model_and_branch_names(unquantized_url, None)

        yield ("从Hugging Face获取分词器文件链接")
        links, sha256, is_lora, is_llamacpp = downloader.get_download_links_from_huggingface(model, branch, text_only=True)
        output_folder = Path(shared.args.model_dir) / (re.sub(r'(?i)\.gguf$', '', gguf_name) + "-HF")

        yield (f"下载分词器到`{output_folder}`")
        downloader.download_model_files(model, branch, links, sha256, output_folder, progress_bar=progress, threads=4, is_llamacpp=False)

        # 移动GGUF文件
        (Path(shared.args.model_dir) / gguf_name).rename(output_folder / gguf_name)

        yield (f"模型已保存到`{output_folder}/`。\n\n现在您可以使用llamacpp_HF加载它。")
    except:
        progress(1.0)
        yield traceback.format_exc().replace('\n', '\n\n')


def update_truncation_length(current_length, state):
    if 'loader' in state:
        if state['loader'].lower().startswith('exllama'):
            return state['max_seq_len']
        elif state['loader'] in ['llama.cpp', 'llamacpp_HF', 'ctransformers']:
            return state['n_ctx']

    return current_length
