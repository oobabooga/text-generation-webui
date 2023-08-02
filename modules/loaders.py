import functools

import gradio as gr

from modules import shared

loaders_and_params = {
    'AutoGPTQ': [
        'triton',
        'no_inject_fused_attention',
        'no_inject_fused_mlp',
        'no_use_cuda_fp16',
        'wbits',
        'groupsize',
        'desc_act',
        'gpu_memory',
        'cpu_memory',
        'cpu',
        'disk',
        'auto_devices',
        'trust_remote_code',
        'autogptq_info',
    ],
    'GPTQ-for-LLaMa': [
        'wbits',
        'groupsize',
        'model_type',
        'pre_layer',
        'gptq_for_llama_info',
    ],
    'llama.cpp': [
        'n_ctx',
        'n_gqa',
        'rms_norm_eps',
        'n_gpu_layers',
        'n_batch',
        'threads',
        'no_mmap',
        'low_vram',
        'mlock',
        'llama_cpp_seed',
        'compress_pos_emb',
        'alpha_value',
    ],
    'llamacpp_HF': [
        'n_ctx',
        'n_gqa',
        'rms_norm_eps',
        'n_gpu_layers',
        'n_batch',
        'threads',
        'no_mmap',
        'low_vram',
        'mlock',
        'llama_cpp_seed',
        'compress_pos_emb',
        'alpha_value',
        'llamacpp_HF_info',
    ],
    'Transformers': [
        'cpu_memory',
        'gpu_memory',
        'trust_remote_code',
        'load_in_8bit',
        'bf16',
        'cpu',
        'disk',
        'auto_devices',
        'load_in_4bit',
        'use_double_quant',
        'quant_type',
        'compute_dtype',
        'trust_remote_code',
        'transformers_info'
    ],
    'ExLlama': [
        'gpu_split',
        'max_seq_len',
        'compress_pos_emb',
        'alpha_value',
        'exllama_info',
    ],
    'ExLlama_HF': [
        'gpu_split',
        'max_seq_len',
        'compress_pos_emb',
        'alpha_value',
        'exllama_HF_info',
    ]
}

loaders_samplers = {
    'Transformers': {
        'temperature',
        'top_p',
        'top_k',
        'typical_p',
        'epsilon_cutoff',
        'eta_cutoff',
        'tfs',
        'top_a',
        'repetition_penalty',
        'repetition_penalty_range',
        'encoder_repetition_penalty',
        'no_repeat_ngram_size',
        'min_length',
        'seed',
        'do_sample',
        'penalty_alpha',
        'num_beams',
        'length_penalty',
        'early_stopping',
        'mirostat_mode',
        'mirostat_tau',
        'mirostat_eta',
        'ban_eos_token',
        'add_bos_token',
        'skip_special_tokens',
    },
    'ExLlama_HF': {
        'temperature',
        'top_p',
        'top_k',
        'typical_p',
        'epsilon_cutoff',
        'eta_cutoff',
        'tfs',
        'top_a',
        'repetition_penalty',
        'repetition_penalty_range',
        'encoder_repetition_penalty',
        'no_repeat_ngram_size',
        'min_length',
        'seed',
        'do_sample',
        'mirostat_mode',
        'mirostat_tau',
        'mirostat_eta',
        'ban_eos_token',
        'add_bos_token',
        'skip_special_tokens',
    },
    'ExLlama': {
        'temperature',
        'top_p',
        'top_k',
        'typical_p',
        'repetition_penalty',
        'repetition_penalty_range',
        'seed',
        'ban_eos_token',
    },
    'AutoGPTQ': {
        'temperature',
        'top_p',
        'top_k',
        'typical_p',
        'epsilon_cutoff',
        'eta_cutoff',
        'tfs',
        'top_a',
        'repetition_penalty',
        'repetition_penalty_range',
        'encoder_repetition_penalty',
        'no_repeat_ngram_size',
        'min_length',
        'seed',
        'do_sample',
        'penalty_alpha',
        'num_beams',
        'length_penalty',
        'early_stopping',
        'mirostat_mode',
        'mirostat_tau',
        'mirostat_eta',
        'ban_eos_token',
        'add_bos_token',
        'skip_special_tokens',
    },
    'GPTQ-for-LLaMa': {
        'temperature',
        'top_p',
        'top_k',
        'typical_p',
        'epsilon_cutoff',
        'eta_cutoff',
        'tfs',
        'top_a',
        'repetition_penalty',
        'repetition_penalty_range',
        'encoder_repetition_penalty',
        'no_repeat_ngram_size',
        'min_length',
        'seed',
        'do_sample',
        'penalty_alpha',
        'num_beams',
        'length_penalty',
        'early_stopping',
        'mirostat_mode',
        'mirostat_tau',
        'mirostat_eta',
        'ban_eos_token',
        'add_bos_token',
        'skip_special_tokens',
    },
    'llama.cpp': {
        'temperature',
        'top_p',
        'top_k',
        'tfs',
        'repetition_penalty',
        'mirostat_mode',
        'mirostat_tau',
        'mirostat_eta',
        'ban_eos_token',
    },
    'llamacpp_HF': {
        'temperature',
        'top_p',
        'top_k',
        'typical_p',
        'epsilon_cutoff',
        'eta_cutoff',
        'tfs',
        'top_a',
        'repetition_penalty',
        'repetition_penalty_range',
        'encoder_repetition_penalty',
        'no_repeat_ngram_size',
        'min_length',
        'seed',
        'do_sample',
        'mirostat_mode',
        'mirostat_tau',
        'mirostat_eta',
        'ban_eos_token',
        'add_bos_token',
        'skip_special_tokens',
    },
}


@functools.cache
def list_all_samplers():
    all_samplers = set()
    for k in loaders_samplers:
        for sampler in loaders_samplers[k]:
            all_samplers.add(sampler)

    return sorted(all_samplers)


def blacklist_samplers(loader):
    all_samplers = list_all_samplers()
    if loader == 'All':
        return [gr.update(visible=True) for sampler in all_samplers]
    else:
        return [gr.update(visible=True) if sampler in loaders_samplers[loader] else gr.update(visible=False) for sampler in all_samplers]


def get_gpu_memory_keys():
    return [k for k in shared.gradio if k.startswith('gpu_memory')]


@functools.cache
def get_all_params():
    all_params = set()
    for k in loaders_and_params:
        for el in loaders_and_params[k]:
            all_params.add(el)

    if 'gpu_memory' in all_params:
        all_params.remove('gpu_memory')
        for k in get_gpu_memory_keys():
            all_params.add(k)

    return sorted(all_params)


def make_loader_params_visible(loader):
    params = []
    all_params = get_all_params()
    if loader in loaders_and_params:
        params = loaders_and_params[loader]

        if 'gpu_memory' in params:
            params.remove('gpu_memory')
            params += get_gpu_memory_keys()

    return [gr.update(visible=True) if k in params else gr.update(visible=False) for k in all_params]
