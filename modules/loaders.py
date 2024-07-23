import functools
from collections import OrderedDict

import gradio as gr

from modules import shared

loaders_and_params = OrderedDict({
    'Transformers': [
        'cpu_memory',
        'gpu_memory',
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
        'no_use_fast',
        'use_flash_attention_2',
        'use_eager_attention',
        'alpha_value',
        'compress_pos_emb',
        'disable_exllama',
        'disable_exllamav2',
        'transformers_info',
    ],
    'llama.cpp': [
        'n_ctx',
        'n_gpu_layers',
        'cache_8bit',
        'cache_4bit',
        'tensor_split',
        'n_batch',
        'threads',
        'threads_batch',
        'no_mmap',
        'mlock',
        'no_mul_mat_q',
        'rope_freq_base',
        'compress_pos_emb',
        'cpu',
        'numa',
        'no_offload_kqv',
        'row_split',
        'tensorcores',
        'flash_attn',
        'streaming_llm',
        'attention_sink_size',
    ],
    'llamacpp_HF': [
        'n_ctx',
        'n_gpu_layers',
        'cache_8bit',
        'cache_4bit',
        'tensor_split',
        'n_batch',
        'threads',
        'threads_batch',
        'no_mmap',
        'mlock',
        'no_mul_mat_q',
        'rope_freq_base',
        'compress_pos_emb',
        'cpu',
        'numa',
        'cfg_cache',
        'trust_remote_code',
        'no_use_fast',
        'logits_all',
        'no_offload_kqv',
        'row_split',
        'tensorcores',
        'flash_attn',
        'streaming_llm',
        'attention_sink_size',
        'llamacpp_HF_info',
    ],
    'ExLlamav2_HF': [
        'gpu_split',
        'max_seq_len',
        'cfg_cache',
        'no_flash_attn',
        'no_xformers',
        'no_sdpa',
        'num_experts_per_token',
        'cache_8bit',
        'cache_4bit',
        'autosplit',
        'alpha_value',
        'compress_pos_emb',
        'trust_remote_code',
        'no_use_fast',
    ],
    'ExLlamav2': [
        'gpu_split',
        'max_seq_len',
        'no_flash_attn',
        'no_xformers',
        'no_sdpa',
        'num_experts_per_token',
        'cache_8bit',
        'cache_4bit',
        'autosplit',
        'alpha_value',
        'compress_pos_emb',
        'exllamav2_info',
    ],
    'AutoGPTQ': [
        'triton',
        'no_inject_fused_mlp',
        'no_use_cuda_fp16',
        'wbits',
        'groupsize',
        'desc_act',
        'disable_exllama',
        'disable_exllamav2',
        'gpu_memory',
        'cpu_memory',
        'cpu',
        'disk',
        'auto_devices',
        'trust_remote_code',
        'no_use_fast',
        'autogptq_info',
    ],
    'HQQ': [
        'hqq_backend',
        'trust_remote_code',
        'no_use_fast',
    ],
    'TensorRT-LLM': [
        'max_seq_len',
        'cpp_runner',
        'tensorrt_llm_info',
    ]
})


def transformers_samplers():
    return {
        'temperature',
        'temperature_last',
        'dynamic_temperature',
        'dynatemp_low',
        'dynatemp_high',
        'dynatemp_exponent',
        'smoothing_factor',
        'smoothing_curve',
        'top_p',
        'min_p',
        'top_k',
        'typical_p',
        'epsilon_cutoff',
        'eta_cutoff',
        'tfs',
        'top_a',
        'repetition_penalty',
        'presence_penalty',
        'frequency_penalty',
        'repetition_penalty_range',
        'encoder_repetition_penalty',
        'no_repeat_ngram_size',
        'dry_multiplier',
        'dry_base',
        'dry_allowed_length',
        'dry_sequence_breakers',
        'seed',
        'do_sample',
        'penalty_alpha',
        'mirostat_mode',
        'mirostat_tau',
        'mirostat_eta',
        'grammar_file_row',
        'grammar_string',
        'guidance_scale',
        'negative_prompt',
        'ban_eos_token',
        'custom_token_bans',
        'sampler_priority',
        'add_bos_token',
        'skip_special_tokens',
        'auto_max_new_tokens',
        'prompt_lookup_num_tokens'
    }


loaders_samplers = {
    'Transformers': transformers_samplers(),
    'AutoGPTQ': transformers_samplers(),
    'HQQ': transformers_samplers(),
    'ExLlamav2': {
        'temperature',
        'temperature_last',
        'top_p',
        'min_p',
        'top_k',
        'typical_p',
        'tfs',
        'top_a',
        'repetition_penalty',
        'presence_penalty',
        'frequency_penalty',
        'repetition_penalty_range',
        'seed',
        'mirostat_mode',
        'mirostat_tau',
        'mirostat_eta',
        'ban_eos_token',
        'add_bos_token',
        'custom_token_bans',
        'skip_special_tokens',
        'auto_max_new_tokens',
    },
    'ExLlamav2_HF': {
        'temperature',
        'temperature_last',
        'dynamic_temperature',
        'dynatemp_low',
        'dynatemp_high',
        'dynatemp_exponent',
        'smoothing_factor',
        'smoothing_curve',
        'top_p',
        'min_p',
        'top_k',
        'typical_p',
        'epsilon_cutoff',
        'eta_cutoff',
        'tfs',
        'top_a',
        'repetition_penalty',
        'presence_penalty',
        'frequency_penalty',
        'repetition_penalty_range',
        'encoder_repetition_penalty',
        'no_repeat_ngram_size',
        'dry_multiplier',
        'dry_base',
        'dry_allowed_length',
        'dry_sequence_breakers',
        'seed',
        'do_sample',
        'mirostat_mode',
        'mirostat_tau',
        'mirostat_eta',
        'grammar_file_row',
        'grammar_string',
        'guidance_scale',
        'negative_prompt',
        'ban_eos_token',
        'custom_token_bans',
        'sampler_priority',
        'add_bos_token',
        'skip_special_tokens',
        'auto_max_new_tokens',
    },
    'llama.cpp': {
        'temperature',
        'top_p',
        'min_p',
        'top_k',
        'typical_p',
        'tfs',
        'repetition_penalty',
        'presence_penalty',
        'frequency_penalty',
        'seed',
        'mirostat_mode',
        'mirostat_tau',
        'mirostat_eta',
        'grammar_file_row',
        'grammar_string',
        'ban_eos_token',
        'custom_token_bans',
    },
    'llamacpp_HF': {
        'temperature',
        'temperature_last',
        'dynamic_temperature',
        'dynatemp_low',
        'dynatemp_high',
        'dynatemp_exponent',
        'smoothing_factor',
        'smoothing_curve',
        'top_p',
        'min_p',
        'top_k',
        'typical_p',
        'epsilon_cutoff',
        'eta_cutoff',
        'tfs',
        'top_a',
        'repetition_penalty',
        'presence_penalty',
        'frequency_penalty',
        'repetition_penalty_range',
        'encoder_repetition_penalty',
        'no_repeat_ngram_size',
        'dry_multiplier',
        'dry_base',
        'dry_allowed_length',
        'dry_sequence_breakers',
        'seed',
        'do_sample',
        'mirostat_mode',
        'mirostat_tau',
        'mirostat_eta',
        'grammar_file_row',
        'grammar_string',
        'guidance_scale',
        'negative_prompt',
        'ban_eos_token',
        'custom_token_bans',
        'sampler_priority',
        'add_bos_token',
        'skip_special_tokens',
        'auto_max_new_tokens',
    },
    'TensorRT-LLM': {
        'temperature',
        'top_p',
        'top_k',
        'repetition_penalty',
        'presence_penalty',
        'frequency_penalty',
        'ban_eos_token',
        'auto_max_new_tokens',
    }
}


@functools.cache
def list_all_samplers():
    all_samplers = set()
    for k in loaders_samplers:
        for sampler in loaders_samplers[k]:
            all_samplers.add(sampler)

    return sorted(all_samplers)


def blacklist_samplers(loader, dynamic_temperature):
    all_samplers = list_all_samplers()
    output = []

    for sampler in all_samplers:
        if loader == 'All' or sampler in loaders_samplers[loader]:
            if sampler.startswith('dynatemp'):
                output.append(gr.update(visible=dynamic_temperature))
            else:
                output.append(gr.update(visible=True))
        else:
            output.append(gr.update(visible=False))

    return output


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
