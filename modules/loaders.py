import functools
from collections import OrderedDict

loaders_and_params = OrderedDict({
    'llama.cpp': [
        'gpu_layers',
        'fit_target',
        'cpu_moe',
        'threads',
        'threads_batch',
        'batch_size',
        'ubatch_size',
        'ctx_size',
        'cache_type',
        'tensor_split',
        'extra_flags',
        'streaming_llm',
        'row_split',
        'no_kv_offload',
        'no_mmap',
        'mlock',
        'numa',
        'parallel',
        'model_draft',
        'draft_max',
        'gpu_layers_draft',
        'device_draft',
        'ctx_size_draft',
        'ngram_header',
        'spec_type',
        'spec_ngram_size_n',
        'spec_ngram_size_m',
        'spec_ngram_min_hits',
        'speculative_decoding_accordion',
        'mmproj',
        'mmproj_accordion',
        'vram_info',
    ],
    'Transformers': [
        'gpu_split',
        'cpu_memory',
        'compute_dtype',
        'quant_type',
        'load_in_8bit',
        'load_in_4bit',
        'attn_implementation',
        'cpu',
        'disk',
        'use_double_quant',
        'bf16',
        'no_use_fast',
    ],
    'ExLlamav3_HF': [
        'ctx_size',
        'cache_type',
        'gpu_split',
        'cfg_cache',
        'no_use_fast',
        'enable_tp',
        'tp_backend',
    ],
    'ExLlamav3': [
        'ctx_size',
        'cache_type',
        'gpu_split',
        'model_draft',
        'draft_max',
        'speculative_decoding_accordion',
        'enable_tp',
        'tp_backend',
    ],
    'TensorRT-LLM': [
        'ctx_size',
        'tensorrt_llm_info',
    ]
})


def transformers_samplers():
    return {
        'temperature',
        'dynatemp_low',
        'dynatemp_high',
        'dynatemp_exponent',
        'smoothing_factor',
        'smoothing_curve',
        'min_p',
        'top_p',
        'top_k',
        'typical_p',
        'xtc_threshold',
        'xtc_probability',
        'epsilon_cutoff',
        'eta_cutoff',
        'tfs',
        'top_a',
        'top_n_sigma',
        'adaptive_target',
        'adaptive_decay',
        'dry_multiplier',
        'dry_allowed_length',
        'dry_base',
        'repetition_penalty',
        'frequency_penalty',
        'presence_penalty',
        'encoder_repetition_penalty',
        'no_repeat_ngram_size',
        'repetition_penalty_range',
        'penalty_alpha',
        'guidance_scale',
        'mirostat_mode',
        'mirostat_tau',
        'mirostat_eta',
        'prompt_lookup_num_tokens',
        'do_sample',
        'dynamic_temperature',
        'temperature_last',
        'auto_max_new_tokens',
        'ban_eos_token',
        'add_bos_token',
        'enable_thinking',
        'reasoning_effort',
        'skip_special_tokens',
        'static_cache',
        'seed',
        'sampler_priority',
        'custom_token_bans',
        'negative_prompt',
        'dry_sequence_breakers',
        'grammar_string',
        'grammar_file_row',
    }


loaders_samplers = {
    'Transformers': transformers_samplers(),
    'ExLlamav3_HF': {
        'temperature',
        'dynatemp_low',
        'dynatemp_high',
        'dynatemp_exponent',
        'smoothing_factor',
        'smoothing_curve',
        'min_p',
        'top_p',
        'top_k',
        'typical_p',
        'xtc_threshold',
        'xtc_probability',
        'epsilon_cutoff',
        'eta_cutoff',
        'tfs',
        'top_a',
        'top_n_sigma',
        'adaptive_target',
        'adaptive_decay',
        'dry_multiplier',
        'dry_allowed_length',
        'dry_base',
        'repetition_penalty',
        'frequency_penalty',
        'presence_penalty',
        'encoder_repetition_penalty',
        'no_repeat_ngram_size',
        'repetition_penalty_range',
        'guidance_scale',
        'mirostat_mode',
        'mirostat_tau',
        'mirostat_eta',
        'do_sample',
        'dynamic_temperature',
        'temperature_last',
        'auto_max_new_tokens',
        'ban_eos_token',
        'add_bos_token',
        'enable_thinking',
        'reasoning_effort',
        'skip_special_tokens',
        'seed',
        'sampler_priority',
        'custom_token_bans',
        'negative_prompt',
        'dry_sequence_breakers',
        'grammar_string',
        'grammar_file_row',
    },
    'ExLlamav3': {
        'temperature',
        'min_p',
        'top_p',
        'top_k',
        'adaptive_target',
        'adaptive_decay',
        'repetition_penalty',
        'frequency_penalty',
        'presence_penalty',
        'repetition_penalty_range',
        'temperature_last',
        'sampler_priority',
        'auto_max_new_tokens',
        'ban_eos_token',
        'add_bos_token',
        'enable_thinking',
        'reasoning_effort',
        'seed',
        'skip_special_tokens',
    },
    'llama.cpp': {
        'temperature',
        'dynatemp_low',
        'dynatemp_high',
        'dynatemp_exponent',
        'min_p',
        'top_p',
        'top_k',
        'typical_p',
        'xtc_threshold',
        'xtc_probability',
        'top_n_sigma',
        'adaptive_target',
        'adaptive_decay',
        'dry_multiplier',
        'dry_allowed_length',
        'dry_base',
        'repetition_penalty',
        'frequency_penalty',
        'presence_penalty',
        'repetition_penalty_range',
        'mirostat_mode',
        'mirostat_tau',
        'mirostat_eta',
        'dynamic_temperature',
        'temperature_last',
        'auto_max_new_tokens',
        'ban_eos_token',
        'add_bos_token',
        'enable_thinking',
        'reasoning_effort',
        'seed',
        'sampler_priority',
        'custom_token_bans',
        'dry_sequence_breakers',
        'grammar_string',
        'grammar_file_row',
    },
    'TensorRT-LLM': {
        'temperature',
        'top_p',
        'top_k',
        'min_p',
        'repetition_penalty',
        'frequency_penalty',
        'presence_penalty',
        'no_repeat_ngram_size',
        'auto_max_new_tokens',
        'ban_eos_token',
        'add_bos_token',
        'skip_special_tokens',
        'seed',
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
    import gradio as gr
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


@functools.cache
def get_all_params():
    all_params = set()
    for k in loaders_and_params:
        for el in loaders_and_params[k]:
            all_params.add(el)

    return sorted(all_params)


def list_model_elements():
    return [
        'filter_by_loader',
        'loader',
        'cpu_memory',
        'gpu_layers',
        'fit_target',
        'cpu_moe',
        'threads',
        'threads_batch',
        'batch_size',
        'ubatch_size',
        'ctx_size',
        'cache_type',
        'tensor_split',
        'extra_flags',
        'streaming_llm',
        'gpu_split',
        'compute_dtype',
        'quant_type',
        'load_in_8bit',
        'load_in_4bit',
        'attn_implementation',
        'cpu',
        'disk',
        'row_split',
        'no_kv_offload',
        'no_mmap',
        'mlock',
        'numa',
        'parallel',
        'use_double_quant',
        'bf16',
        'enable_tp',
        'tp_backend',
        'cfg_cache',
        'no_use_fast',
        'model_draft',
        'draft_max',
        'gpu_layers_draft',
        'device_draft',
        'ctx_size_draft',
        'spec_type',
        'spec_ngram_size_n',
        'spec_ngram_size_m',
        'spec_ngram_min_hits',
        'mmproj',
    ]


def make_loader_params_visible(loader):
    import gradio as gr
    params = []
    all_params = get_all_params()
    if loader in loaders_and_params:
        params = loaders_and_params[loader]

    return [gr.update(visible=True) if k in params else gr.update(visible=False) for k in all_params]
