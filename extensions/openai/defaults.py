import copy

# Slightly different defaults for OpenAI's API
# Data type is important, Ex. use 0.0 for a float 0
default_req_params = {
    'max_new_tokens': 16,  # 'Inf' for chat
    'auto_max_new_tokens': False,
    'max_tokens_second': 0,
    'temperature': 1.0,
    'top_p': 1.0,
    'top_k': 1,  # choose 20 for chat in absence of another default
    'repetition_penalty': 1.18,
    'repetition_penalty_range': 0,
    'encoder_repetition_penalty': 1.0,
    'suffix': None,
    'stream': False,
    'echo': False,
    'seed': -1,
    # 'n' : default(body, 'n', 1),  # 'n' doesn't have a direct map
    'truncation_length': 2048,  # first use shared.settings value
    'add_bos_token': True,
    'do_sample': True,
    'typical_p': 1.0,
    'epsilon_cutoff': 0.0,  # In units of 1e-4
    'eta_cutoff': 0.0,  # In units of 1e-4
    'tfs': 1.0,
    'top_a': 0.0,
    'min_length': 0,
    'no_repeat_ngram_size': 0,
    'num_beams': 1,
    'penalty_alpha': 0.0,
    'length_penalty': 1.0,
    'early_stopping': False,
    'mirostat_mode': 0,
    'mirostat_tau': 5.0,
    'mirostat_eta': 0.1,
    'guidance_scale': 1,
    'negative_prompt': '',
    'ban_eos_token': False,
    'custom_token_bans': '',
    'skip_special_tokens': True,
    'custom_stopping_strings': '',
    # 'logits_processor' - conditionally passed
    # 'stopping_strings' - temporarily used
    # 'logprobs' - temporarily used
    # 'requested_model' - temporarily used
}


def get_default_req_params():
    return copy.deepcopy(default_req_params)


def default(dic, key, default):
    '''
    little helper to get defaults if arg is present but None and should be the same type as default.
    '''
    val = dic.get(key, default)
    if not isinstance(val, type(default)):
        # maybe it's just something like 1 instead of 1.0
        try:
            v = type(default)(val)
            if type(val)(v) == val:  # if it's the same value passed in, it's ok.
                return v
        except:
            pass

        val = default
    return val


def clamp(value, minvalue, maxvalue):
    return max(minvalue, min(value, maxvalue))
