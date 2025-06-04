import time
import traceback

import numpy as np

from modules import models, shared
from modules.logging_colors import logger
from modules.models import load_model
from modules.text_generation import generate_reply
from modules.utils import check_model_loaded

global_scores = None


def get_next_logits(*args, **kwargs):
    if shared.args.idle_timeout > 0 and shared.model is None and shared.model_name not in [None, 'None']:
        shared.model, shared.tokenizer = load_model(shared.model_name)

    needs_lock = not args[2]  # use_samplers
    if needs_lock:
        shared.generation_lock.acquire()

    try:
        result = _get_next_logits(*args, **kwargs)
    except Exception:
        traceback.print_exc()
        result = None

    if needs_lock:
        models.last_generation_time = time.time()
        shared.generation_lock.release()

    return result


def _get_next_logits(prompt, state, use_samplers, previous, top_logits=25, return_dict=False):
    model_is_loaded, error_message = check_model_loaded()
    if not model_is_loaded:
        return error_message, previous

    # llama.cpp case
    if shared.model.__class__.__name__ == 'LlamaServer':
        logprobs = shared.model.get_logits(prompt, state, n_probs=top_logits, use_samplers=use_samplers)

        if return_dict:
            output = {}
            for entry in logprobs:
                token = repr(entry['token'])
                if len(token) > 2 and token.startswith("'") and token.endswith("'"):
                    token = token[1:-1]

                prob = entry['prob'] if use_samplers else np.exp(entry['logprob'])
                output[token] = prob
            return output
        else:
            output = ''
            for entry in logprobs:
                token = repr(entry['token'])
                if len(token) > 2 and token.startswith("'") and token.endswith("'"):
                    token = token[1:-1]

                prob = entry['prob'] if use_samplers else np.exp(entry['logprob'])
                output += f"{prob:.5f}  -  {token}\n"
            return output, previous

    # All other model types
    else:
        import torch

        from modules import sampler_hijack
        from modules.torch_utils import get_device

        is_non_hf_exllamav2 = shared.model.__class__.__name__ == 'Exllamav2Model'

        if not use_samplers:
            state = {'stream': True}

        if use_samplers:
            if is_non_hf_exllamav2:
                # sampling is all done in C++ for exllama, so it is really hard to hijack
                logger.error("Sampler hijacking is not supported non-Huggingface loaders.")
                return 'Error: Sampler hijacking is not supported non-Huggingface loaders. Please disable the "Use samplers" option.', previous

            state['max_new_tokens'] = 1
            state['auto_max_new_tokens'] = False
            for _ in generate_reply(prompt, state):
                pass

            scores = sampler_hijack.global_scores[-1]
        else:
            if is_non_hf_exllamav2:
                device = get_device()
                tokens = shared.tokenizer.encode(prompt)
                if device:
                    tokens = tokens.to(device)

                scores = shared.model.get_logits(tokens)[-1][-1]
            else:
                device = get_device()
                tokens = shared.tokenizer.encode(prompt, return_tensors='pt')
                if device:
                    tokens = tokens.to(device)

                output = shared.model(input_ids=tokens)
                scores = output['logits'][-1][-1]

        probs = torch.softmax(scores, dim=-1, dtype=torch.float)
        topk_values, topk_indices = torch.topk(probs, k=top_logits, largest=True, sorted=True)
        if hasattr(shared.tokenizer, 'convert_ids_to_tokens'):
            tokens = [shared.tokenizer.convert_ids_to_tokens(int(i)) for i in topk_indices]
        else:
            tokens = [shared.tokenizer.decode(i) for i in topk_indices]

        if return_dict:
            topk_values = [float(i) for i in topk_values]
            output = {}
            for row in list(zip(topk_values, tokens)):
                key = row[1]
                if isinstance(key, bytes):
                    try:
                        key = key.decode()
                    except:
                        key = key.decode('latin')

                output[key] = row[0]

            return output
        else:
            topk_values = [f"{float(i):.5f}" for i in topk_values]
            output = ''
            for row in list(zip(topk_values, tokens)):
                output += f"{row[0]}  -  {repr(row[1])}\n"

            return output, previous
