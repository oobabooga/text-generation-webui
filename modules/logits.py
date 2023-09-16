import torch

from modules import sampler_hijack, shared
from modules.exllama import ExllamaModel
from modules.exllamav2 import Exllamav2Model
from modules.logging_colors import logger
from modules.text_generation import generate_reply

global_scores = None


def get_next_logits(prompt, state, use_samplers, previous):
    if shared.model is None:
        logger.error("No model is loaded! Select one in the Model tab.")
        return '', previous
    is_non_hf_exllamav2 = isinstance(shared.model, Exllamav2Model)
    is_non_hf_exllamav1 = isinstance(shared.model, ExllamaModel)
    if use_samplers:
        if is_non_hf_exllamav2 or is_non_hf_exllamav1:
            logger.error("Sampler hijacking is not supported non-Huggingface loaders.")
            # sampling is all done in c for exllama, so it is really hard to hijack
            return '', previous
        state['max_new_tokens'] = 1
        state['auto_max_new_tokens'] = False
        for _ in generate_reply(prompt, state):
            pass

        scores = sampler_hijack.global_scores[-1]
    else:
        if is_non_hf_exllamav2 or is_non_hf_exllamav1:
            tokens = shared.tokenizer.encode(prompt).cuda()
            scores = shared.model.get_logits(tokens)[-1][-1]
        else:
            tokens = shared.tokenizer.encode(prompt, return_tensors='pt').cuda()
            output = shared.model(input_ids=tokens)
            scores = output['logits'][-1][-1]

    probs = torch.softmax(scores, dim=-1, dtype=torch.float)
    topk_values, topk_indices = torch.topk(probs, k=25, largest=True, sorted=True)
    topk_values = [f"{float(i):.5f}" for i in topk_values]
    if is_non_hf_exllamav1:
        topk_indices = [i.expand((1, 1)) for i in topk_indices]
    tokens = [shared.tokenizer.decode(i) for i in topk_indices]

    output = ''
    for row in list(zip(topk_values, tokens)):
        output += f"{row[0]}  -  {row[1]}\n"

    return output, previous
