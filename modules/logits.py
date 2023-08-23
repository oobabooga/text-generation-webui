import torch

from modules import sampler_hijack, shared
from modules.text_generation import generate_reply

global_scores = None


def get_next_logits(prompt, state, use_samplers, previous):
    if use_samplers:
        state['max_new_tokens'] = 1
        state['auto_max_new_tokens'] = False
        for _ in generate_reply(prompt, state):
            pass

        scores = sampler_hijack.global_scores[-1]
    else:
        tokens = shared.tokenizer.encode(prompt, return_tensors='pt').cuda()
        output = shared.model(input_ids=tokens)
        scores = output['logits'][-1][-1]

    probs = torch.softmax(scores, dim=-1, dtype=torch.float)
    topk_values, topk_indices = torch.topk(probs, k=25, largest=True, sorted=True)
    topk_values = [f"{float(i):.5f}" for i in topk_values]
    tokens = [shared.tokenizer.decode(i) for i in topk_indices]

    output = ''
    for row in list(zip(topk_values, tokens)):
        output += f"{row[0]}  -  {row[1]}\n"

    return output, previous
