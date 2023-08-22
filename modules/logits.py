import torch

from modules import shared


def get_next_logits(prompt):
    tokens = shared.tokenizer.encode(prompt, return_tensors='pt').cuda()
    output = shared.model(input_ids=tokens)

    scores = output['logits'][-1][-1]
    probs = torch.softmax(scores, dim=-1, dtype=torch.float)

    topk_values, topk_indices = torch.topk(probs, k=20, largest=True, sorted=True)
    topk_values = [f"{float(i):.5f}" % i for i in topk_values]
    output = ''
    for row in list(zip(topk_values, shared.tokenizer.convert_ids_to_tokens(topk_indices))):
        output += f"{row[0]} {row[1]}\n"

    return output
