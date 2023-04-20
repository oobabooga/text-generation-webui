from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

from modules import shared
from modules.text_generation import encode

past_evaluations = []


def generate_markdown_table():
    if len(past_evaluations) == 0:
        return ''

    markdown_table = '|Model|LoRAs|Input file|Stride|Perplexity|\n|-----|-----|------|------|-----|\n'
    for evaluation in past_evaluations:
        markdown_table += '|{}|{}|{}|{}|{}|\n'.format(*evaluation)

    return markdown_table


def calculate_perplexity(input_dataset, stride):
    '''
    Based on:
    https://huggingface.co/docs/transformers/perplexity#calculating-ppl-with-fixedlength-models
    '''

    global past_evaluations
    yield "Loading the input dataset..."

    # Copied from https://github.com/qwopqwop200/GPTQ-for-LLaMa/blob/triton/utils/datautils.py
    if input_dataset == 'wikitext':
        data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        text = "\n\n".join(data['text'])
    elif input_dataset == 'ptb':
        data = load_dataset('ptb_text_only', 'penn_treebank', split='validation')
        text = "\n\n".join(data['sentence'])
    elif input_dataset == 'ptb_new':
        data = load_dataset('ptb_text_only', 'penn_treebank', split='test')
        text = " ".join(data['sentence'])
    else:
        with open(Path(f'training/datasets/{input_dataset}.txt'), 'r', encoding='utf-8') as f:
            text = f.read()

    yield "Tokenizing the input dataset..."
    encodings = encode(text, add_special_tokens=False)
    max_length = shared.model.config.max_position_embeddings
    seq_len = encodings.shape[1]
    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        yield f"Evaluating... {100*begin_loc/seq_len:.2f}%"
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = shared.model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    past_evaluations.append([shared.model_name, ', '.join(shared.lora_names), input_dataset, stride, ppl])
    yield generate_markdown_table()
