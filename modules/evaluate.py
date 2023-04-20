from pathlib import Path

import torch
from tqdm import tqdm

from modules import shared
from modules.text_generation import encode

past_evaluations = []

def generate_markdown_table():
    markdown_table = '|Model|LoRAs|Input file|Perplexity|\n|-----|-----|----------|----------|\n'
    for evaluation in past_evaluations:
        markdown_table += '|{}|{}|{}|{}|\n'.format(evaluation[0], evaluation[1], evaluation[2], evaluation[3])

    return markdown_table

def calculate_perplexity(input_file, stride):
    '''
    Based on:
    https://huggingface.co/docs/transformers/perplexity#calculating-ppl-with-fixedlength-models
    '''

    global past_evaluations

    # Encode the text file
    with open(Path(f'training/datasets/{input_file}.txt'), 'r', encoding='utf-8') as f:
        text = f.read()

    yield "Tokenizing the input dataset..."
    encodings = encode(text, add_special_tokens=False)
    max_length = shared.model.config.max_position_embeddings
    stride = 512
    seq_len = encodings.shape[1]
    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        yield f"Evaluating... {100*begin_loc/seq_len:.2f}%%"
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
    past_evaluations.append([shared.model_name, ', '.join(shared.lora_names), input_file, ppl])
    yield generate_markdown_table()
