from math import exp, log

import torch
from torch.nn.functional import softmax
from tqdm import tqdm
from pathlib import Path

from modules import shared
from modules.text_generation import encode


def calculate_perplexity(input_file, context_size, window):
    # Encode the text file
    with open(Path(f'training/datasets/{input_file}.txt'), 'r', encoding='utf-8') as f:
        text = f.read()

    # Tokenize the text
    tokens = encode(text, add_special_tokens=False)

    # Loop through the tokens and generate text token-by-token
    yield "Evaluating..."
    perplexity_sum = 0.0
    print(context_size, tokens.shape[1], window)
    for i in tqdm(range(context_size, tokens.shape[1], window)):
        # Extract the context from the tokens
        context = tokens[:, i - context_size:i]
        ground_truth_next_token = tokens[:, i]

        # Calculate the perplexity for the next token probability
        with torch.no_grad():
            outputs = shared.model(context)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_probs = softmax(next_token_logits, dim=-1)  # Softmax to obtain probabilities
            next_token_prob = next_token_probs.gather(dim=-1, index=ground_truth_next_token.unsqueeze(1)).squeeze(1)  # Probability assigned by the model to the ground truth next token
            perplexity_sum += -log(next_token_prob)  # Add the log probability to the perplexity

    average_perplexity = exp(perplexity_sum / (tokens.shape[1] - 2))
    yield f"The resulting perplexity for {shared.model_name} is: {average_perplexity}"
