from torch_grammar import GrammarSampler
from transformers.generation.logits_process import LogitsProcessor

from modules import shared

sampler = None
grammar = None
grammar_string = ''


class GrammarLogitsProcessor(LogitsProcessor):
    def __init__(self, string):

        global sampler, grammar, grammar_string

        if string != grammar_string:
            grammar_string = string
            if string.strip() != '':
                string = string.strip() + '\n'
                sampler = GrammarSampler(string, 'root', shared.tokenizer)
            else:
                sampler = None

        if sampler is not None:
            grammar = sampler.logits_processor()
        else:
            grammar = None

    def __call__(self, input_ids, scores):
        if grammar is not None:
            scores = grammar(input_ids, scores)

        return scores
