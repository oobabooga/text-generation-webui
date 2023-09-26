from transformers.generation.logits_process import LogitsProcessor

from modules import shared
from modules.relative_imports import RelativeImport

with RelativeImport("repositories/torch-grammar"):
    from torch_grammar import GrammarSampler


grammar = None
grammar_string = ''


class GrammarLogitsProcessor(LogitsProcessor):
    def __init__(self, string):

        global grammar, grammar_string

        if string != grammar_string:
            grammar_string = string
            if string.strip() != '':
                string = string.strip() + '\n'
                grammar = GrammarSampler(string, 'root', shared.tokenizer).logits_processor()
            else:
                grammar = None

    def __call__(self, input_ids, scores):
        if grammar is not None:
            scores = grammar(input_ids, scores)

        return scores
