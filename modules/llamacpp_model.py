import os
from pathlib import Path
import modules.shared as shared
from modules.callbacks import Iteratorize

import llamacpp


class LlamaCppTokenizer:
    """A thin wrapper over the llamacpp tokenizer"""
    def __init__(self, model: llamacpp.PyLLAMA):
        self._tokenizer = model.get_tokenizer()
        self.eos_token_id = 2
        self.bos_token_id = 0

    @classmethod
    def from_model(cls, model: llamacpp.PyLLAMA):
        return cls(model)

    def encode(self, prompt):
        return self._tokenizer.tokenize(prompt)

    def decode(self, ids):
        return self._tokenizer.detokenize(ids)


class LlamaCppModel:
    def __init__(self):
        self.initialized = False

    @classmethod
    def from_pretrained(self, path):
        params = llamacpp.gpt_params(
            str(path),  # model
            2048,  # ctx_size
            200,  # n_predict
            40,  # top_k
            0.95,  # top_p
            0.80,  # temp
            1.30,  # repeat_penalty
            -1,  # seed
            8,  # threads
            64,  # repeat_last_n
            8,  # batch_size
        )

        _model = llamacpp.PyLLAMA(params)

        result = self()
        result.model = _model

        tokenizer = LlamaCppTokenizer.from_model(_model)
        return result, tokenizer

    # TODO: Allow passing in params for each inference
    def generate(self, context="", num_tokens=10, callback=None):
        # params = self.params
        # params.n_predict = token_count
        # params.top_p = top_p
        # params.top_k = top_k
        # params.temp = temperature
        # params.repeat_penalty = repetition_penalty
        # params.repeat_last_n = repeat_last_n

        # model.params = params
        if not self.initialized:
            self.model.add_bos()

        self.model.update_input(context)
        if not self.initialized:
            self.model.prepare_context()
            self.initialized = True

        output = ""
        is_end_of_text = False
        ctr = 0
        while not self.model.is_finished() and ctr < num_tokens and not is_end_of_text:
            if self.model.has_unconsumed_input():
                self.model.ingest_all_pending_input(False)
            else:
                text, is_end_of_text = self.model.infer_text()
                if callback:
                    callback(text)
                output += text
                ctr += 1

        return output

    def generate_with_streaming(self, **kwargs):
        with Iteratorize(self.generate, kwargs, callback=None) as generator:
            reply = kwargs['context']
            for token in generator:
                reply += token
                yield reply
