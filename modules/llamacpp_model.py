import multiprocessing

import llamacpp

from modules import shared
from modules.callbacks import Iteratorize


class LlamaCppTokenizer:
    """A thin wrapper over the llamacpp tokenizer"""
    def __init__(self, model: llamacpp.LlamaInference):
        self._tokenizer = model.get_tokenizer()
        self.eos_token_id = 2
        self.bos_token_id = 0

    @classmethod
    def from_model(cls, model: llamacpp.LlamaInference):
        return cls(model)

    def encode(self, prompt: str):
        return self._tokenizer.tokenize(prompt)

    def decode(self, ids):
        return self._tokenizer.detokenize(ids)


class LlamaCppModel:
    def __init__(self):
        self.initialized = False

    @classmethod
    def from_pretrained(self, path):
        params = llamacpp.InferenceParams()
        params.path_model = str(path)
        params.n_threads = shared.args.threads or multiprocessing.cpu_count() // 2

        _model = llamacpp.LlamaInference(params)

        result = self()
        result.model = _model
        result.params = params

        tokenizer = LlamaCppTokenizer.from_model(_model)
        return result, tokenizer

    def generate(self, context="", token_count=20, temperature=1, top_p=1, top_k=50, repetition_penalty=1, callback=None):
        params = self.params
        params.n_predict = token_count
        params.top_p = top_p
        params.top_k = top_k
        params.temp = temperature
        params.repeat_penalty = repetition_penalty
        # params.repeat_last_n = repeat_last_n

        # self.model.params = params
        self.model.add_bos()
        self.model.update_input(context)

        output = ""
        is_end_of_text = False
        ctr = 0
        while ctr < token_count and not is_end_of_text:
            if self.model.has_unconsumed_input():
                self.model.ingest_all_pending_input()
            else:
                self.model.eval()
                token = self.model.sample()
                text = self.model.token_to_str(token)
                output += text
                is_end_of_text = token == self.model.token_eos()
                if callback:
                    callback(text)
                ctr += 1

        return output

    def generate_with_streaming(self, **kwargs):
        with Iteratorize(self.generate, kwargs, callback=None) as generator:
            reply = ''
            for token in generator:
                reply += token
                yield reply
