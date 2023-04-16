'''
Based on
https://github.com/abetlen/llama-cpp-python

Documentation:
https://abetlen.github.io/llama-cpp-python/
'''

from llama_cpp import Llama, LlamaCache

from modules import shared
from modules.callbacks import Iteratorize


class LlamaCppModel:
    def __init__(self):
        self.initialized = False

    @classmethod
    def from_pretrained(self, path):
        result = self()

        params = {
            'model_path': str(path),
            'n_ctx': 2048,
            'seed': 0,
            'n_threads': shared.args.threads or None
        }
        self.model = Llama(**params)
        self.model.set_cache(LlamaCache)

        # This is ugly, but the model and the tokenizer are the same object in this library.
        return result, result

    def encode(self, string):
        if type(string) is str:
            string = string.encode()
        return self.model.tokenize(string)

    def generate(self, context="", token_count=20, temperature=1, top_p=1, top_k=50, repetition_penalty=1, callback=None):
        if type(context) is str:
            context = context.encode()
        tokens = self.model.tokenize(context)

        output = b""
        count = 0
        for token in self.model.generate(tokens, top_k=top_k, top_p=top_p, temp=temperature, repeat_penalty=repetition_penalty):
            text = self.model.detokenize([token])
            output += text
            if callback:
                callback(text.decode())

            count += 1
            if count >= token_count or (token == self.model.token_eos()):
                break

        return output.decode()

    def generate_with_streaming(self, **kwargs):
        with Iteratorize(self.generate, kwargs, callback=None) as generator:
            reply = ''
            for token in generator:
                reply += token
                yield reply
