'''
Based on
https://github.com/abetlen/llama-cpp-python

Documentation:
https://abetlen.github.io/llama-cpp-python/
'''

import logging
import re

from llama_cpp import Llama, LlamaCache

from modules import shared
from modules.callbacks import Iteratorize


class LlamaCppModel:
    def __init__(self):
        self.initialized = False

    def __del__(self):        
        self.model.__del__()

    @classmethod
    def from_pretrained(self, path):
        result = self()

        cache_capacity = 0
        if shared.args.cache_capacity is not None:
            if 'GiB' in shared.args.cache_capacity:
                cache_capacity = int(re.sub('[a-zA-Z]', '', shared.args.cache_capacity)) * 1000 * 1000 * 1000
            elif 'MiB' in shared.args.cache_capacity:
                cache_capacity = int(re.sub('[a-zA-Z]', '', shared.args.cache_capacity)) * 1000 * 1000
            else:
                cache_capacity = int(shared.args.cache_capacity)

        logging.info("Cache capacity is " + str(cache_capacity) + " bytes")

        params = {
            'model_path': str(path),
            'n_ctx': 2048,
            'seed': 0,
            'n_threads': shared.args.threads or None,
            'n_batch': shared.args.n_batch,
            'use_mmap': not shared.args.no_mmap,
            'use_mlock': shared.args.mlock,
            'n_gpu_layers': shared.args.n_gpu_layers
        }
        self.model = Llama(**params)
        if cache_capacity > 0:
            self.model.set_cache(LlamaCache(capacity_bytes=cache_capacity))

        # This is ugly, but the model and the tokenizer are the same object in this library.
        return result, result

    def encode(self, string):
        if type(string) is str:
            string = string.encode()
        return self.model.tokenize(string)

    def generate(self, context="", token_count=20, temperature=1, top_p=1, top_k=50, repetition_penalty=1, callback=None):
        context = context if type(context) is str else context.decode()
        completion_chunks = self.model.create_completion(
            prompt=context,
            max_tokens=token_count,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repetition_penalty,
            stream=True
        )
        output = ""
        for completion_chunk in completion_chunks:
            text = completion_chunk['choices'][0]['text']
            output += text
            if callback:
                callback(text)
        return output

    def generate_with_streaming(self, **kwargs):
        with Iteratorize(self.generate, kwargs, callback=None) as generator:
            reply = ''
            for token in generator:
                reply += token
                yield reply
