'''
Based on
https://github.com/abetlen/llama-cpp-python

Documentation:
https://abetlen.github.io/llama-cpp-python/
'''

import re
from functools import partial

from llama_cpp import Llama, LlamaCache, LogitsProcessorList

from modules import shared
from modules.callbacks import Iteratorize
from modules.logging_colors import logger


def ban_eos_logits_processor(eos_token, input_ids, logits):
    logits[eos_token] = -float('inf')
    return logits


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

        logger.info("Cache capacity is " + str(cache_capacity) + " bytes")
        params = {
            'model_path': str(path),
            'n_ctx': shared.args.n_ctx,
            'seed': int(shared.args.llama_cpp_seed),
            'n_threads': shared.args.threads or None,
            'n_batch': shared.args.n_batch,
            'use_mmap': not shared.args.no_mmap,
            'use_mlock': shared.args.mlock,
            'n_gpu_layers': shared.args.n_gpu_layers
        }

        result.model = Llama(**params)
        if cache_capacity > 0:
            result.model.set_cache(LlamaCache(capacity_bytes=cache_capacity))

        # This is ugly, but the model and the tokenizer are the same object in this library.
        return result, result

    def encode(self, string):
        if type(string) is str:
            string = string.encode()

        return self.model.tokenize(string)

    def decode(self, tokens):
        return self.model.detokenize(tokens)

    def generate(self, prompt, state, callback=None):
        prompt = prompt if type(prompt) is str else prompt.decode()
        completion_chunks = self.model.create_completion(
            prompt=prompt,
            max_tokens=state['max_new_tokens'],
            temperature=state['temperature'],
            top_p=state['top_p'],
            top_k=state['top_k'],
            repeat_penalty=state['repetition_penalty'],
            tfs_z=state['tfs'],
            mirostat_mode=int(state['mirostat_mode']),
            mirostat_tau=state['mirostat_tau'],
            mirostat_eta=state['mirostat_eta'],
            stream=True,
            logits_processor=LogitsProcessorList([
                partial(ban_eos_logits_processor, self.model.token_eos()),
            ]) if state['ban_eos_token'] else None,
        )

        output = ""
        for completion_chunk in completion_chunks:
            text = completion_chunk['choices'][0]['text']
            output += text
            if callback:
                callback(text)

        return output

    def generate_with_streaming(self, *args, **kwargs):
        with Iteratorize(self.generate, args, kwargs, callback=None) as generator:
            reply = ''
            for token in generator:
                reply += token
                yield reply
