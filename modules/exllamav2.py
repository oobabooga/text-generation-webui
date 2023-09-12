from pathlib import Path

import torch

from modules import shared
from modules.relative_imports import RelativeImport

with RelativeImport("repositories/exllamav2"):
    from exllamav2 import (
        ExLlamaV2,
        ExLlamaV2Cache,
        ExLlamaV2Config,
        ExLlamaV2Tokenizer
    )

torch.cuda._lazy_init()


class Exllamav2Model:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(self, path_to_model):

        path_to_model = Path(f'{shared.args.model_dir}') / Path(path_to_model)

        config = ExLlamaV2Config()
        config.model_dir = path_to_model
        config.prepare()

        config.max_seq_len = shared.args.max_seq_len
        model = ExLlamaV2(config)

        split = None
        if shared.args.gpu_split:
            split = [float(alloc) for alloc in shared.args.gpu_split.split(",")]

        model.load(split)

        tokenizer = ExLlamaV2Tokenizer(config)

        cache = ExLlamaV2Cache(model)

        result = self()
        result.model = model
        result.cache = cache
        return result, tokenizer

    def generate_with_streaming(self, prompt, state):
        with torch.inference_mode():

            self.cache.current_seq_len = 0

            ids = shared.tokenizer.encode(prompt)
            initial_len = ids.shape[-1]
            self.model.forward(ids[:, -1:])

            if state['auto_max_new_tokens']:
                max_new_tokens = state['truncation_length'] - ids.shape[-1]
            else:
                max_new_tokens = state['max_new_tokens']

            if ids.shape[-1] > 1:
                self.model.forward(ids[:, :-1], self.cache, preprocess_only=True)

            torch.cuda.synchronize()
            has_leading_space = False
            for i in range(max_new_tokens):
                logits = self.model.forward(ids[:, -1:], self.cache)
                token = torch.argmax(logits[0, -1]).cpu().unsqueeze(0).unsqueeze(0)
                ids = torch.cat((ids, token), dim=-1)

                if i == 0 and shared.tokenizer.tokenizer.IdToPiece(int(token)).startswith('▁'):
                    has_leading_space = True

                decoded_text = shared.tokenizer.decode(ids[:, initial_len:])[0]
                if has_leading_space:
                    decoded_text = ' ' + decoded_text

                yield decoded_text

    def generate(self, prompt, state):
        output = ''
        for output in self.generate_with_streaming(prompt, state):
            pass

        return output

    def encode(self, string, **kwargs):
        return shared.tokenizer.encode(string)

    def decode(self, string, **kwargs):
        return shared.tokenizer.decode(string)[0]
