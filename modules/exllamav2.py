import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import fastparquet
import pandas
import torch
import torch.nn.functional as F

from modules import shared
from modules.logging_colors import logger
from modules.models import clear_torch_cache
from modules.relative_imports import RelativeImport
from modules.text_generation import get_max_prompt_length

with RelativeImport("repositories/exllamav2"):
    from conversion.quantize import list_live_tensors
    from conversion.tokenize import get_tokens
    from exllamav2 import (
        ExLlamaV2,
        ExLlamaV2Cache,
        ExLlamaV2Config,
        ExLlamaV2Tokenizer,
        model_init
    )

torch.cuda._lazy_init()

class Exllamav2Model:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(self, path_to_model):

        path_to_model = Path(f'{shared.args.model_dir}') / Path(path_to_model)
        tokenizer_model_path = path_to_model / "tokenizer.model"
        model_config_path = path_to_model / "config.json"

        # Find the model checkpoint
        model_path = None
        for ext in ['.safetensors', '.pt', '.bin']:
            found = list(path_to_model.glob(f"*{ext}"))
            if len(found) > 0:
                if len(found) > 1:
                    logger.warning(f'More than one {ext} model has been found. The last one will be selected. It could be wrong.')

                model_path = found[-1]
                break

        config = ExLlamaV2Config()
        config.model_dir = path_to_model
        config.prepare()

        config.max_seq_len = shared.args.max_seq_len
        model = ExLlamaV2(config)

        split = None
        if shared.args.gpu_split:
            split = [float(alloc) for alloc in args.gpu_split.split(",")]

        model.load(split)

        tokenizer = ExLlamaV2Tokenizer(config)

        cache = ExLlamaV2Cache(model)

        result = self()
        result.model = model 
        result.cache = cache
        return result, tokenizer

    def generate_with_streaming(self, prompt, state):
        with torch.inference_mode():
            ids = shared.tokenizer.encode(prompt)
            initial_len = ids.shape[-1]
            tokens_prompt = ids.shape[-1]
            self.model.forward(ids[:, -1:])

            if state['auto_max_new_tokens']:
                max_new_tokens = state['truncation_length'] - ids.shape[-1]
            else:
                max_new_tokens = state['max_new_tokens']

            if ids.shape[-1] > 1:
                self.model.forward(ids[:, :-1], self.cache, preprocess_only = True)

            torch.cuda.synchronize()
            time_prompt = time.time()
            for i in range(max_new_tokens):

                # text1 = shared.tokenizer.decode(ids[:, -2:])[0]

                logits = self.model.forward(ids[:, -1:], self.cache)
                sample = torch.argmax(logits[0, -1]).cpu().unsqueeze(0).unsqueeze(0)
                ids = torch.cat((ids, sample), dim = -1)

                # text2 = shared.tokenizer.decode(ids[:, -3:])[0]
                text2 = shared.tokenizer.decode(ids[:,initial_len:])[0]
                # text2 = text2[len(text1):]

                yield text2

    def generate(self, prompt, state):
        output = ''
        for output in self.generate_with_streaming(prompt, state):
            pass

        return output

    def encode(self, string, **kwargs):
        result = shared.tokenizer.encode(string)
        print(result)

    def decode(self, string, **kwargs):
        return shared.tokenizer.decode(string)[0]
