import copy
import os
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

import modules.shared as shared
from modules.callbacks import Iteratorize

from ast import literal_eval

np.set_printoptions(precision=4, suppress=True, linewidth=200)

os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '1' if shared.args.rwkv_cuda_on else '0'  # use CUDA kernel for seq mode (much faster)

from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS


class RWKVModel:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(self, path, dtype="fp16", device="cuda", world=False):
        tokenizer_path = str(Path(f"{path.parent}/20B_tokenizer.json")) if not world else 'rwkv_vocab_v20230424'
        if shared.args.rwkv_strategy is None:
            model = RWKV(model=str(path), strategy=f'{device} {dtype}')
        else:
            model = RWKV(model=str(path), strategy=shared.args.rwkv_strategy)

        pipeline = PIPELINE(model, tokenizer_path)
        result = self()
        result.pipeline = pipeline
        result.model = model
        result.cached_context = ""
        result.cached_model_state = None
        result.cached_output_logits = None
        return result

    def generate(self, prompt, state, callback=None):
        args = PIPELINE_ARGS(
            temperature=state['temperature'],
            top_p=state['top_p'],
            top_k=state['top_k'],
            alpha_frequency=0.1,  # Frequency Penalty (as in GPT-3)
            alpha_presence=0.1,  # Presence Penalty (as in GPT-3)
            token_ban=[0],  # ban the generation of some tokens
            token_stop=[]
        )

        if self.cached_context != "":
            if prompt.startswith(self.cached_context):
                prompt = prompt[len(self.cached_context):]
            else:
                self.cached_context = ""
                self.cached_model_state = None
                self.cached_output_logits = None

        # out = self.pipeline.generate(prompt, token_count=state['max_new_tokens'], args=args, callback=callback)
        out = self.generate_from_cached_state(prompt, token_count=state['max_new_tokens'], args=args, callback=callback)
        return out

    def generate_with_streaming(self, *args, **kwargs):
        with Iteratorize(self.generate, args, kwargs, callback=None) as generator:
            reply = ''
            for token in generator:
                reply += token
                yield reply

    # Similar to the PIPELINE.generate, but lets us maintain the cached_model_state
    def generate_from_cached_state(self, ctx="", token_count=20, args=None, callback=None):
        all_tokens = []
        out_str = ''
        occurrence = {}
        state = copy.deepcopy(self.cached_model_state) if self.cached_model_state is not None else None

        # if we ended up with an empty context, just reuse the cached logits
        # this can happen if a user undoes a message and then sends the exact message again
        # in that case the full context ends up being the same as the cached_context, so the remaining context is empty.
        if ctx == "":
            out = self.cached_output_logits

        token = None
        for i in range(token_count):
            # forward
            tokens = self.pipeline.encode(ctx) if i == 0 else [token]
            while len(tokens) > 0:
                out, state = self.model.forward(tokens[:args.chunk_len], state)
                tokens = tokens[args.chunk_len:]
            if i == 0:
                begin_token = len(all_tokens)
                last_token_posi = begin_token
            # cache the model state after scanning the context
            # we don't cache the state after processing our own generated tokens because
            # the output string might be post-processed arbitrarily. Therefore, what's fed into the model
            # on the next round of chat might be slightly different what what it output on the previous round
            if i == 0:
                self.cached_context += ctx
                self.cached_model_state = copy.deepcopy(state)
                self.cached_output_logits = copy.deepcopy(out)

            # adjust probabilities
            for n in args.token_ban:
                out[n] = -float('inf')

            for n in occurrence:
                out[n] -= (args.alpha_presence + occurrence[n] * args.alpha_frequency)

            # sampler
            token = self.pipeline.sample_logits(out, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
            if token in args.token_stop:
                break

            all_tokens += [token]
            if token not in occurrence:
                occurrence[token] = 1
            else:
                occurrence[token] += 1

            # output
            tmp = self.pipeline.decode(all_tokens[last_token_posi:])
            if '\ufffd' not in tmp:  # is valid utf-8 string?
                if callback:
                    callback(tmp)

                out_str += tmp
                last_token_posi = begin_token + i + 1
        return out_str


class RWKVTokenizer:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(self, path):
        tokenizer_path = path / "20B_tokenizer.json"
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        result = self()
        result.tokenizer = tokenizer
        return result

    def encode(self, prompt):
        return self.tokenizer.encode(prompt).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)


class RWKVWorldTokenizer:
    __slots__ = ('tok2val', 'root')

    def __init__(self, file_name) -> None:
        self.tok2val = {}
        self.root = {}

        with open(file_name, 'rt', encoding = 'utf-8') as file:
            for line in file:
                token_str, value = line.rstrip().split(' ', 1)
                value, len_str = value.rsplit(' ', 1)
                value = literal_eval(value)
                value = value if isinstance(value, bytes) else value.encode('utf-8')
                assert len(value) == int(len_str)
                self.add_token(int(token_str), value)

    def add_token(self, token, value) -> None:
        self.tok2val[token] = value
        pos = self.root
        for byte in value[:-1]: pos = pos.setdefault(byte, (None, {}))[1]
        pos.setdefault(value[-1], (token, {}))

    def next_token(self, src):
        last_token = None
        last = self.root
        for i in range(0, len(src)):
            if current := last.get(src[i]):
                if token := current[0]: last_token = token
                last = current[1]
            else: break
        return last_token

    def encode_bytes(self, src):
        start, stop, tokens = 0, len(src), []
        while start < stop:
            last_token = None
            last = self.root

            for i in range(start, stop):
                if current := last.get(src[i]):
                    if token := current[0]:
                        last_token = token
                        start = i + 1
                    last = current[1]
                else: break

            if last_token: tokens.append(last_token)
            else: break
        return tokens

    def decode_bytes(self, tokens):
        return b''.join(map(self.tok2val.__getitem__, tokens))

    def encode(self, src):
        return self.encode_bytes(src.encode('utf-8'))

    def decode(self, tokens):
        return self.decode_bytes(tokens).decode('utf-8')
