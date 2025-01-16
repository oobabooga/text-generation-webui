import re
import io
import itertools
from contextlib import redirect_stderr
from functools import partial

import numpy as np
import torch

from modules import shared
from modules.callbacks import Iteratorize
from modules.llama_cpp_python_hijack import llama_cpp_lib
from modules.logging_colors import logger
from modules.text_generation import get_max_prompt_length

llamacpp_quant_mapping = {
    'f32': 0,
    'fp16': 1,
    'q4_0': 2,
    'q4_1': 3,
    'q5_0': 6,
    'q5_1': 7,
    'q8_0': 8,
    'q8_1': 9,
    'q2_k': 10,
    'q3_k': 11,
    'q4_k': 12,
    'q5_k': 13,
    'q6_k': 14,
    'q8_k': 15,
    'iq4_nl': 20,
    'bf16': 30,
}

llamacpp_valid_cache_types = {'fp16', 'q8_0', 'q4_0'}


def get_llamacpp_cache_type_for_string(quant_type: str):
    quant_type = quant_type.lower()
    if quant_type in llamacpp_valid_cache_types:
        return llamacpp_quant_mapping[quant_type]
    else:
        raise ValueError(f"Invalid cache type for llama.cpp: {quant_type}. Valid options are: fp16, q8_0, q4_0.")


def ban_eos_logits_processor(eos_token, input_ids, logits):
    logits[eos_token] = -float('inf')
    return logits


def custom_token_ban_logits_processor(token_ids, input_ids, logits):
    for token_id in token_ids:
        logits[token_id] = -float('inf')

    return logits


class LlamaSmallModelDraft(llama_cpp_lib().llama_speculative.LlamaDraftModel):
    """
    modified from https://gist.github.com/acasto/dce5f559fbe5da5ceed2c62db7afc262

    Optimized draft model for speculative decoding.

    Key Changes:
    - Removed unnecessary prints and I/O overhead.
    - Using greedy decoding parameters (top_k=1, top_p=1.0) if acceptable.
    - Using itertools.islice to grab tokens in a single step rather than a loop.
    - Consider adjusting n_ctx, n_batch, and model quantization to improve performance.
    """

    def __init__(
            self,
            model_path: str,
            num_pred_tokens: int = 5,
            temperature: float = 0.0,
            n_ctx: int = 2048,
            n_batch: int = 512,
    ):
        # Suppress unwanted stderr output during model load
        f = io.StringIO()
        with redirect_stderr(f):
            self.draft_model = llama_cpp_lib().Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_batch=n_batch,
                n_gpu_layers=-1,
                verbose=False
            )
        self.num_pred_tokens = num_pred_tokens
        self.temperature = temperature

    def __call__(
            self,
            input_ids,
            /,
            **kwargs
    ):
        # Convert numpy array to list for llama_cpp
        input_tokens = input_ids.tolist()

        # Generate tokens greedily or with minimal sampling complexity for speed
        generated = itertools.islice(
            self.draft_model.original_generate(
                tokens=input_tokens,
                temp=self.temperature,
                top_k=1,      # Greedy decoding
                top_p=1.0,    # Greedy decoding
                reset=True,   # Reset state for a fresh decode
            ),
            self.num_pred_tokens
        )

        # Collect and convert to a numpy array
        draft_tokens = np.fromiter(generated, dtype=np.intc, count=self.num_pred_tokens)
        return draft_tokens


class LlamaCppModel:
    def __init__(self):
        self.initialized = False
        self.grammar_string = ''
        self.grammar = None

    def __del__(self):
        del self.model

    @classmethod
    def from_pretrained(self, path):

        Llama = llama_cpp_lib().Llama
        LlamaCache = llama_cpp_lib().LlamaCache
        LlamaPromptLookupDecoding = llama_cpp_lib().llama_speculative.LlamaPromptLookupDecoding

        result = self()
        cache_capacity = 0
        if shared.args.cache_capacity is not None:
            if 'GiB' in shared.args.cache_capacity:
                cache_capacity = int(re.sub('[a-zA-Z]', '', shared.args.cache_capacity)) * 1000 * 1000 * 1000
            elif 'MiB' in shared.args.cache_capacity:
                cache_capacity = int(re.sub('[a-zA-Z]', '', shared.args.cache_capacity)) * 1000 * 1000
            else:
                cache_capacity = int(shared.args.cache_capacity)

        if cache_capacity > 0:
            logger.info("Cache capacity is " + str(cache_capacity) + " bytes")

        if shared.args.tensor_split is None or shared.args.tensor_split.strip() == '':
            tensor_split_list = None
        else:
            tensor_split_list = [float(x) for x in shared.args.tensor_split.strip().split(",")]

        if shared.args.num_pred_tokens > 0 and shared.args.draft_model.endswith(".gguf"):
            draft_model = LlamaSmallModelDraft(model_path=f"{shared.args.model_dir}/{shared.args.draft_model}", num_pred_tokens=shared.args.num_pred_tokens)
        elif shared.args.num_pred_tokens > 0:
            draft_model = LlamaPromptLookupDecoding(num_pred_tokens=shared.args.num_pred_tokens)
        else:
            draft_model = None

        params = {
            'model_path': str(path),
            'n_ctx': shared.args.n_ctx,
            'n_threads': shared.args.threads or None,
            'n_threads_batch': shared.args.threads_batch or None,
            'n_batch': shared.args.n_batch,
            'use_mmap': not shared.args.no_mmap,
            'use_mlock': shared.args.mlock,
            'mul_mat_q': not shared.args.no_mul_mat_q,
            'numa': shared.args.numa,
            'n_gpu_layers': shared.args.n_gpu_layers,
            'rope_freq_base': shared.args.rope_freq_base,
            'tensor_split': tensor_split_list,
            'rope_freq_scale': 1.0 / shared.args.compress_pos_emb,
            'offload_kqv': not shared.args.no_offload_kqv,
            'split_mode': 1 if not shared.args.row_split else 2,
            'flash_attn': shared.args.flash_attn,
            'draft_model': draft_model
        }

        if shared.args.cache_type != 'fp16':
            params["type_k"] = get_llamacpp_cache_type_for_string(shared.args.cache_type)
            params["type_v"] = get_llamacpp_cache_type_for_string(shared.args.cache_type)

        result.model = Llama(**params)
        if cache_capacity > 0:
            result.model.set_cache(LlamaCache(capacity_bytes=cache_capacity))

        # This is ugly, but the model and the tokenizer are the same object in this library.
        return result, result

    def encode(self, string):
        if type(string) is str:
            string = string.encode()

        return self.model.tokenize(string)

    def decode(self, ids, **kwargs):
        detokenized = self.model.detokenize(ids)
        try:
            # Attempt strict UTF-8 decoding first
            return detokenized.decode('utf-8', 'strict')
        except UnicodeDecodeError as e:
            # Log the error and fall back to UTF-8 with replacement
            logger.warning(f"Invalid UTF-8 in detokenized output. Using replacement characters.\n{e}")
            return detokenized.decode('utf-8', 'replace')

    def get_logits(self, tokens):
        self.model.reset()
        self.model.eval(tokens)
        logits = self.model._scores
        logits = np.expand_dims(logits, 0)  # batch dim is expected
        return torch.tensor(logits, dtype=torch.float32)

    def load_grammar(self, string):
        if string != self.grammar_string:
            self.grammar_string = string
            if string.strip() != '':
                self.grammar = llama_cpp_lib().LlamaGrammar.from_string(string)
            else:
                self.grammar = None

    def generate(self, prompt, state, callback=None):
        LogitsProcessorList = llama_cpp_lib().LogitsProcessorList
        prompt = prompt if type(prompt) is str else prompt.decode()

        # Handle truncation
        prompt = self.encode(prompt)
        prompt = prompt[-get_max_prompt_length(state):]
        prompt = self.decode(prompt)

        self.load_grammar(state['grammar_string'])
        logit_processors = LogitsProcessorList()
        if state['ban_eos_token']:
            logit_processors.append(partial(ban_eos_logits_processor, self.model.token_eos()))

        if state['custom_token_bans']:
            to_ban = [int(x) for x in state['custom_token_bans'].split(',')]
            if len(to_ban) > 0:
                logit_processors.append(partial(custom_token_ban_logits_processor, to_ban))

        completion_chunks = self.model.create_completion(
            prompt=prompt,
            max_tokens=state['max_new_tokens'],
            temperature=state['temperature'],
            top_p=state['top_p'] if state['top_p'] < 1 else 0.999,
            min_p=state['min_p'],
            typical_p=state['typical_p'],
            frequency_penalty=state['frequency_penalty'],
            presence_penalty=state['presence_penalty'],
            repeat_penalty=state['repetition_penalty'],
            top_k=state['top_k'],
            stream=True,
            seed=int(state['seed']) if state['seed'] != -1 else None,
            tfs_z=state['tfs'],
            mirostat_mode=int(state['mirostat_mode']),
            mirostat_tau=state['mirostat_tau'],
            mirostat_eta=state['mirostat_eta'],
            logits_processor=logit_processors,
            grammar=self.grammar
        )

        output = ""
        for completion_chunk in completion_chunks:
            if shared.stop_everything:
                break

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
