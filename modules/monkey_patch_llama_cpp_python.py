from typing import Generator, Optional, Sequence

import torch
from llama_cpp import LlamaGrammar, LogitsProcessorList, StoppingCriteriaList

from modules import shared
from modules.cache_utils import (
    find_longest_common_substring_indices,
    find_prefix_length
)

try:
    import llama_cpp
except:
    llama_cpp = None

try:
    import llama_cpp_cuda
except:
    llama_cpp_cuda = None


if llama_cpp is not None:
    from llama_cpp import (
        LlamaGrammar,
        LogitsProcessorList,
        StoppingCriteriaList
    )
else:
    from llama_cpp_cuda import (
        LlamaGrammar,
        LogitsProcessorList,
        StoppingCriteriaList
    )


def my_generate(
    self,
    tokens: Sequence[int],
    top_k: int = 40,
    top_p: float = 0.95,
    min_p: float = 0.05,
    typical_p: float = 1.0,
    temp: float = 0.80,
    repeat_penalty: float = 1.1,
    reset: bool = True,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    tfs_z: float = 1.0,
    mirostat_mode: int = 0,
    mirostat_tau: float = 5.0,
    mirostat_eta: float = 0.1,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    grammar: Optional[LlamaGrammar] = None,
) -> Generator[int, Optional[Sequence[int]], None]:

    '''
    Customized Llama.generate to include StreamingLLM.
    If any new parameters are added to upstream Llama.generate in the future,
    they must be added here as well.
    '''

    seq = tokens
    seq_tensor = torch.tensor(seq)
    past_seq = torch.tensor(self._input_ids)

    if reset and self.n_tokens > 0:
        if shared.args.streaming_llm:
            i1, i2, j1, j2 = find_longest_common_substring_indices(past_seq.tolist(), seq)
            overlap_length = i2 - i1 + 1

            # A removed chunk has been found
            if i1 > 0:
                reset = False

                prefix_length = find_prefix_length(past_seq[:i1], seq_tensor[:j1])
                sink_length = prefix_length
                if sink_length < shared.args.attention_sink_size:
                    sink_length = shared.args.attention_sink_size

                removed_length = i1 - sink_length

                matching_prefix = past_seq[:prefix_length]
                removed_chunk = past_seq[sink_length:i1]
                overlapping_sequence = seq_tensor[j1:j2+1]
                added_chunk = seq_tensor[j2+1:]

                print('\n\n')
                print('MATCHING PREFIX=', repr(shared.tokenizer.decode(matching_prefix)))
                print('REMOVED CHUNK=', repr(shared.tokenizer.decode(removed_chunk)))
                # print('OVERLAPPING SEQUENCE=', repr(shared.tokenizer.decode(overlapping_sequence)))
                print('ADDED CHUNK=', repr(shared.tokenizer.decode(added_chunk)))
                print('\n\n')

                # Remove interval [sink_length, sink_length + removed_length) from the context
                # Subtract removed_length from self.n_tokens
                self._ctx.kv_cache_seq_rm(0, sink_length, sink_length+removed_length)
                self._ctx.kv_cache_seq_shift(0, sink_length+removed_length, -1, -removed_length)

                self.n_tokens -= removed_length
                self.eval(seq[prefix_length+overlap_length:])

            # No removed chunk has been found
            else:
                prefix_length = find_prefix_length(past_seq, seq_tensor)
                if prefix_length > 0:
                    reset = False
                    self.n_tokens = prefix_length
                    if len(seq_tensor) - prefix_length > 0:
                        self.eval(seq[prefix_length:])

        else:
            prefix_length = find_prefix_length(past_seq, seq_tensor)
            if prefix_length > 0:
                reset = False
                self.n_tokens = prefix_length
                if len(seq_tensor) - prefix_length > 0:
                    self.eval(seq[prefix_length:])

    if reset:
        self.reset()
        self.eval(seq)

    if grammar is not None:
        grammar.reset()

    while True:
        token = self.sample(
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            typical_p=typical_p,
            temp=temp,
            repeat_penalty=repeat_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            tfs_z=tfs_z,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            logits_processor=logits_processor,
            grammar=grammar,
        )
        if stopping_criteria is not None and stopping_criteria(
            self._input_ids, self._scores[-1, :]
        ):
            return
        tokens_or_none = yield token
        tokens = [token]
        if tokens_or_none is not None:
            tokens.extend(tokens_or_none)

        self.eval(tokens)


if llama_cpp:
    llama_cpp.Llama.generate = my_generate

if llama_cpp_cuda:
    llama_cpp_cuda.Llama.generate = my_generate
