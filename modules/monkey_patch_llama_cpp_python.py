from typing import Generator, Optional, Sequence

import torch

from modules.cache_utils import handle_llamacpp_prefix_and_streamingllm

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
        reset = handle_llamacpp_prefix_and_streamingllm(self, past_seq, seq, seq_tensor)

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
