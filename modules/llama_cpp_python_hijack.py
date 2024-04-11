from typing import Sequence

from tqdm import tqdm

from modules import shared
from modules.cache_utils import process_llamacpp_cache

try:
    import llama_cpp
except:
    llama_cpp = None

try:
    import llama_cpp_cuda
except:
    llama_cpp_cuda = None

try:
    import llama_cpp_cuda_tensorcores
except:
    llama_cpp_cuda_tensorcores = None


def eval_with_progress(self, tokens: Sequence[int]):
    """
    A copy of

    https://github.com/abetlen/llama-cpp-python/blob/main/llama_cpp/llama.py

    with tqdm to show prompt processing progress.
    """
    assert self._ctx.ctx is not None
    assert self._batch.batch is not None
    self._ctx.kv_cache_seq_rm(-1, self.n_tokens, -1)

    if len(tokens) > 1:
        progress_bar = tqdm(range(0, len(tokens), self.n_batch), desc="Prompt evaluation", leave=False)
    else:
        progress_bar = range(0, len(tokens), self.n_batch)

    for i in progress_bar:
        batch = tokens[i : min(len(tokens), i + self.n_batch)]
        n_past = self.n_tokens
        n_tokens = len(batch)
        self._batch.set_batch(
            batch=batch, n_past=n_past, logits_all=self.context_params.logits_all
        )
        self._ctx.decode(self._batch)
        # Save tokens
        self.input_ids[n_past : n_past + n_tokens] = batch
        # Save logits
        if self.context_params.logits_all:
            rows = n_tokens
            cols = self._n_vocab
            logits = self._ctx.get_logits()[: rows * cols]
            self.scores[n_past : n_past + n_tokens, :].reshape(-1)[: :] = logits
        else:
            rows = 1
            cols = self._n_vocab
            logits = self._ctx.get_logits()[: rows * cols]
            self.scores[n_past + n_tokens - 1, :].reshape(-1)[: :] = logits
        # Update n_tokens
        self.n_tokens += n_tokens


def monkey_patch_generate(lib):

    def my_generate(self, *args, **kwargs):

        if shared.args.streaming_llm:
            new_sequence = args[0]
            past_sequence = self._input_ids

            # Do the cache trimming for StreamingLLM
            process_llamacpp_cache(self, new_sequence, past_sequence)

        for output in self.original_generate(*args, **kwargs):
            yield output

    lib.Llama.original_generate = lib.Llama.generate
    lib.Llama.generate = my_generate


for lib in [llama_cpp, llama_cpp_cuda, llama_cpp_cuda_tensorcores]:
    if lib is not None:
        lib.Llama.eval = eval_with_progress
        monkey_patch_generate(lib)
