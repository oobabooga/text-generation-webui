from pathlib import Path

from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm.llmapi import SamplingParams

from modules import shared
from modules.logging_colors import logger


class TensorRTLLMModel:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(cls, path_to_model):
        path_to_model = Path(f'{shared.args.model_dir}') / Path(path_to_model)

        llm = LLM(
            model=str(path_to_model),
            skip_tokenizer_init=False,
        )

        result = cls()
        result.llm = llm
        result.tokenizer = llm.tokenizer
        return result

    def generate_with_streaming(self, prompt, state):
        sampling_params = SamplingParams(
            max_tokens=state['max_new_tokens'] if not state['auto_max_new_tokens']
                       else state['truncation_length'] - len(shared.tokenizer.encode(prompt)),
            end_id=shared.tokenizer.eos_token_id,
            temperature=state['temperature'],
            top_k=state['top_k'],
            top_p=state['top_p'],
            min_p=state['min_p'],
            repetition_penalty=state['repetition_penalty'],
            presence_penalty=state['presence_penalty'],
            frequency_penalty=state['frequency_penalty'],
            no_repeat_ngram_size=state['no_repeat_ngram_size'] if state['no_repeat_ngram_size'] > 0 else None,
            seed=state['seed'],
            ignore_eos=state['ban_eos_token'],
            add_special_tokens=state['add_bos_token'],
            skip_special_tokens=state['skip_special_tokens'],
        )

        stop_event = state.get('stop_event')
        result = self.llm.generate_async(prompt, sampling_params=sampling_params, streaming=True)

        cumulative_reply = ''
        for output in result:
            if shared.stop_everything or (stop_event and stop_event.is_set()):
                result.abort()
                break

            text_diff = output.outputs[0].text_diff
            if text_diff:
                cumulative_reply += text_diff
                yield cumulative_reply

    def generate(self, prompt, state):
        output = ''
        for output in self.generate_with_streaming(prompt, state):
            pass

        return output

    def unload(self):
        if hasattr(self, 'llm') and self.llm is not None:
            self.llm.shutdown()
            self.llm = None
