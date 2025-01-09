from pathlib import Path

import tensorrt_llm
import torch
from tensorrt_llm.runtime import ModelRunner, ModelRunnerCpp

from modules import shared
from modules.logging_colors import logger
from modules.text_generation import (
    get_max_prompt_length,
    get_reply_from_output_ids
)


class TensorRTLLMModel:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(self, path_to_model):

        path_to_model = Path(f'{shared.args.model_dir}') / Path(path_to_model)
        runtime_rank = tensorrt_llm.mpi_rank()

        # Define model settings
        runner_kwargs = dict(
            engine_dir=str(path_to_model),
            lora_dir=None,
            rank=runtime_rank,
            debug_mode=False,
            lora_ckpt_source="hf",
        )

        if shared.args.cpp_runner:
            logger.info("TensorRT-LLM: Using \"ModelRunnerCpp\"")
            runner_kwargs.update(
                max_batch_size=1,
                max_input_len=shared.args.max_seq_len - 512,
                max_output_len=512,
                max_beam_width=1,
                max_attention_window_size=None,
                sink_token_length=None,
            )
        else:
            logger.info("TensorRT-LLM: Using \"ModelRunner\"")

        # Load the model
        runner_cls = ModelRunnerCpp if shared.args.cpp_runner else ModelRunner
        runner = runner_cls.from_dir(**runner_kwargs)

        result = self()
        result.model = runner
        result.runtime_rank = runtime_rank

        return result

    def generate_with_streaming(self, prompt, state):
        batch_input_ids = []
        input_ids = shared.tokenizer.encode(
            prompt,
            add_special_tokens=True,
            truncation=False,
        )
        input_ids = torch.tensor(input_ids, dtype=torch.int32)
        input_ids = input_ids[-get_max_prompt_length(state):]  # Apply truncation_length
        batch_input_ids.append(input_ids)

        if shared.args.cpp_runner:
            max_new_tokens = min(512, state['max_new_tokens'])
        elif state['auto_max_new_tokens']:
            max_new_tokens = state['truncation_length'] - input_ids.shape[-1]
        else:
            max_new_tokens = state['max_new_tokens']

        with torch.no_grad():
            generator = self.model.generate(
                batch_input_ids,
                max_new_tokens=max_new_tokens,
                max_attention_window_size=None,
                sink_token_length=None,
                end_id=shared.tokenizer.eos_token_id if not state['ban_eos_token'] else -1,
                pad_id=shared.tokenizer.pad_token_id or shared.tokenizer.eos_token_id,
                temperature=state['temperature'],
                top_k=state['top_k'],
                top_p=state['top_p'],
                num_beams=1,
                length_penalty=1.0,
                repetition_penalty=state['repetition_penalty'],
                presence_penalty=state['presence_penalty'],
                frequency_penalty=state['frequency_penalty'],
                stop_words_list=None,
                bad_words_list=None,
                lora_uids=None,
                prompt_table_path=None,
                prompt_tasks=None,
                streaming=not shared.args.cpp_runner,
                output_sequence_lengths=True,
                return_dict=True,
                medusa_choices=None
            )

        torch.cuda.synchronize()

        cumulative_reply = ''
        starting_from = batch_input_ids[0].shape[-1]

        if shared.args.cpp_runner:
            sequence_length = generator['sequence_lengths'][0].item()
            output_ids = generator['output_ids'][0][0][:sequence_length].tolist()

            cumulative_reply += get_reply_from_output_ids(output_ids, state, starting_from=starting_from)
            starting_from = sequence_length
            yield cumulative_reply
        else:
            for curr_outputs in generator:
                if shared.stop_everything:
                    break

                sequence_length = curr_outputs['sequence_lengths'][0].item()
                output_ids = curr_outputs['output_ids'][0][0][:sequence_length].tolist()

                cumulative_reply += get_reply_from_output_ids(output_ids, state, starting_from=starting_from)
                starting_from = sequence_length
                yield cumulative_reply

    def generate(self, prompt, state):
        output = ''
        for output in self.generate_with_streaming(prompt, state):
            pass

        return output
