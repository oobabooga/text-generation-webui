import ast
import copy
import html
import pprint
import random
import time
import traceback

import numpy as np
import torch
import transformers
from transformers import (
    LogitsProcessorList,
    is_torch_npu_available,
    is_torch_xpu_available
)

import modules.shared as shared
from modules import models
from modules.cache_utils import process_llamacpp_cache
from modules.callbacks import (
    Iteratorize,
    Stream,
    _StopEverythingStoppingCriteria
)
from modules.extensions import apply_extensions
from modules.grammar.grammar_utils import initialize_grammar
from modules.grammar.logits_process import GrammarConstrainedLogitsProcessor
from modules.html_generator import generate_basic_html
from modules.logging_colors import logger
from modules.models import clear_torch_cache, load_model


def generate_reply(*args, **kwargs):
    if shared.args.idle_timeout > 0 and shared.model is None and shared.model_name not in [None, 'None']:
        shared.model, shared.tokenizer = load_model(shared.model_name)

    shared.generation_lock.acquire()
    try:
        for result in _generate_reply(*args, **kwargs):
            yield result
    finally:
        models.last_generation_time = time.time()
        shared.generation_lock.release()


def _generate_reply(question, state, stopping_strings=None, is_chat=False, escape_html=False, for_ui=False):

    # Find the appropriate generation function
    generate_func = apply_extensions('custom_generate_reply')
    if generate_func is None:
        if shared.model_name == 'None' or shared.model is None:
            logger.error("No model is loaded! Select one in the Model tab.")
            yield ''
            return

        if shared.model.__class__.__name__ in ['LlamaCppModel', 'Exllamav2Model', 'TensorRTLLMModel']:
            generate_func = generate_reply_custom
        else:
            generate_func = generate_reply_HF

    if generate_func != generate_reply_HF and shared.args.verbose:
        logger.info("PROMPT=")
        print_prompt(question)

    # Prepare the input
    original_question = question
    if not is_chat:
        state = apply_extensions('state', state)
        question = apply_extensions('input', question, state)

    # Find the stopping strings
    all_stop_strings = []
    for st in (stopping_strings, state['custom_stopping_strings']):
        if type(st) is str:
            st = ast.literal_eval(f"[{st}]")

        if type(st) is list and len(st) > 0:
            all_stop_strings += st

    shared.stop_everything = False
    clear_torch_cache()
    seed = set_manual_seed(state['seed'])
    last_update = -1
    reply = ''
    is_stream = state['stream']
    if len(all_stop_strings) > 0 and not state['stream']:
        state = copy.deepcopy(state)
        state['stream'] = True

    min_update_interval = 0
    if state.get('max_updates_second', 0) > 0:
        min_update_interval = 1 / state['max_updates_second']

    # Generate
    for reply in generate_func(question, original_question, seed, state, stopping_strings, is_chat=is_chat):
        reply, stop_found = apply_stopping_strings(reply, all_stop_strings)
        if escape_html:
            reply = html.escape(reply)

        if is_stream:
            cur_time = time.time()

            # Limit number of tokens/second to make text readable in real time
            if state['max_tokens_second'] > 0:
                diff = 1 / state['max_tokens_second'] - (cur_time - last_update)
                if diff > 0:
                    time.sleep(diff)

                last_update = time.time()
                yield reply

            # Limit updates to avoid lag in the Gradio UI
            # API updates are not limited
            else:
                if cur_time - last_update > min_update_interval:
                    last_update = cur_time
                    yield reply

                yield reply

        if stop_found or (state['max_tokens_second'] > 0 and shared.stop_everything):
            break

    if not is_chat:
        reply = apply_extensions('output', reply, state)

    yield reply


def encode(prompt, add_special_tokens=True, add_bos_token=True, truncation_length=None):
    if shared.tokenizer is None:
        raise ValueError('No tokenizer is loaded')

    if shared.model.__class__.__name__ in ['LlamaCppModel', 'Exllamav2Model', 'TensorRTLLMModel']:
        input_ids = shared.tokenizer.encode(str(prompt))
        if shared.model.__class__.__name__ not in ['Exllamav2Model']:
            input_ids = np.array(input_ids).reshape(1, len(input_ids))
    else:
        input_ids = shared.tokenizer.encode(str(prompt), return_tensors='pt', add_special_tokens=add_special_tokens)

        if hasattr(shared.tokenizer, 'bos_token_id') and shared.tokenizer.bos_token_id is not None:
            if add_bos_token:
                if (len(input_ids[0]) > 0 and input_ids[0][0] != shared.tokenizer.bos_token_id) or len(input_ids[0]) == 0:
                    # Add a missing bos token (it may not have been added due to faulty model metadata)
                    bos_tensor = torch.tensor([[shared.tokenizer.bos_token_id]])
                    input_ids = torch.cat((bos_tensor, input_ids), 1)

                # Prevent double bos token due to jinja templates with <s> somewhere
                while len(input_ids[0]) > 1 and input_ids[0][0] == shared.tokenizer.bos_token_id and input_ids[0][1] == shared.tokenizer.bos_token_id:
                    input_ids = input_ids[:, 1:]
            else:
                # Remove any bos token that may have been added
                while len(input_ids[0]) > 0 and input_ids[0][0] == shared.tokenizer.bos_token_id:
                    input_ids = input_ids[:, 1:]

    # Handling truncation
    if truncation_length is not None:
        input_ids = input_ids[:, -truncation_length:]

    if shared.model.__class__.__name__ in ['LlamaCppModel', 'Exllamav2Model', 'TensorRTLLMModel'] or shared.args.cpu:
        return input_ids
    elif shared.args.deepspeed:
        import deepspeed
        return input_ids.to(deepspeed.get_accelerator().current_device_name())
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        return input_ids.to(device)
    elif is_torch_xpu_available():
        return input_ids.to("xpu:0")
    elif is_torch_npu_available():
        return input_ids.to("npu:0")
    else:
        return input_ids.cuda()


def decode(output_ids, skip_special_tokens=True):
    if shared.tokenizer is None:
        raise ValueError('No tokenizer is loaded')

    return shared.tokenizer.decode(output_ids, skip_special_tokens=skip_special_tokens)


def get_encoded_length(prompt):
    length_after_extensions = apply_extensions('tokenized_length', prompt)
    if length_after_extensions is not None:
        return length_after_extensions

    return len(encode(prompt)[0])


def get_token_ids(prompt):
    tokens = encode(prompt)[0]
    decoded_tokens = [shared.tokenizer.decode([i]) for i in tokens]

    output = ''
    for row in list(zip(tokens, decoded_tokens)):
        output += f"{str(int(row[0])).ljust(5)}  -  {repr(row[1])}\n"

    return output


def get_max_prompt_length(state):
    return state['truncation_length'] - state['max_new_tokens']


def generate_reply_wrapper(question, state, stopping_strings=None):
    """
    Returns formatted outputs for the UI
    """
    reply = question if not shared.is_seq2seq else ''
    yield formatted_outputs(reply, shared.model_name)

    for reply in generate_reply(question, state, stopping_strings, is_chat=False, escape_html=True, for_ui=True):
        if not shared.is_seq2seq:
            reply = question + reply

        yield formatted_outputs(reply, shared.model_name)


def formatted_outputs(reply, model_name):
    return html.unescape(reply), generate_basic_html(reply)


def set_manual_seed(seed):
    seed = int(seed)
    if seed == -1:
        seed = random.randint(1, 2**31)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif is_torch_xpu_available():
        torch.xpu.manual_seed_all(seed)
    elif is_torch_npu_available():
        torch.npu.manual_seed_all(seed)

    return seed


def stop_everything_event():
    shared.stop_everything = True


def apply_stopping_strings(reply, all_stop_strings):
    stop_found = False
    for string in all_stop_strings:
        idx = reply.find(string)
        if idx != -1:
            reply = reply[:idx]
            stop_found = True
            break

    if not stop_found:
        # If something like "\nYo" is generated just before "\nYou:"
        # is completed, trim it
        for string in all_stop_strings:
            for j in range(len(string) - 1, 0, -1):
                if reply[-j:] == string[:j]:
                    reply = reply[:-j]
                    break
            else:
                continue

            break

    return reply, stop_found


def get_reply_from_output_ids(output_ids, state=None, starting_from=0):
    reply = decode(output_ids[starting_from:], state['skip_special_tokens'] if state else True)

    # Handle tokenizers that do not add the leading space for the first token
    if (hasattr(shared.tokenizer, 'convert_ids_to_tokens') and len(output_ids) > starting_from) and not reply.startswith(' '):
        first_token = shared.tokenizer.convert_ids_to_tokens(int(output_ids[starting_from]))
        if isinstance(first_token, (bytes,)):
            # try to decode the bytes to a string
            # if it fails, which means it's not a string in this turn, just ignore it
            try:
                first_token = first_token.decode('utf8')
            except UnicodeDecodeError:
                first_token = ''

        if first_token.startswith('▁'):
            reply = ' ' + reply

    return reply


def generate_reply_HF(question, original_question, seed, state, stopping_strings=None, is_chat=False):
    generate_params = {}
    for k in ['max_new_tokens', 'temperature', 'temperature_last', 'dynamic_temperature', 'dynatemp_low', 'dynatemp_high', 'dynatemp_exponent', 'smoothing_factor', 'smoothing_curve', 'top_p', 'min_p', 'top_k', 'repetition_penalty', 'presence_penalty', 'frequency_penalty', 'repetition_penalty_range', 'typical_p', 'tfs', 'top_a', 'guidance_scale', 'penalty_alpha', 'mirostat_mode', 'mirostat_tau', 'mirostat_eta', 'do_sample', 'encoder_repetition_penalty', 'no_repeat_ngram_size', 'dry_multiplier', 'dry_base', 'dry_allowed_length', 'dry_sequence_breakers', 'xtc_threshold', 'xtc_probability']:
        if k in state:
            generate_params[k] = state[k]

    if isinstance(state['sampler_priority'], list) and len(state['sampler_priority']) > 0:
        generate_params['sampler_priority'] = state['sampler_priority']
    elif isinstance(state['sampler_priority'], str) and state['sampler_priority'].strip() != '':
        generate_params['sampler_priority'] = [x.strip() for x in state['sampler_priority'].replace('\n', ',').split(',') if x.strip()]

    if state['negative_prompt'] != '':
        generate_params['negative_prompt_ids'] = encode(state['negative_prompt'])

    if state['prompt_lookup_num_tokens'] > 0:
        generate_params['prompt_lookup_num_tokens'] = state['prompt_lookup_num_tokens']

    for k in ['epsilon_cutoff', 'eta_cutoff']:
        if state[k] > 0:
            generate_params[k] = state[k] * 1e-4

    if state['ban_eos_token']:
        generate_params['suppress_tokens'] = [shared.tokenizer.eos_token_id]

    if state['custom_token_bans']:
        to_ban = [int(x) for x in state['custom_token_bans'].split(',')]
        if len(to_ban) > 0:
            if generate_params.get('suppress_tokens', None):
                generate_params['suppress_tokens'] += to_ban
            else:
                generate_params['suppress_tokens'] = to_ban

    generate_params.update({'use_cache': not shared.args.no_cache})
    if shared.args.deepspeed:
        generate_params.update({'synced_gpus': True})

    # Encode the input
    input_ids = encode(question, add_bos_token=state['add_bos_token'], truncation_length=get_max_prompt_length(state))
    output = input_ids[0]
    cuda = not any((shared.args.cpu, shared.args.deepspeed))
    if state['auto_max_new_tokens']:
        generate_params['max_new_tokens'] = state['truncation_length'] - input_ids.shape[-1]

    # Add the encoded tokens to generate_params
    question, input_ids, inputs_embeds = apply_extensions('tokenizer', state, question, input_ids, None)
    original_input_ids = input_ids
    generate_params.update({'inputs': input_ids})
    if inputs_embeds is not None:
        generate_params.update({'inputs_embeds': inputs_embeds})

    # Stopping criteria / eos token
    eos_token_ids = [shared.tokenizer.eos_token_id] if shared.tokenizer.eos_token_id is not None else []
    generate_params['eos_token_id'] = eos_token_ids
    generate_params['stopping_criteria'] = transformers.StoppingCriteriaList()
    generate_params['stopping_criteria'].append(_StopEverythingStoppingCriteria())

    # Logits processor
    processor = state.get('logits_processor', LogitsProcessorList([]))
    if not isinstance(processor, LogitsProcessorList):
        processor = LogitsProcessorList([processor])

    # Grammar
    if state['grammar_string'].strip() != '':
        grammar = initialize_grammar(state['grammar_string'])
        grammar_processor = GrammarConstrainedLogitsProcessor(grammar)
        processor.append(grammar_processor)

    apply_extensions('logits_processor', processor, input_ids)
    generate_params['logits_processor'] = processor

    if shared.args.verbose:
        logger.info("GENERATE_PARAMS=")
        filtered_params = {key: value for key, value in generate_params.items() if not isinstance(value, torch.Tensor)}
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(filtered_params)
        print()

        logger.info("PROMPT=")
        print_prompt(decode(input_ids[0], skip_special_tokens=False))

    # Handle StreamingLLM for llamacpp_HF
    if shared.model.__class__.__name__ == 'LlamacppHF' and shared.args.streaming_llm:
        tmp = process_llamacpp_cache(shared.model.model, input_ids[-1].tolist(), shared.model.model._input_ids.tolist())
        shared.model.past_seq = torch.tensor(tmp)
        shared.model.save_cache()

    t0 = time.time()
    try:
        if not is_chat and not shared.is_seq2seq:
            yield ''

        # Generate the entire reply at once.
        if not state['stream']:
            with torch.no_grad():
                output = shared.model.generate(**generate_params)[0]
                if cuda:
                    output = output.cuda()

            starting_from = 0 if shared.is_seq2seq else len(input_ids[0])
            yield get_reply_from_output_ids(output, state, starting_from=starting_from)

        # Stream the reply 1 token at a time.
        # This is based on the trick of using 'stopping_criteria' to create an iterator.
        else:

            def generate_with_callback(callback=None, *args, **kwargs):
                kwargs['stopping_criteria'].append(Stream(callback_func=callback))
                clear_torch_cache()
                with torch.no_grad():
                    shared.model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(generate_with_callback, [], kwargs, callback=None)

            with generate_with_streaming(**generate_params) as generator:
                cumulative_reply = ''
                starting_from = 0 if shared.is_seq2seq else len(input_ids[0])
                for output in generator:
                    if output[-1] in eos_token_ids:
                        break

                    new_content = get_reply_from_output_ids(output, state, starting_from=starting_from)
                    # check the partial unicode character
                    if chr(0xfffd) in new_content:
                        continue

                    cumulative_reply += new_content
                    starting_from = len(output)
                    yield cumulative_reply

    except Exception:
        traceback.print_exc()
    finally:
        t1 = time.time()
        original_tokens = len(original_input_ids[0])
        new_tokens = len(output) - (original_tokens if not shared.is_seq2seq else 0)
        print(f'Output generated in {(t1-t0):.2f} seconds ({new_tokens/(t1-t0):.2f} tokens/s, {new_tokens} tokens, context {original_tokens}, seed {seed})')
        return


def generate_reply_custom(question, original_question, seed, state, stopping_strings=None, is_chat=False):
    """
    For models that do not use the transformers library for sampling
    """
    seed = set_manual_seed(state['seed'])

    t0 = time.time()
    reply = ''
    try:
        if not is_chat:
            yield ''

        if not state['stream']:
            reply = shared.model.generate(question, state)
            yield reply
        else:
            for reply in shared.model.generate_with_streaming(question, state):
                yield reply

    except Exception:
        traceback.print_exc()
    finally:
        t1 = time.time()
        original_tokens = len(encode(original_question)[0])
        new_tokens = len(encode(original_question + reply)[0]) - original_tokens
        print(f'Output generated in {(t1-t0):.2f} seconds ({new_tokens/(t1-t0):.2f} tokens/s, {new_tokens} tokens, context {original_tokens}, seed {seed})')
        return


def print_prompt(prompt, max_chars=2000):
    DARK_YELLOW = "\033[38;5;3m"
    RESET = "\033[0m"

    if len(prompt) > max_chars:
        half_chars = max_chars // 2
        hidden_len = len(prompt[half_chars:-half_chars])
        hidden_msg = f"{DARK_YELLOW}[...{hidden_len} characters hidden...]{RESET}"
        print(prompt[:half_chars] + hidden_msg + prompt[-half_chars:])
    else:
        print(prompt)

    print()
