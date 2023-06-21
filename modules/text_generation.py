import ast
import random
import re
import time
import traceback

import numpy as np
import torch
import transformers

import modules.shared as shared
from modules.callbacks import (Iteratorize, Stream,
                               _SentinelTokenStoppingCriteria)
from modules.extensions import apply_extensions
from modules.html_generator import generate_4chan_html, generate_basic_html
from modules.logging_colors import logger
from modules.models import clear_torch_cache, local_rank


def generate_reply(*args, **kwargs):
    shared.generation_lock.acquire()
    try:
        for result in _generate_reply(*args, **kwargs):
            yield result
    finally:
        shared.generation_lock.release()


def get_max_prompt_length(state):
    return state['truncation_length'] - state['max_new_tokens']


def encode(prompt, add_special_tokens=True, add_bos_token=True, truncation_length=None):
    if shared.model.__class__.__name__ in ['LlamaCppModel', 'RWKVModel']:
        input_ids = shared.tokenizer.encode(str(prompt))
        input_ids = np.array(input_ids).reshape(1, len(input_ids))
        return input_ids
    else:
        input_ids = shared.tokenizer.encode(str(prompt), return_tensors='pt', add_special_tokens=add_special_tokens)

        # This is a hack for making replies more creative.
        if not add_bos_token and input_ids[0][0] == shared.tokenizer.bos_token_id:
            input_ids = input_ids[:, 1:]

        # Llama adds this extra token when the first character is '\n', and this
        # compromises the stopping criteria, so we just remove it
        if type(shared.tokenizer) is transformers.LlamaTokenizer and input_ids[0][0] == 29871:
            input_ids = input_ids[:, 1:]

    # Handling truncation
    if truncation_length is not None:
        input_ids = input_ids[:, -truncation_length:]

    if shared.model.__class__.__name__ in ['LlamaCppModel', 'RWKVModel', 'ExllamaModel'] or shared.args.cpu:
        return input_ids
    elif shared.args.flexgen:
        return input_ids.numpy()
    elif shared.args.deepspeed:
        return input_ids.to(device=local_rank)
    elif torch.has_mps:
        device = torch.device('mps')
        return input_ids.to(device)
    else:
        return input_ids.cuda()


def get_encoded_length(prompt):
    length_after_extensions = apply_extensions('tokenized_length', prompt)
    if length_after_extensions is not None:
        return length_after_extensions

    return len(encode(prompt)[0])


def decode(output_ids, skip_special_tokens=True):
    return shared.tokenizer.decode(output_ids, skip_special_tokens)


# Removes empty replies from gpt4chan outputs
def fix_gpt4chan(s):
    for i in range(10):
        s = re.sub("--- [0-9]*\n>>[0-9]*\n---", "---", s)
        s = re.sub("--- [0-9]*\n *\n---", "---", s)
        s = re.sub("--- [0-9]*\n\n\n---", "---", s)

    return s


# Fix the LaTeX equations in galactica
def fix_galactica(s):
    s = s.replace(r'\[', r'$')
    s = s.replace(r'\]', r'$')
    s = s.replace(r'\(', r'$')
    s = s.replace(r'\)', r'$')
    s = s.replace(r'$$', r'$')
    s = re.sub(r'\n', r'\n\n', s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s


def get_reply_from_output_ids(output_ids, input_ids, original_question, state, is_chat=False):
    if shared.is_seq2seq:
        reply = decode(output_ids, state['skip_special_tokens'])
    else:
        new_tokens = len(output_ids) - len(input_ids[0])
        reply = decode(output_ids[-new_tokens:], state['skip_special_tokens'])
        # Prevent LlamaTokenizer from skipping a space
        if type(shared.tokenizer) in [transformers.LlamaTokenizer, transformers.LlamaTokenizerFast] and len(output_ids) > 0:
            if shared.tokenizer.convert_ids_to_tokens(int(output_ids[-new_tokens])).startswith('▁'):
                reply = ' ' + reply

    if not is_chat:
        reply = apply_extensions('output', reply)

    return reply


def formatted_outputs(reply, model_name):
    if any(s in model_name for s in ['gpt-4chan', 'gpt4chan']):
        reply = fix_gpt4chan(reply)
        return reply, generate_4chan_html(reply)
    else:
        return reply, generate_basic_html(reply)


def set_manual_seed(seed):
    seed = int(seed)
    if seed == -1:
        seed = random.randint(1, 2**31)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return seed


def stop_everything_event():
    shared.stop_everything = True


def generate_reply_wrapper(question, state, eos_token=None, stopping_strings=None):
    for reply in generate_reply(question, state, eos_token, stopping_strings, is_chat=False):
        if not shared.is_seq2seq:
            reply = question + reply

        yield formatted_outputs(reply, shared.model_name)


def _generate_reply(question, state, eos_token=None, stopping_strings=None, is_chat=False):
    state = apply_extensions('state', state)
    generate_func = apply_extensions('custom_generate_reply')
    if generate_func is None:
        if shared.model_name == 'None' or shared.model is None:
            logger.error("No model is loaded! Select one in the Model tab.")
            yield ''
            return

        if shared.model.__class__.__name__ in ['LlamaCppModel', 'RWKVModel', 'ExllamaModel']:
            generate_func = generate_reply_custom
        elif shared.args.flexgen:
            generate_func = generate_reply_flexgen
        else:
            generate_func = generate_reply_HF

    # Preparing the input
    original_question = question
    if not is_chat:
        question = apply_extensions('input', question)

    if shared.args.verbose:
        print(f'\n\n{question}\n--------------------\n')

    shared.stop_everything = False
    clear_torch_cache()
    seed = set_manual_seed(state['seed'])
    is_stream = state['stream']
    last_update = -1
    reply = ''
    for reply in generate_func(question, original_question, seed, state, eos_token, stopping_strings, is_chat=is_chat):
        if is_stream:
            cur_time = time.time()
            if cur_time - last_update > 0.041666666666666664:  # Limit streaming to 24 fps
                last_update = cur_time
                yield reply
        else:
            yield reply

    if is_stream:
        yield reply


def generate_reply_HF(question, original_question, seed, state, eos_token=None, stopping_strings=None, is_chat=False):
    generate_params = {}
    for k in ['max_new_tokens', 'do_sample', 'temperature', 'top_p', 'typical_p', 'repetition_penalty', 'encoder_repetition_penalty', 'top_k', 'min_length', 'no_repeat_ngram_size', 'num_beams', 'penalty_alpha', 'length_penalty', 'early_stopping', 'tfs', 'top_a', 'mirostat_mode', 'mirostat_tau', 'mirostat_eta']:
        generate_params[k] = state[k]

    for k in ['epsilon_cutoff', 'eta_cutoff']:
        if state[k] > 0:
            generate_params[k] = state[k] * 1e-4

    if state['ban_eos_token']:
        generate_params['suppress_tokens'] = [shared.tokenizer.eos_token_id]

    if shared.args.no_cache:
        generate_params.update({'use_cache': False})

    if shared.args.deepspeed:
        generate_params.update({'synced_gpus': True})

    # Encode the input
    input_ids = encode(question, add_bos_token=state['add_bos_token'], truncation_length=get_max_prompt_length(state))
    output = input_ids[0]
    cuda = not any((shared.args.cpu, shared.args.deepspeed))

    # Find the eos tokens
    eos_token_ids = [shared.tokenizer.eos_token_id] if shared.tokenizer.eos_token_id is not None else []
    if eos_token is not None:
        eos_token_ids.append(int(encode(eos_token)[0][-1]))

    # Add the encoded tokens to generate_params
    question, input_ids, inputs_embeds = apply_extensions('tokenizer', state, question, input_ids, None)
    original_input_ids = input_ids
    generate_params.update({'inputs': input_ids})
    if inputs_embeds is not None:
        generate_params.update({'inputs_embeds': inputs_embeds})

    # Create the StoppingCriteriaList with the stopping strings (needs to be done after tokenizer extensions)
    stopping_criteria_list = transformers.StoppingCriteriaList()
    for st in (stopping_strings, ast.literal_eval(f"[{state['custom_stopping_strings']}]")):
        if type(st) is list and len(st) > 0:
            sentinel_token_ids = [encode(string, add_special_tokens=False) for string in st]
            stopping_criteria_list.append(_SentinelTokenStoppingCriteria(sentinel_token_ids=sentinel_token_ids, starting_idx=len(input_ids[0])))
            break

    # Update generate_params with the eos token and the stopping strings
    generate_params['eos_token_id'] = eos_token_ids
    generate_params['stopping_criteria'] = stopping_criteria_list

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

            yield get_reply_from_output_ids(output, input_ids, original_question, state, is_chat=is_chat)

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
                for output in generator:
                    yield get_reply_from_output_ids(output, input_ids, original_question, state, is_chat=is_chat)
                    if output[-1] in eos_token_ids:
                        break

    except Exception:
        traceback.print_exc()
    finally:
        t1 = time.time()
        original_tokens = len(original_input_ids[0])
        new_tokens = len(output) - (original_tokens if not shared.is_seq2seq else 0)
        print(f'Output generated in {(t1-t0):.2f} seconds ({new_tokens/(t1-t0):.2f} tokens/s, {new_tokens} tokens, context {original_tokens}, seed {seed})')
        return


def generate_reply_custom(question, original_question, seed, state, eos_token=None, stopping_strings=None, is_chat=False):
    seed = set_manual_seed(state['seed'])

    t0 = time.time()
    reply = ''
    try:
        if not is_chat:
            yield ''

        if not state['stream']:
            reply = shared.model.generate(question, state)
            if not is_chat:
                reply = apply_extensions('output', reply)

            yield reply
        else:
            for reply in shared.model.generate_with_streaming(question, state):
                if not is_chat:
                    reply = apply_extensions('output', reply)

                yield reply

    except Exception:
        traceback.print_exc()
    finally:
        t1 = time.time()
        original_tokens = len(encode(original_question)[0])
        new_tokens = len(encode(original_question + reply)[0]) - original_tokens
        print(f'Output generated in {(t1-t0):.2f} seconds ({new_tokens/(t1-t0):.2f} tokens/s, {new_tokens} tokens, context {original_tokens}, seed {seed})')
        return


def generate_reply_flexgen(question, original_question, seed, state, eos_token=None, stopping_strings=None, is_chat=False):
    generate_params = {}
    for k in ['max_new_tokens', 'do_sample', 'temperature']:
        generate_params[k] = state[k]

    if state['stream']:
        generate_params['max_new_tokens'] = 8

    # Encode the input
    input_ids = encode(question, add_bos_token=state['add_bos_token'], truncation_length=get_max_prompt_length(state))
    output = input_ids[0]

    # Find the eos tokens
    eos_token_ids = [shared.tokenizer.eos_token_id] if shared.tokenizer.eos_token_id is not None else []
    if eos_token is not None:
        eos_token_ids.append(int(encode(eos_token)[0][-1]))

    # Add the encoded tokens to generate_params
    question, input_ids, inputs_embeds = apply_extensions('tokenizer', state, question, input_ids, None)
    original_input_ids = input_ids
    generate_params.update({'inputs': input_ids})
    if inputs_embeds is not None:
        generate_params.update({'inputs_embeds': inputs_embeds})

    # Update generate_params with the eos token and the stopping strings
    generate_params['stop'] = eos_token_ids[-1]

    t0 = time.time()
    try:
        if not is_chat:
            yield ''

        # Generate the entire reply at once.
        if not state['stream']:
            with torch.no_grad():
                output = shared.model.generate(**generate_params)[0]

            yield get_reply_from_output_ids(output, input_ids, original_question, state, is_chat=is_chat)

        # Stream the output naively for FlexGen since it doesn't support 'stopping_criteria'
        else:
            for i in range(state['max_new_tokens'] // 8 + 1):
                if shared.stop_everything:
                    break

                clear_torch_cache()
                with torch.no_grad():
                    output = shared.model.generate(**generate_params)[0]

                if np.count_nonzero(np.isin(input_ids[0], eos_token_ids)) < np.count_nonzero(np.isin(output, eos_token_ids)):
                    break

                yield get_reply_from_output_ids(output, original_input_ids, original_question, state)
                input_ids = np.reshape(output, (1, output.shape[0]))
                generate_params.update({'inputs': input_ids})

    except Exception:
        traceback.print_exc()
    finally:
        t1 = time.time()
        original_tokens = len(original_input_ids[0])
        new_tokens = len(output) - (original_tokens if not shared.is_seq2seq else 0)
        print(f'Output generated in {(t1-t0):.2f} seconds ({new_tokens/(t1-t0):.2f} tokens/s, {new_tokens} tokens, context {original_tokens}, seed {seed})')
        return
