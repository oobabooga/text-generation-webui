import gc
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
from modules.models import local_rank


def get_max_prompt_length(tokens):
    max_length = 2048-tokens
    if shared.soft_prompt:
        max_length -= shared.soft_prompt_tensor.shape[1]
    return max_length

def encode(prompt, tokens_to_generate=0, add_special_tokens=True):
    if any((shared.is_RWKV, shared.is_llamacpp)):
        input_ids = shared.tokenizer.encode(str(prompt))
        input_ids = np.array(input_ids).reshape(1, len(input_ids))
        return input_ids
    else:
        input_ids = shared.tokenizer.encode(str(prompt), return_tensors='pt', truncation=True, max_length=get_max_prompt_length(tokens_to_generate), add_special_tokens=add_special_tokens)
        if shared.args.cpu:
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

def decode(output_ids):
    # Open Assistant relies on special tokens like <|endoftext|>
    if re.match('.*(oasst|galactica)-*', shared.model_name.lower()):
        return shared.tokenizer.decode(output_ids, skip_special_tokens=False)
    else:
        reply = shared.tokenizer.decode(output_ids, skip_special_tokens=True)
        reply = reply.replace(r'<|endoftext|>', '')
        return reply

def generate_softprompt_input_tensors(input_ids):
    inputs_embeds = shared.model.transformer.wte(input_ids)
    inputs_embeds = torch.cat((shared.soft_prompt_tensor, inputs_embeds), dim=1)
    filler_input_ids = torch.zeros((1, inputs_embeds.shape[1]), dtype=input_ids.dtype).to(shared.model.device)
    #filler_input_ids += shared.model.config.bos_token_id # setting dummy input_ids to bos tokens
    return inputs_embeds, filler_input_ids

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

def formatted_outputs(reply, model_name):
    if not shared.is_chat():
        if 'galactica' in model_name.lower():
            reply = fix_galactica(reply)
            return reply, reply, generate_basic_html(reply)
        elif any((k in shared.model_name.lower() for k in ['gpt4chan', 'gpt-4chan'])):
            reply = fix_gpt4chan(reply)
            return reply, 'Only applicable for GALACTICA models.', generate_4chan_html(reply)
        else:
            return reply, 'Only applicable for GALACTICA models.', generate_basic_html(reply)
    else:
        return reply

def clear_torch_cache():
    gc.collect()
    if not shared.args.cpu:
        torch.cuda.empty_cache()

def set_manual_seed(seed):
    if seed != -1:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

def stop_everything_event():
    shared.stop_everything = True

def generate_reply(question, max_new_tokens, do_sample, temperature, top_p, typical_p, repetition_penalty, encoder_repetition_penalty, top_k, min_length, no_repeat_ngram_size, num_beams, penalty_alpha, length_penalty, early_stopping, seed, eos_token=None, stopping_strings=[]):
    clear_torch_cache()
    set_manual_seed(seed)
    shared.stop_everything = False
    t0 = time.time()

    original_question = question
    if not shared.is_chat():
        question = apply_extensions(question, "input")
    if shared.args.verbose:
        print(f"\n\n{question}\n--------------------\n")

    # These models are not part of Hugging Face, so we handle them
    # separately and terminate the function call earlier
    if any((shared.is_RWKV, shared.is_llamacpp)):
        try:
            if shared.args.no_stream:
                reply = shared.model.generate(context=question, token_count=max_new_tokens, temperature=temperature, top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty)
                output = original_question+reply
                if not shared.is_chat():
                    reply = original_question + apply_extensions(reply, "output")
                yield formatted_outputs(reply, shared.model_name)
            else:
                if not shared.is_chat():
                    yield formatted_outputs(question, shared.model_name)

                # RWKV has proper streaming, which is very nice.
                # No need to generate 8 tokens at a time.
                for reply in shared.model.generate_with_streaming(context=question, token_count=max_new_tokens, temperature=temperature, top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty):
                    output = original_question+reply
                    if not shared.is_chat():
                        reply = original_question + apply_extensions(reply, "output")
                    yield formatted_outputs(reply, shared.model_name)

        except Exception:
            traceback.print_exc()
        finally:
            t1 = time.time()
            original_tokens = len(encode(original_question)[0])
            new_tokens = len(encode(output)[0]) - original_tokens
            print(f"Output generated in {(t1-t0):.2f} seconds ({new_tokens/(t1-t0):.2f} tokens/s, {new_tokens} tokens, context {original_tokens})")
            return

    input_ids = encode(question, max_new_tokens)
    original_input_ids = input_ids
    output = input_ids[0]

    cuda = not any((shared.args.cpu, shared.args.deepspeed, shared.args.flexgen))
    eos_token_ids = [shared.tokenizer.eos_token_id] if shared.tokenizer.eos_token_id is not None else []
    if eos_token is not None:
        eos_token_ids.append(int(encode(eos_token)[0][-1]))
    stopping_criteria_list = transformers.StoppingCriteriaList()
    if type(stopping_strings) is list and len(stopping_strings) > 0:
        t = [encode(string, 0, add_special_tokens=False) for string in stopping_strings]
        stopping_criteria_list.append(_SentinelTokenStoppingCriteria(sentinel_token_ids=t, starting_idx=len(input_ids[0])))

    generate_params = {}
    if not shared.args.flexgen:
        generate_params.update({
            "max_new_tokens": max_new_tokens,
            "eos_token_id": eos_token_ids,
            "stopping_criteria": stopping_criteria_list,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "typical_p": typical_p,
            "repetition_penalty": repetition_penalty,
            "encoder_repetition_penalty": encoder_repetition_penalty,
            "top_k": top_k,
            "min_length": min_length if shared.args.no_stream else 0,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "num_beams": num_beams,
            "penalty_alpha": penalty_alpha,
            "length_penalty": length_penalty,
            "early_stopping": early_stopping,
        })
    else:
        generate_params.update({
            "max_new_tokens": max_new_tokens if shared.args.no_stream else 8,
            "do_sample": do_sample,
            "temperature": temperature,
            "stop": eos_token_ids[-1],
        })
    if shared.args.no_cache:
        generate_params.update({"use_cache": False})
    if shared.args.deepspeed:
        generate_params.update({"synced_gpus": True})
    if shared.soft_prompt:
        inputs_embeds, filler_input_ids = generate_softprompt_input_tensors(input_ids)
        generate_params.update({"inputs_embeds": inputs_embeds})
        generate_params.update({"inputs": filler_input_ids})
    else:
        generate_params.update({"inputs": input_ids})

    try:
        # Generate the entire reply at once.
        if shared.args.no_stream:
            with torch.no_grad():
                output = shared.model.generate(**generate_params)[0]
                if cuda:
                    output = output.cuda()
            if shared.soft_prompt:
                output = torch.cat((input_ids[0], output[filler_input_ids.shape[1]:]))

            new_tokens = len(output) - len(input_ids[0])
            reply = decode(output[-new_tokens:])
            if not shared.is_chat():
                reply = original_question + apply_extensions(reply, "output")

            yield formatted_outputs(reply, shared.model_name)

        # Stream the reply 1 token at a time.
        # This is based on the trick of using 'stopping_criteria' to create an iterator.
        elif not shared.args.flexgen:

            def generate_with_callback(callback=None, **kwargs):
                kwargs['stopping_criteria'].append(Stream(callback_func=callback))
                clear_torch_cache()
                with torch.no_grad():
                    shared.model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(generate_with_callback, kwargs, callback=None)

            if not shared.is_chat():
                yield formatted_outputs(original_question, shared.model_name)
            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    if shared.soft_prompt:
                        output = torch.cat((input_ids[0], output[filler_input_ids.shape[1]:]))

                    new_tokens = len(output) - len(input_ids[0])
                    reply = decode(output[-new_tokens:])
                    if not shared.is_chat():
                        reply = original_question + apply_extensions(reply, "output")

                    if output[-1] in eos_token_ids:
                        break
                    yield formatted_outputs(reply, shared.model_name)

        # Stream the output naively for FlexGen since it doesn't support 'stopping_criteria'
        else:
            for i in range(max_new_tokens//8+1):
                clear_torch_cache()
                with torch.no_grad():
                    output = shared.model.generate(**generate_params)[0]
                if shared.soft_prompt:
                    output = torch.cat((input_ids[0], output[filler_input_ids.shape[1]:]))

                new_tokens = len(output) - len(original_input_ids[0])
                reply = decode(output[-new_tokens:])
                if not shared.is_chat():
                    reply = original_question + apply_extensions(reply, "output")

                if np.count_nonzero(np.isin(input_ids[0], eos_token_ids)) < np.count_nonzero(np.isin(output, eos_token_ids)):
                    break
                yield formatted_outputs(reply, shared.model_name)

                input_ids = np.reshape(output, (1, output.shape[0]))
                if shared.soft_prompt:
                    inputs_embeds, filler_input_ids = generate_softprompt_input_tensors(input_ids)
                    generate_params.update({"inputs_embeds": inputs_embeds})
                    generate_params.update({"inputs": filler_input_ids})
                else:
                    generate_params.update({"inputs": input_ids})

            yield formatted_outputs(reply, shared.model_name)

    except Exception:
        traceback.print_exc()
    finally:
        t1 = time.time()
        original_tokens = len(original_input_ids[0])
        new_tokens = len(output)-original_tokens
        print(f"Output generated in {(t1-t0):.2f} seconds ({new_tokens/(t1-t0):.2f} tokens/s, {new_tokens} tokens, context {original_tokens})")
        return
