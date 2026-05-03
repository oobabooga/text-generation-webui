import copy
import functools
import json
import time
from collections import deque
from pathlib import Path

import tiktoken
import yaml
from pydantic import ValidationError

from .errors import InvalidRequestError
from .typing import ToolDefinition
from .utils import debug_msg
from modules.tool_parsing import get_tool_call_id, parse_tool_call, detect_tool_call_format
from modules import shared, utils
from modules.reasoning import extract_reasoning
from modules.chat import (
    generate_chat_prompt,
    generate_chat_reply,
    load_character_memoized,
    load_instruction_template_memoized
)
from modules.image_utils import convert_openai_messages_to_images
from modules.logging_colors import logger
from modules.presets import load_preset_memoized
from modules.text_generation import decode, encode, generate_reply


@functools.cache
def load_chat_template_file(filepath):
    """Load a chat template from a file path (.jinja, .jinja2, or .yaml/.yml)."""
    filepath = Path(filepath)
    text = filepath.read_text(encoding='utf-8')
    if filepath.suffix.lower() in utils.YAML_EXTENSIONS:
        data = yaml.safe_load(text) or {}
        return data.get('instruction_template', '')
    return text


def _first_token_display_str(token_id, prompt, tokenizer):
    """Return the display string for the first prompt token.

    Returns empty string for BOS or tokens that don't appear at the start
    of the prompt text, so they don't shift text_offset for subsequent tokens.
    """
    token_id = int(token_id)
    bos_id = getattr(tokenizer, 'bos_token_id', None)
    if bos_id is not None and token_id == bos_id:
        return ""

    import torch
    tok = tokenizer.decode(torch.tensor([token_id]))
    if not prompt.startswith(tok):
        return ""

    return tok


def _compute_prompt_logprob_entries(prompt, logprobs_count, input_ids=None):
    """Compute logprob entries for prompt tokens via a forward pass.

    Returns a list of logprob entries in the standard format.
    The first token gets a null entry (no conditioning context).

    Supported for HF-compatible loaders (Transformers, ExLlamav3_HF, etc.)
    via a single forward pass, and for llama.cpp via the server's
    prompt_logprobs parameter. Returns [] for unsupported loaders.
    """
    if input_ids is None:
        input_ids = encode(prompt)  # (1, seq_len) tensor or array

    token_ids = input_ids[0]
    n_tokens = len(token_ids)

    if n_tokens == 0:
        return []

    loader = shared.args.loader
    model = shared.model

    if loader == 'llama.cpp':
        return model.get_prompt_logprob_entries(token_ids, max(logprobs_count, 1), prompt=prompt)

    first_token_str = _first_token_display_str(token_ids[0], prompt, shared.tokenizer)

    if n_tokens <= 1:
        return [{"token": first_token_str, "null_logprob": True}]

    import torch
    from modules.torch_utils import clear_torch_cache

    if hasattr(model, 'get_prompt_logits'):
        logits = model.get_prompt_logits(input_ids)

    elif hasattr(model, 'forward'):
        # HF-compatible loaders (Transformers, etc.). Loaders that need a
        # custom path (e.g. wrappers that only compute last-token logits in
        # __call__) should expose get_prompt_logits() above.
        input_ids_tensor = input_ids if isinstance(input_ids, torch.Tensor) else torch.tensor(input_ids, dtype=torch.long)
        if hasattr(model, 'device'):
            input_ids_tensor = input_ids_tensor.to(model.device)
        with torch.inference_mode():
            outputs = model(input_ids=input_ids_tensor)
            logits = outputs.logits  # keep on device, (1, seq_len, vocab) in model dtype
            del outputs

    else:
        return []

    entries = [{"token": first_token_str, "null_logprob": True}]

    logprobs_count = max(logprobs_count, 1)
    k = min(logprobs_count, logits.shape[-1])
    chunk_size = 2048
    unique_ids = set(int(tid) for tid in token_ids[1:])

    # Process logits in chunks, only move top-K results to CPU
    all_top_log_probs_list = []
    all_top_indices_list = []
    all_actual_lps = []

    for start in range(0, n_tokens - 1, chunk_size):
        end = min(start + chunk_size, n_tokens - 1)
        chunk_logits = logits[0, start:end].float()  # (chunk, vocab) on logits.device
        chunk_lse = torch.logsumexp(chunk_logits, dim=-1)
        chunk_top_values, chunk_top_indices = torch.topk(chunk_logits, k=k, dim=-1)
        chunk_top_log_probs = chunk_top_values - chunk_lse.unsqueeze(-1)

        # Compute logprob for actual next tokens in this chunk
        chunk_top_sets = [set(chunk_top_indices[j].tolist()) for j in range(end - start)]
        for j in range(end - start):
            actual_tid = int(token_ids[start + j + 1])
            if actual_tid not in chunk_top_sets[j]:
                all_actual_lps.append((chunk_logits[j, actual_tid] - chunk_lse[j]).item())
            else:
                all_actual_lps.append(None)  # will use top_log_probs

        all_top_log_probs_list.append(chunk_top_log_probs.cpu())
        all_top_indices_list.append(chunk_top_indices.cpu())
        unique_ids.update(int(tid) for tid in chunk_top_indices.flatten().tolist())
        del chunk_logits, chunk_lse, chunk_top_values

    del logits
    clear_torch_cache()

    all_top_log_probs = torch.cat(all_top_log_probs_list, dim=0)
    all_top_indices = torch.cat(all_top_indices_list, dim=0)

    unique_ids_list = sorted(unique_ids)
    decoded_list = shared.tokenizer.batch_decode([[tid] for tid in unique_ids_list]) if hasattr(shared.tokenizer, 'batch_decode') else [shared.tokenizer.decode(torch.tensor([tid])) for tid in unique_ids_list]
    decoded_strs = dict(zip(unique_ids_list, decoded_list))

    for i in range(1, n_tokens):
        token_id = int(token_ids[i])
        idx = i - 1
        top_log_probs = all_top_log_probs[idx]
        top_ids = all_top_indices[idx].tolist()
        actual_token_str = decoded_strs[token_id]

        if token_id in top_ids:
            actual_lp = top_log_probs[top_ids.index(token_id)].item()
            alternatives = [
                {"token": decoded_strs[top_ids[j]], "token_id": top_ids[j], "logprob": top_log_probs[j].item()}
                for j in range(k) if top_ids[j] != token_id
            ]
        else:
            actual_lp = all_actual_lps[idx]
            alternatives = [
                {"token": decoded_strs[top_ids[j]], "token_id": top_ids[j], "logprob": top_log_probs[j].item()}
                for j in range(k - 1)
            ]

        entry = {"top_logprobs": [{"token": actual_token_str, "token_id": token_id, "logprob": actual_lp}] + alternatives}
        entries.append(entry)

    return entries


def _get_raw_logprob_entries(offset=0):
    """Get raw logprob entries from llama.cpp/ExLlamav3 backend, starting from offset.

    Returns (new_entries, new_offset).
    """
    if not hasattr(shared.model, 'last_completion_probabilities') or not shared.model.last_completion_probabilities:
        return [], offset

    all_entries = shared.model.last_completion_probabilities
    new_entries = all_entries[offset:]
    return new_entries, len(all_entries)


def _dict_to_logprob_entries(token_dict):
    """Convert a flat {token: logprob} dict (from LogprobProcessor) to raw entry format."""
    if not token_dict:
        return []

    return [{"top_logprobs": [{"token": t, "logprob": lp} for t, lp in token_dict.items()]}]


def _parse_entry_top(entry):
    """Extract the top logprobs list from a raw entry, handling both key names."""
    return entry.get('top_logprobs', entry.get('top_probs', []))


def _extract_sampled_token(entry, top):
    """Get the actually sampled token and its logprob from a logprob entry.

    Uses the entry-level token/logprob when available (the actually sampled
    token), falling back to top[0] (highest-probability alternative) which
    may differ with non-greedy sampling.
    """
    if 'token' in entry:
        return entry['token'], entry.get('logprob', entry.get('prob', 0))

    token_str = top[0].get('token', '')
    token_logprob = top[0].get('logprob', top[0].get('prob', 0))
    return token_str, token_logprob


def format_chat_logprobs(entries):
    """Format logprob entries into OpenAI chat completions logprobs format.

    Output: {"content": [{"token", "logprob", "bytes", "top_logprobs": [...]}]}
    """
    if not entries:
        return None

    content = []
    for entry in entries:
        top = _parse_entry_top(entry)
        if not top:
            continue

        token_str, token_logprob = _extract_sampled_token(entry, top)

        top_list = []
        for item in top:
            t = item.get('token', '')
            lp = item.get('logprob', item.get('prob', 0))
            top_list.append({
                "token": t,
                "logprob": lp,
                "bytes": list(t.encode('utf-8')) if t else None
            })

        content.append({
            "token": token_str,
            "logprob": token_logprob,
            "bytes": list(token_str.encode('utf-8')) if token_str else None,
            "top_logprobs": top_list
        })

    return {"content": content, "refusal": None} if content else None


def format_completion_logprobs(entries):
    """Format logprob entries into OpenAI completions logprobs format.

    Output: {"tokens", "token_logprobs", "top_logprobs": [{token: prob}], "top_logprobs_ids": [{token_id: prob}], "text_offset"}
    """
    if not entries:
        return None

    tokens = []
    token_logprobs = []
    top_logprobs = []
    top_logprobs_ids = []
    text_offset = []
    offset = 0

    for entry in entries:
        # Handle null logprob entries (first prompt token with echo)
        if entry.get("null_logprob"):
            token_str = entry.get("token", "")
            tokens.append(token_str)
            token_logprobs.append(None)
            top_logprobs.append(None)
            top_logprobs_ids.append(None)
            text_offset.append(offset)
            offset += len(token_str)
            continue

        top = _parse_entry_top(entry)
        if not top:
            continue

        token_str, token_logprob = _extract_sampled_token(entry, top)

        tokens.append(token_str)
        token_logprobs.append(token_logprob)
        text_offset.append(offset)
        offset += len(token_str)

        top_dict = {}
        top_dict_ids = {}
        for item in top:
            t = item.get('token', '')
            lp = item.get('logprob', item.get('prob', 0))
            top_dict[t] = lp
            tid = item.get('token_id', item.get('id'))
            if tid is not None:
                top_dict_ids[tid] = lp
        top_logprobs.append(top_dict)
        top_logprobs_ids.append(top_dict_ids if top_dict_ids else None)

    if not tokens:
        return None

    result = {
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "text_offset": text_offset
    }
    if any(x is not None for x in top_logprobs_ids):
        result["top_logprobs_ids"] = top_logprobs_ids
    return result


def process_parameters(body, is_legacy=False):
    generate_params = body
    max_tokens_str = 'length' if is_legacy else 'max_tokens'
    generate_params['max_new_tokens'] = body.pop(max_tokens_str)
    if generate_params['truncation_length'] == 0:
        generate_params['truncation_length'] = shared.settings['truncation_length']

    if generate_params['temperature'] == 0:
        generate_params['do_sample'] = False
        generate_params['top_k'] = 1

    if body['preset'] is not None:
        preset = load_preset_memoized(body['preset'])
        generate_params.update(preset)

    generate_params['custom_stopping_strings'] = []
    if 'stop' in body:  # str or array, max len 4 (ignored)
        if isinstance(body['stop'], str):
            generate_params['custom_stopping_strings'] = [body['stop']]
        elif isinstance(body['stop'], list):
            generate_params['custom_stopping_strings'] = body['stop']

    # Resolve logprobs: for chat completions, logprobs is a bool and the count
    # comes from top_logprobs. Normalize to an int for all backends.
    logprobs = body.get('logprobs', None)
    top_logprobs = body.get('top_logprobs', None)
    if logprobs is True:
        logprobs = max(top_logprobs, 1) if top_logprobs is not None else 5
        generate_params['logprobs'] = logprobs

    # For llama.cpp and ExLlamav3 native, logit_bias and logprobs are forwarded natively
    if shared.args.loader not in ('llama.cpp', 'ExLlamav3'):
        from transformers import LogitsProcessorList

        from modules.transformers_loader import (
            LogitsBiasProcessor,
            LogprobProcessor
        )

        logits_processor = []
        logit_bias = body.get('logit_bias', None)
        if logit_bias:  # {str: float, ...}
            logits_processor = [LogitsBiasProcessor(logit_bias)]

        if logprobs is not None and logprobs > 0:
            generate_params['logprob_proc'] = LogprobProcessor(logprobs)
            logits_processor.extend([generate_params['logprob_proc']])

        if logits_processor:  # requires logits_processor support
            generate_params['logits_processor'] = LogitsProcessorList(logits_processor)

    return generate_params


def process_multimodal_content(content):
    """Extract text and add image placeholders from OpenAI multimodal format"""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        image_placeholders = ""
        for item in content:
            if not isinstance(item, dict):
                continue

            item_type = item.get('type', '')
            if item_type == 'text':
                text_parts.append(item.get('text', ''))
            elif item_type == 'image_url':
                image_placeholders += "<__media__>"

        final_text = ' '.join(text_parts)
        if image_placeholders:
            return f"{image_placeholders}\n\n{final_text}"
        else:
            return final_text

    return str(content)


def convert_history(history):
    '''
    Chat histories in this program are in the format [message, reply].
    This function converts OpenAI histories to that format.
    '''
    chat_dialogue = []
    current_message = ""
    current_reply = ""
    user_input = ""
    user_input_last = True
    system_message = ""
    seen_non_system = False

    for entry in history:
        content = entry["content"]
        role = entry["role"]

        if role == "user":
            seen_non_system = True
            # Extract text content (images handled by model-specific code)
            content = process_multimodal_content(content)
            user_input = content
            user_input_last = True

            if current_message:
                chat_dialogue.append([current_message, '', '', {}])
                current_message = ""

            current_message = content
        elif role == "assistant":
            seen_non_system = True
            meta = {}
            tool_calls = entry.get("tool_calls")
            if tool_calls and isinstance(tool_calls, list):
                meta["tool_calls"] = tool_calls
                if content.strip() == "":
                    content = ""  # keep empty content, don't skip

            current_reply = content
            user_input_last = False
            if current_message:
                chat_dialogue.append([current_message, current_reply, '', meta])
                current_message = ""
                current_reply = ""
            else:
                chat_dialogue.append(['', current_reply, '', meta])
        elif role == "tool":
            seen_non_system = True
            user_input_last = False
            meta = {}
            if "tool_call_id" in entry:
                meta["tool_call_id"] = entry["tool_call_id"]
            chat_dialogue.append(['', '', content, meta])
        elif role in ("system", "developer"):
            if not seen_non_system:
                # Leading system messages go to custom_system_message (placed at top)
                system_message += f"\n{content}" if system_message else content
            else:
                # Mid-conversation system messages: preserve position in history
                if current_message:
                    chat_dialogue.append([current_message, '', '', {}])
                    current_message = ""
                chat_dialogue.append([content, '', '', {"role": "system"}])

    if not user_input_last:
        user_input = ""

    return user_input, system_message, {
        'internal': chat_dialogue,
        'visible': copy.deepcopy(chat_dialogue),
        'messages': history  # Store original messages for multimodal models
    }


def chat_completions_common(body: dict, is_legacy: bool = False, stream=False, prompt_only=False, stop_event=None) -> dict:
    if body.get('functions', []):
        raise InvalidRequestError(message="functions is not supported.", param='functions')

    if body.get('function_call', ''):
        raise InvalidRequestError(message="function_call is not supported.", param='function_call')

    if 'messages' not in body:
        raise InvalidRequestError(message="messages is required", param='messages')

    tools = None
    if 'tools' in body and body['tools'] is not None and isinstance(body['tools'], list) and body['tools']:
        tools = validateTools(body['tools'])  # raises InvalidRequestError if validation fails

    tool_choice = body.get('tool_choice', None)
    if tool_choice == "none":
        tools = None  # Disable tool detection entirely

    messages = body['messages']
    for m in messages:
        if 'role' not in m:
            raise InvalidRequestError(message="messages: missing role", param='messages')
        elif m['role'] == 'function':
            raise InvalidRequestError(message="role: function is not supported.", param='messages')

        # Handle multimodal content validation
        content = m.get('content')
        if content is None:
            # OpenAI allows content: null on assistant messages when tool_calls is present
            if m['role'] == 'assistant' and m.get('tool_calls'):
                m['content'] = ''
            else:
                raise InvalidRequestError(message="messages: missing content", param='messages')

        # Validate multimodal content structure
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict) or 'type' not in item:
                    raise InvalidRequestError(message="messages: invalid content item format", param='messages')
                if item['type'] not in ['text', 'image_url']:
                    raise InvalidRequestError(message="messages: unsupported content type", param='messages')
                if item['type'] == 'text' and 'text' not in item:
                    raise InvalidRequestError(message="messages: missing text in content item", param='messages')
                if item['type'] == 'image_url' and ('image_url' not in item or 'url' not in item['image_url']):
                    raise InvalidRequestError(message="messages: missing image_url in content item", param='messages')

    # Chat Completions
    object_type = 'chat.completion' if not stream else 'chat.completion.chunk'
    created_time = int(time.time())
    cmpl_id = "chatcmpl-%d" % (int(time.time() * 1000000000))
    resp_list = 'data' if is_legacy else 'choices'

    # generation parameters
    generate_params = process_parameters(body, is_legacy=is_legacy)
    if stop_event is not None:
        generate_params['stop_event'] = stop_event
    continue_ = body['continue_']

    # Instruction template
    if body['instruction_template_str']:
        instruction_template_str = body['instruction_template_str']
    elif body['instruction_template']:
        instruction_template = body['instruction_template']
        instruction_template = "Alpaca" if instruction_template == "None" else instruction_template
        instruction_template_str = load_instruction_template_memoized(instruction_template)
    elif shared.args.chat_template_file:
        instruction_template_str = load_chat_template_file(shared.args.chat_template_file)
    else:
        instruction_template_str = shared.settings['instruction_template_str']

    chat_template_str = body['chat_template_str'] or shared.default_settings['chat_template_str']
    chat_instruct_command = body['chat_instruct_command'] or shared.default_settings['chat-instruct_command']

    # Chat character
    character = body['character'] or shared.default_settings['character']
    character = "Assistant" if character == "None" else character
    name1 = body['user_name'] or shared.default_settings['name1']
    name1, name2, _, greeting, context = load_character_memoized(character, name1, '')
    name2 = body['bot_name'] or name2
    context = body['context'] or context
    greeting = body['greeting'] or greeting
    user_bio = body['user_bio'] or ''

    # History
    user_input, custom_system_message, history = convert_history(messages)

    generate_params.update({
        'mode': body['mode'],
        'name1': name1,
        'name2': name2,
        'context': context,
        'greeting': greeting,
        'user_bio': user_bio,
        'instruction_template_str': instruction_template_str,
        'custom_system_message': custom_system_message,
        'chat_template_str': chat_template_str,
        'chat-instruct_command': chat_instruct_command,
        'tools': tools,
        'history': history,
        'stream': stream
    })

    max_tokens = generate_params['max_new_tokens']
    if max_tokens is not None and max_tokens <= 0:
        raise InvalidRequestError(message="max_tokens must be greater than 0.", param="max_tokens")

    if max_tokens is None:
        generate_params['max_new_tokens'] = 512
        generate_params['auto_max_new_tokens'] = True

    requested_model = generate_params.pop('model')
    logprob_proc = generate_params.pop('logprob_proc', None)
    if logprob_proc:
        logprob_proc.token_alternatives_history.clear()
    chat_logprobs_offset = [0]  # mutable for closure access in streaming

    def chat_streaming_chunk(content=None, chunk_tool_calls=None, include_role=False, reasoning_content=None):
        # begin streaming
        delta = {}
        if include_role:
            delta['role'] = 'assistant'
            delta['refusal'] = None
        if content is not None:
            delta['content'] = content
        if reasoning_content is not None:
            delta['reasoning_content'] = reasoning_content
        if chunk_tool_calls:
            delta['tool_calls'] = chunk_tool_calls

        chunk = {
            "id": cmpl_id,
            "object": object_type,
            "created": created_time,
            "model": shared.model_name,
            "system_fingerprint": None,
            resp_list: [{
                "index": 0,
                "finish_reason": None,
                "delta": delta,
                "logprobs": None,
            }],
        }

        if logprob_proc:
            entries = _dict_to_logprob_entries(logprob_proc.token_alternatives)
            formatted = format_chat_logprobs(entries)
            if formatted:
                chunk[resp_list][0]["logprobs"] = formatted
        elif shared.args.loader in ('llama.cpp', 'ExLlamav3'):
            entries, chat_logprobs_offset[0] = _get_raw_logprob_entries(chat_logprobs_offset[0])
            if entries:
                formatted = format_chat_logprobs(entries)
                if formatted:
                    chunk[resp_list][0]["logprobs"] = formatted

        return chunk

    # Check if usage should be included in streaming chunks per OpenAI spec
    stream_options = body.get('stream_options')
    include_usage = bool(stream_options) and bool(stream_options.get('include_usage') if isinstance(stream_options, dict) else getattr(stream_options, 'include_usage', False))

    # generate reply #######################################
    if prompt_only:
        prompt = generate_chat_prompt(user_input, generate_params, _continue=continue_)
        yield {'prompt': prompt}
        return

    if stream:
        chunk = chat_streaming_chunk('', include_role=True)
        if include_usage:
            chunk['usage'] = None
        yield chunk

    generator = generate_chat_reply(
        user_input, generate_params, regenerate=False, _continue=continue_, loading_message=False)

    answer = ''
    seen_content = ''
    seen_reasoning = ''

    tool_calls = []
    end_last_tool_call = 0
    supported_tools = [x["function"]["name"] for x in tools] if tools is not None else None
    _tool_parsers = None

    # Filter supported_tools when tool_choice specifies a particular function
    if supported_tools and isinstance(tool_choice, dict):
        specified_func = tool_choice.get("function", {}).get("name")
        if specified_func and specified_func in supported_tools:
            supported_tools = [specified_func]

    if supported_tools is not None:
        _template_str = generate_params.get('instruction_template_str', '') if generate_params.get('mode') == 'instruct' else generate_params.get('chat_template_str', '')
        _tool_parsers, _, _ = detect_tool_call_format(_template_str)

    for a in generator:
        answer = a['internal'][-1][1]

        if supported_tools is not None:
            tool_call = parse_tool_call(answer[end_last_tool_call:], supported_tools, parsers=_tool_parsers) if len(answer) > 0 else []
            if len(tool_call) > 0:
                for tc in tool_call:
                    tc["id"] = get_tool_call_id()
                    if stream:
                        tc["index"] = len(tool_calls)
                    tc["function"]["arguments"] = json.dumps(tc["function"]["arguments"])
                    tool_calls.append(tc)
                end_last_tool_call = len(answer)

        # Stop generation before streaming content if tool_calls were detected,
        # so that raw tool markup is not sent as content deltas.
        if len(tool_calls) > 0:
            break

        if stream:
            # Strip reasoning/thinking blocks so only final content is streamed.
            # Reasoning is emitted separately as reasoning_content deltas.
            reasoning, content = extract_reasoning(answer)
            if reasoning is not None:
                new_reasoning = reasoning[len(seen_reasoning):]
                new_content = content[len(seen_content):]
            else:
                new_reasoning = None
                new_content = answer[len(seen_content):]

            if (not new_content and not new_reasoning) or chr(0xfffd) in (new_content or '') + (new_reasoning or ''):
                continue

            chunk = chat_streaming_chunk(
                content=new_content if new_content else None,
                reasoning_content=new_reasoning if new_reasoning else None,
            )
            if include_usage:
                chunk['usage'] = None

            if reasoning is not None:
                seen_reasoning = reasoning
                seen_content = content
            else:
                seen_content = answer
            yield chunk

    token_count = shared.model.last_prompt_token_count if hasattr(shared.model, 'last_prompt_token_count') else 0
    completion_token_count = len(encode(answer)[0])
    if len(tool_calls) > 0:
        stop_reason = "tool_calls"
    elif token_count + completion_token_count >= generate_params['truncation_length'] or completion_token_count >= generate_params['max_new_tokens']:
        stop_reason = "length"
    else:
        stop_reason = "stop"

    if stream:
        chunk = chat_streaming_chunk(chunk_tool_calls=tool_calls)
        chunk[resp_list][0]['finish_reason'] = stop_reason
        usage = {
            "prompt_tokens": token_count,
            "completion_tokens": completion_token_count,
            "total_tokens": token_count + completion_token_count
        }

        if include_usage:
            chunk['usage'] = None
            yield chunk
            # Separate usage-only chunk with choices: [] per OpenAI spec
            yield {
                "id": cmpl_id,
                "object": object_type,
                "created": created_time,
                "model": shared.model_name,
                "system_fingerprint": None,
                resp_list: [],
                "usage": usage
            }
        else:
            yield chunk
    else:
        reasoning, content = extract_reasoning(answer)
        message = {
            "role": "assistant",
            "refusal": None,
            "content": None if tool_calls else content,
            **({"reasoning_content": reasoning} if reasoning else {}),
            **({"tool_calls": tool_calls} if tool_calls else {}),
        }
        resp = {
            "id": cmpl_id,
            "object": object_type,
            "created": created_time,
            "model": shared.model_name,
            "system_fingerprint": None,
            resp_list: [{
                "index": 0,
                "finish_reason": stop_reason,
                "message": message,
                "logprobs": None,
            }],
            "usage": {
                "prompt_tokens": token_count,
                "completion_tokens": completion_token_count,
                "total_tokens": token_count + completion_token_count
            }
        }
        if logprob_proc:
            all_entries = []
            for alt in logprob_proc.token_alternatives_history:
                all_entries.extend(_dict_to_logprob_entries(alt))
            formatted = format_chat_logprobs(all_entries)
            if formatted:
                resp[resp_list][0]["logprobs"] = formatted
        elif shared.args.loader in ('llama.cpp', 'ExLlamav3'):
            raw = getattr(shared.model, 'last_completion_probabilities', None)
            if raw:
                formatted = format_chat_logprobs(raw)
                if formatted:
                    resp[resp_list][0]["logprobs"] = formatted

        yield resp


def completions_common(body: dict, is_legacy: bool = False, stream=False, stop_event=None):
    object_type = 'text_completion'
    created_time = int(time.time())
    cmpl_id = "cmpl-%d" % (int(time.time() * 1000000000))
    resp_list = 'data' if is_legacy else 'choices'

    prompt_str = 'context' if is_legacy else 'prompt'

    # Handle both prompt and messages format for unified multimodal support
    if prompt_str not in body or body[prompt_str] is None:
        if 'messages' in body:
            # Convert messages format to prompt for completions endpoint
            prompt_text = ""
            for message in body.get('messages', []):
                if isinstance(message, dict) and 'content' in message:
                    # Extract text content from multimodal messages
                    content = message['content']
                    if isinstance(content, str):
                        prompt_text += content
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get('type') == 'text':
                                prompt_text += item.get('text', '')

            # Allow empty prompts for image-only requests
            body[prompt_str] = prompt_text
        else:
            raise InvalidRequestError("Missing required input", param=prompt_str)

    # common params
    generate_params = process_parameters(body, is_legacy=is_legacy)
    max_tokens = generate_params['max_new_tokens']
    if max_tokens is None:
        generate_params['max_new_tokens'] = 512
        generate_params['auto_max_new_tokens'] = True
        max_tokens = 512
    elif max_tokens < 0:
        raise InvalidRequestError(message="max_tokens must be greater than or equal to 0.", param="max_tokens")
    elif max_tokens == 0 and body.get('logprobs') is None:
        raise InvalidRequestError(message="max_tokens is 0 but no logprobs parameter was specified.", param="max_tokens")

    generate_params['stream'] = stream
    if stop_event is not None:
        generate_params['stop_event'] = stop_event
    requested_model = generate_params.pop('model')
    logprob_proc = generate_params.pop('logprob_proc', None)
    if logprob_proc:
        logprob_proc.token_alternatives_history.clear()
    suffix = body['suffix'] if body['suffix'] else ''
    echo = body['echo']

    # Add messages to generate_params if present for multimodal processing
    if body.get('messages'):
        generate_params['messages'] = body['messages']
        raw_images = convert_openai_messages_to_images(generate_params['messages'])
        if raw_images:
            logger.info(f"Found {len(raw_images)} image(s) in request.")
            generate_params['raw_images'] = raw_images

    n_completions = body.get('n', 1) or 1

    if not stream:
        prompt_arg = body[prompt_str]

        # Handle empty/None prompts (e.g., image-only requests)
        if prompt_arg is None:
            prompt_arg = ""

        if isinstance(prompt_arg, str) or (isinstance(prompt_arg, list) and len(prompt_arg) > 0 and isinstance(prompt_arg[0], int)):
            prompt_arg = [prompt_arg]

        resp_list_data = []
        total_completion_token_count = 0
        total_prompt_token_count = 0
        choice_index = 0

        for idx, prompt in enumerate(prompt_arg, start=0):
            if isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], int):
                # token lists
                if requested_model == shared.model_name:
                    prompt = decode(prompt)[0]
                else:
                    try:
                        encoder = tiktoken.encoding_for_model(requested_model)
                        prompt = encoder.decode(prompt)
                    except KeyError:
                        prompt = decode(prompt)[0]

            prefix = prompt if echo else ''
            prompt_input_ids = encode(prompt)
            token_count = len(prompt_input_ids[0])
            total_prompt_token_count += token_count

            # Compute prompt logprobs once per prompt (shared across n_completions)
            logprobs_val = body.get('logprobs', None)
            if echo and logprobs_val is not None and logprobs_val >= 0:
                prompt_entries = _compute_prompt_logprob_entries(prompt, logprobs_val, input_ids=prompt_input_ids)
            else:
                prompt_entries = None

            original_seed = generate_params.get('seed', -1)
            for _n in range(n_completions):
                # Increment seed for each completion to ensure diversity (matches llama.cpp native behavior)
                if original_seed >= 0:
                    generate_params['seed'] = original_seed + _n

                if logprob_proc:
                    logprob_proc.token_alternatives_history.clear()

                # generate reply #######################################
                if max_tokens == 0:
                    answer = ''
                    completion_token_count = 0
                    stop_reason = "stop"
                else:
                    debug_msg({'prompt': prompt, 'generate_params': generate_params})
                    generator = generate_reply(prompt, generate_params, is_chat=False)
                    answer = ''

                    for a in generator:
                        answer = a

                    completion_token_count = len(encode(answer)[0])
                    stop_reason = "stop"
                    if token_count + completion_token_count >= generate_params['truncation_length'] or completion_token_count >= max_tokens:
                        stop_reason = "length"

                total_completion_token_count += completion_token_count

                if max_tokens == 0:
                    all_entries = []
                else:
                    if logprob_proc:
                        all_entries = []
                        for alt in logprob_proc.token_alternatives_history:
                            all_entries.extend(_dict_to_logprob_entries(alt))
                    elif shared.args.loader in ('llama.cpp', 'ExLlamav3'):
                        all_entries = getattr(shared.model, 'last_completion_probabilities', None) or []
                    else:
                        all_entries = []

                if prompt_entries:
                    all_entries = prompt_entries + all_entries

                completion_logprobs = format_completion_logprobs(all_entries) if all_entries else None

                respi = {
                    "index": choice_index,
                    "finish_reason": stop_reason,
                    "text": prefix + answer + suffix,
                    "logprobs": completion_logprobs,
                }

                resp_list_data.append(respi)
                choice_index += 1

        resp = {
            "id": cmpl_id,
            "object": object_type,
            "created": created_time,
            "model": shared.model_name,
            "system_fingerprint": None,
            resp_list: resp_list_data,
            "usage": {
                "prompt_tokens": total_prompt_token_count,
                "completion_tokens": total_completion_token_count,
                "total_tokens": total_prompt_token_count + total_completion_token_count
            }
        }

        yield resp
    else:
        prompt = body[prompt_str]
        if isinstance(prompt, list):
            if prompt and isinstance(prompt[0], int):
                try:
                    encoder = tiktoken.encoding_for_model(requested_model)
                    prompt = encoder.decode(prompt)
                except KeyError:
                    prompt = decode(prompt)[0]
            else:
                raise InvalidRequestError(message="API Batched generation not yet supported.", param=prompt_str)

        prefix = prompt if echo else ''
        prompt_input_ids = encode(prompt)
        token_count = len(prompt_input_ids[0])

        # Check if usage should be included in streaming chunks per OpenAI spec
        stream_options = body.get('stream_options')
        include_usage = bool(stream_options) and bool(stream_options.get('include_usage') if isinstance(stream_options, dict) else getattr(stream_options, 'include_usage', False))
        cmpl_logprobs_offset = [0]  # mutable for closure access in streaming

        def text_streaming_chunk(content):
            # begin streaming
            if logprob_proc:
                chunk_logprobs = format_completion_logprobs(_dict_to_logprob_entries(logprob_proc.token_alternatives))
            elif shared.args.loader in ('llama.cpp', 'ExLlamav3'):
                entries, cmpl_logprobs_offset[0] = _get_raw_logprob_entries(cmpl_logprobs_offset[0])
                chunk_logprobs = format_completion_logprobs(entries) if entries else None
            else:
                chunk_logprobs = None

            chunk = {
                "id": cmpl_id,
                "object": object_type,
                "created": created_time,
                "model": shared.model_name,
                "system_fingerprint": None,
                resp_list: [{
                    "index": 0,
                    "finish_reason": None,
                    "text": content,
                    "logprobs": chunk_logprobs,
                }],
            }

            return chunk

        logprobs_val = body.get('logprobs', None)
        if echo and logprobs_val is not None and logprobs_val >= 0:
            prompt_entries = _compute_prompt_logprob_entries(prompt, logprobs_val, input_ids=prompt_input_ids)
            prompt_logprobs_formatted = format_completion_logprobs(prompt_entries) if prompt_entries else None
        else:
            prompt_logprobs_formatted = None

        # Clear stale logprobs from any previous request before building the
        # first chunk, so text_streaming_chunk doesn't pick up old data.
        if hasattr(shared.model, 'last_completion_probabilities'):
            shared.model.last_completion_probabilities = []
        cmpl_logprobs_offset[0] = 0

        chunk = text_streaming_chunk(prefix)
        if prompt_logprobs_formatted is not None:
            chunk[resp_list][0]["logprobs"] = prompt_logprobs_formatted
        if include_usage:
            chunk['usage'] = None
        yield chunk

        # generate reply #######################################
        if max_tokens == 0:
            answer = ''
            completion_token_count = 0
            stop_reason = "stop"
        else:
            debug_msg({'prompt': prompt, 'generate_params': generate_params})
            generator = generate_reply(prompt, generate_params, is_chat=False)
            answer = ''
            seen_content = ''
            completion_token_count = 0

            for a in generator:
                answer = a

                len_seen = len(seen_content)
                new_content = answer[len_seen:]

                if not new_content or chr(0xfffd) in new_content:  # partial unicode character, don't send it yet.
                    continue

                seen_content = answer
                chunk = text_streaming_chunk(new_content)
                if include_usage:
                    chunk['usage'] = None
                yield chunk

            completion_token_count = len(encode(answer)[0])
            stop_reason = "stop"
            if token_count + completion_token_count >= generate_params['truncation_length'] or completion_token_count >= max_tokens:
                stop_reason = "length"

        chunk = text_streaming_chunk(suffix)
        chunk[resp_list][0]["finish_reason"] = stop_reason
        usage = {
            "prompt_tokens": token_count,
            "completion_tokens": completion_token_count,
            "total_tokens": token_count + completion_token_count
        }

        if include_usage:
            chunk['usage'] = None
            yield chunk
            # Separate usage-only chunk with choices: [] per OpenAI spec
            yield {
                "id": cmpl_id,
                "object": object_type,
                "created": created_time,
                "model": shared.model_name,
                "system_fingerprint": None,
                resp_list: [],
                "usage": usage
            }
        else:
            yield chunk


def chat_completions(body: dict, is_legacy: bool = False, stop_event=None) -> dict:
    generator = chat_completions_common(body, is_legacy, stream=False, stop_event=stop_event)
    return deque(generator, maxlen=1).pop()


def stream_chat_completions(body: dict, is_legacy: bool = False, stop_event=None):
    for resp in chat_completions_common(body, is_legacy, stream=True, stop_event=stop_event):
        yield resp


def completions(body: dict, is_legacy: bool = False, stop_event=None) -> dict:
    generator = completions_common(body, is_legacy, stream=False, stop_event=stop_event)
    return deque(generator, maxlen=1).pop()


def stream_completions(body: dict, is_legacy: bool = False, stop_event=None):
    for resp in completions_common(body, is_legacy, stream=True, stop_event=stop_event):
        yield resp


def validateTools(tools: list[dict]):
    # Validate each tool definition in the JSON array
    valid_tools = None
    for idx in range(len(tools)):
        tool = tools[idx]
        try:
            tool_definition = ToolDefinition(**tool)
            # Backfill defaults so Jinja2 templates don't crash on missing fields
            func = tool.get("function", {})
            if "description" not in func:
                func["description"] = ""
            if "parameters" not in func:
                func["parameters"] = {"type": "object", "properties": {}}
            if valid_tools is None:
                valid_tools = []
            valid_tools.append(tool)
        except ValidationError:
            raise InvalidRequestError(message=f"Invalid tool specification at index {idx}.", param='tools')

    return valid_tools
