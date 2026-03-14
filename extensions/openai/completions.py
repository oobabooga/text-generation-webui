import copy
import functools
import json
import time
from collections import deque
from pathlib import Path

import tiktoken
import yaml
from pydantic import ValidationError

from extensions.openai.errors import InvalidRequestError
from extensions.openai.typing import ToolDefinition
from extensions.openai.utils import debug_msg
from modules.tool_parsing import get_tool_call_id, parse_tool_call
from modules import shared
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
    ext = filepath.suffix.lower()
    text = filepath.read_text(encoding='utf-8')
    if ext in ['.yaml', '.yml']:
        data = yaml.safe_load(text)
        return data.get('instruction_template', '')
    return text


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

        chosen = top[0]
        token_str = chosen.get('token', '')
        token_logprob = chosen.get('logprob', chosen.get('prob', 0))

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

    Output: {"tokens", "token_logprobs", "top_logprobs": [{token: prob}], "text_offset"}
    """
    if not entries:
        return None

    tokens = []
    token_logprobs = []
    top_logprobs = []
    text_offset = []
    offset = 0

    for entry in entries:
        top = _parse_entry_top(entry)
        if not top:
            continue

        chosen = top[0]
        token_str = chosen.get('token', '')
        token_logprob = chosen.get('logprob', chosen.get('prob', 0))

        tokens.append(token_str)
        token_logprobs.append(token_logprob)
        text_offset.append(offset)
        offset += len(token_str)

        top_dict = {}
        for item in top:
            t = item.get('token', '')
            lp = item.get('logprob', item.get('prob', 0))
            top_dict[t] = lp
        top_logprobs.append(top_dict)

    if not tokens:
        return None

    return {
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "text_offset": text_offset
    }


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
            if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
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
    if 'tools' in body and body['tools'] is not None and isinstance(body['tools'], list) and len(body['tools']) > 0:
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
    if max_tokens in [None, 0]:
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

    # Filter supported_tools when tool_choice specifies a particular function
    if supported_tools and isinstance(tool_choice, dict):
        specified_func = tool_choice.get("function", {}).get("name")
        if specified_func and specified_func in supported_tools:
            supported_tools = [specified_func]

    for a in generator:
        answer = a['internal'][-1][1]

        if supported_tools is not None:
            tool_call = parse_tool_call(answer[end_last_tool_call:], supported_tools) if len(answer) > 0 else []
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
            token_count = len(encode(prompt)[0])
            total_prompt_token_count += token_count

            original_seed = generate_params.get('seed', -1)
            for _n in range(n_completions):
                # Increment seed for each completion to ensure diversity (matches llama.cpp native behavior)
                if original_seed >= 0:
                    generate_params['seed'] = original_seed + _n

                if logprob_proc:
                    logprob_proc.token_alternatives_history.clear()

                # generate reply #######################################
                debug_msg({'prompt': prompt, 'generate_params': generate_params})
                generator = generate_reply(prompt, generate_params, is_chat=False)
                answer = ''

                for a in generator:
                    answer = a

                completion_token_count = len(encode(answer)[0])
                total_completion_token_count += completion_token_count
                stop_reason = "stop"
                if token_count + completion_token_count >= generate_params['truncation_length'] or completion_token_count >= max_tokens:
                    stop_reason = "length"

                if logprob_proc:
                    all_entries = []
                    for alt in logprob_proc.token_alternatives_history:
                        all_entries.extend(_dict_to_logprob_entries(alt))
                    completion_logprobs = format_completion_logprobs(all_entries)
                elif shared.args.loader in ('llama.cpp', 'ExLlamav3'):
                    raw = getattr(shared.model, 'last_completion_probabilities', None)
                    completion_logprobs = format_completion_logprobs(raw)
                else:
                    completion_logprobs = None

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
        token_count = len(encode(prompt)[0])

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

        chunk = text_streaming_chunk(prefix)
        if include_usage:
            chunk['usage'] = None
        yield chunk

        # generate reply #######################################
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
