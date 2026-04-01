import json
import time

from modules import shared


def convert_request(body: dict) -> dict:
    """Transform Anthropic Messages API body into the dict that chat_completions_common expects."""
    messages = []

    # System message
    system = body.get('system')
    if system:
        if isinstance(system, list):
            # List of content blocks like [{"type":"text","text":"..."}]
            text_parts = [block.get('text', '') for block in system if isinstance(block, dict) and block.get('type') == 'text']
            system_text = '\n'.join(text_parts)
        else:
            system_text = str(system)
        if system_text:
            messages.append({"role": "system", "content": system_text})

    # Convert messages
    for msg in body.get('messages', []):
        role = msg.get('role')
        content = msg.get('content')

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            messages.append({"role": role, "content": str(content) if content else ""})
            continue

        if role == 'assistant':
            # Split into text content, tool_calls, and skip thinking blocks
            text_parts = []
            tool_calls = []
            for block in content:
                btype = block.get('type')
                if btype == 'text':
                    text_parts.append(block.get('text', ''))
                elif btype == 'tool_use':
                    tool_calls.append({
                        "id": block.get('id', ''),
                        "type": "function",
                        "function": {
                            "name": block.get('name', ''),
                            "arguments": json.dumps(block.get('input', {}))
                        }
                    })
                elif btype == 'thinking':
                    pass  # Strip thinking blocks

            assistant_msg = {"role": "assistant", "content": '\n'.join(text_parts) if text_parts else ""}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

        elif role == 'user':
            # Handle tool_result blocks and regular content
            regular_parts = []
            for block in content:
                btype = block.get('type')
                if btype == 'tool_result':
                    # Emit any accumulated regular content first
                    if regular_parts:
                        if len(regular_parts) == 1 and regular_parts[0].get('type') == 'text':
                            messages.append({"role": "user", "content": regular_parts[0]['text']})
                        else:
                            messages.append({"role": "user", "content": regular_parts})
                        regular_parts = []
                    # Convert tool_result to OpenAI tool message
                    tool_content = block.get('content', '')
                    if isinstance(tool_content, list):
                        tool_content = '\n'.join(
                            b.get('text', '') for b in tool_content
                            if isinstance(b, dict) and b.get('type') == 'text'
                        )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": block.get('tool_use_id', ''),
                        "content": str(tool_content)
                    })
                elif btype == 'text':
                    regular_parts.append({"type": "text", "text": block.get('text', '')})
                elif btype == 'image':
                    source = block.get('source', {})
                    if source.get('type') == 'base64':
                        media_type = source.get('media_type', 'image/png')
                        data = source.get('data', '')
                        regular_parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{data}"}
                        })
                elif btype == 'thinking':
                    pass  # Strip thinking blocks

            if regular_parts:
                if len(regular_parts) == 1 and regular_parts[0].get('type') == 'text':
                    messages.append({"role": "user", "content": regular_parts[0]['text']})
                else:
                    messages.append({"role": "user", "content": regular_parts})
        else:
            messages.append({"role": role, "content": str(content)})

    # Start with all fields from the original body (includes GenerationOptions defaults)
    result = dict(body)

    # Remove Anthropic-specific fields that don't map directly
    for key in ('system', 'stop_sequences', 'tools', 'tool_choice', 'thinking', 'metadata'):
        result.pop(key, None)

    # Set converted fields
    result['messages'] = messages
    result['max_tokens'] = body.get('max_tokens', 4096)
    result['stream'] = body.get('stream', False)
    result['mode'] = 'instruct'

    # Ensure ChatCompletionRequestParams defaults are present
    result.setdefault('continue_', False)
    result.setdefault('instruction_template', None)
    result.setdefault('instruction_template_str', None)
    result.setdefault('character', None)
    result.setdefault('bot_name', None)
    result.setdefault('context', None)
    result.setdefault('greeting', None)
    result.setdefault('user_name', None)
    result.setdefault('user_bio', None)
    result.setdefault('chat_template_str', None)
    result.setdefault('chat_instruct_command', 'Continue the chat dialogue below. Write a single reply for the character "<|character|>".\n\n<|prompt|>')
    result.setdefault('frequency_penalty', None)
    result.setdefault('presence_penalty', None)
    result.setdefault('logit_bias', None)
    result.setdefault('logprobs', None)
    result.setdefault('top_logprobs', None)
    result.setdefault('n', 1)
    result.setdefault('model', None)
    result.setdefault('functions', None)
    result.setdefault('function_call', None)
    result.setdefault('stream_options', None)
    result.setdefault('user', None)
    result.setdefault('stop', None)
    result.setdefault('tool_choice', None)

    # Always request usage in streaming so the usage-only chunk triggers
    # the deferred message_delta/message_stop with accurate output_tokens
    if body.get('stream', False):
        result['stream_options'] = {'include_usage': True}

    # Map stop_sequences -> stop
    if body.get('stop_sequences'):
        result['stop'] = body['stop_sequences']

    # Tools
    if body.get('tools'):
        result['tools'] = [
            {
                "type": "function",
                "function": {
                    "name": t.get('name', ''),
                    "description": t.get('description', ''),
                    "parameters": t.get('input_schema', {"type": "object", "properties": {}})
                }
            }
            for t in body['tools']
        ]

    # Tool choice
    tc = body.get('tool_choice')
    if tc and isinstance(tc, dict):
        tc_type = tc.get('type')
        if tc_type == 'auto':
            result['tool_choice'] = 'auto'
        elif tc_type == 'any':
            result['tool_choice'] = 'required'
        elif tc_type == 'tool':
            result['tool_choice'] = {"type": "function", "function": {"name": tc.get('name', '')}}
        elif tc_type == 'none':
            result['tool_choice'] = 'none'
    else:
        result.setdefault('tool_choice', None)

    # Thinking
    thinking = body.get('thinking')
    if thinking and isinstance(thinking, dict) and thinking.get('type') in ('enabled', 'adaptive'):
        result['enable_thinking'] = True

    return result


_FINISH_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}


def build_response(openai_resp: dict, model: str) -> dict:
    """Transform OpenAI chat completion response dict into Anthropic Messages format."""
    resp_id = openai_resp.get('id', 'msg_unknown')
    if resp_id.startswith('chatcmpl-'):
        resp_id = 'msg_' + resp_id[9:]

    choice = openai_resp.get('choices', [{}])[0]
    message = choice.get('message', {})

    content = []

    # Reasoning/thinking content
    reasoning = message.get('reasoning_content')
    if reasoning:
        content.append({"type": "thinking", "thinking": reasoning, "signature": ""})

    # Text content
    text = message.get('content')
    if text:
        content.append({"type": "text", "text": text})

    # Tool calls
    tool_calls = message.get('tool_calls')
    if tool_calls:
        for tc in tool_calls:
            func = tc.get('function', {})
            try:
                input_data = json.loads(func.get('arguments', '{}'))
            except (json.JSONDecodeError, TypeError):
                input_data = {}
            content.append({
                "type": "tool_use",
                "id": tc.get('id', ''),
                "name": func.get('name', ''),
                "input": input_data
            })

    finish_reason = choice.get('finish_reason', 'stop')
    stop_reason = _FINISH_REASON_MAP.get(finish_reason, 'end_turn')

    usage = openai_resp.get('usage', {})

    return {
        "id": resp_id,
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get('prompt_tokens', 0),
            "output_tokens": usage.get('completion_tokens', 0),
        }
    }


class StreamConverter:
    """Stateful converter: processes one OpenAI chunk at a time, yields Anthropic SSE events.

    When include_usage is enabled in the OpenAI request, the final chunk with
    finish_reason has usage=None, followed by a separate usage-only chunk
    (choices=[], usage={...}).  We defer emitting message_delta and message_stop
    until we receive that usage chunk so output_tokens is accurate.
    """

    def __init__(self, model: str):
        self.model = model
        self.msg_id = "msg_%d" % int(time.time() * 1000000000)
        self.block_index = 0
        self.in_thinking = False
        self.in_text = False
        self.input_tokens = 0
        self.output_tokens = 0
        self.tool_calls_accum = {}
        self.stop_reason = "end_turn"
        self._pending_finish = False  # True after we've seen finish_reason

    def process_chunk(self, chunk: dict) -> list[dict]:
        """Process a single OpenAI streaming chunk; return list of Anthropic SSE event dicts."""
        events = []
        choices = chunk.get('choices', [])
        usage = chunk.get('usage')

        if usage:
            self.input_tokens = usage.get('prompt_tokens', self.input_tokens)
            self.output_tokens = usage.get('completion_tokens', self.output_tokens)

        # Usage-only chunk (choices=[]) arrives after the finish chunk
        if not choices:
            if self._pending_finish:
                events.extend(self.finish())
            return events

        choice = choices[0]
        delta = choice.get('delta', {})
        finish_reason = choice.get('finish_reason')

        # First chunk with role
        if 'role' in delta:
            events.append({
                "event": "message_start",
                "data": json.dumps({
                    "type": "message_start",
                    "message": {
                        "id": self.msg_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": self.model,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": self.input_tokens, "output_tokens": 0}
                    }
                })
            })
            events.append({"event": "ping", "data": json.dumps({"type": "ping"})})
            return events

        # Reasoning content
        reasoning_content = delta.get('reasoning_content')
        if reasoning_content:
            if not self.in_thinking:
                self.in_thinking = True
                events.append({
                    "event": "content_block_start",
                    "data": json.dumps({
                        "type": "content_block_start",
                        "index": self.block_index,
                        "content_block": {"type": "thinking", "thinking": "", "signature": ""}
                    })
                })
            events.append({
                "event": "content_block_delta",
                "data": json.dumps({
                    "type": "content_block_delta",
                    "index": self.block_index,
                    "delta": {"type": "thinking_delta", "thinking": reasoning_content}
                })
            })
            return events

        # Text content
        text_content = delta.get('content')
        if text_content:
            if self.in_thinking:
                events.append({
                    "event": "content_block_stop",
                    "data": json.dumps({"type": "content_block_stop", "index": self.block_index})
                })
                self.in_thinking = False
                self.block_index += 1

            if not self.in_text:
                self.in_text = True
                events.append({
                    "event": "content_block_start",
                    "data": json.dumps({
                        "type": "content_block_start",
                        "index": self.block_index,
                        "content_block": {"type": "text", "text": ""}
                    })
                })
            events.append({
                "event": "content_block_delta",
                "data": json.dumps({
                    "type": "content_block_delta",
                    "index": self.block_index,
                    "delta": {"type": "text_delta", "text": text_content}
                })
            })
            return events

        # Tool calls in delta
        chunk_tool_calls = delta.get('tool_calls')
        if chunk_tool_calls:
            for tc in chunk_tool_calls:
                tc_id = tc.get('id', '')
                tc_idx = tc.get('index', 0)
                func = tc.get('function', {})
                if tc_id:
                    self.tool_calls_accum[tc_idx] = {
                        "id": tc_id,
                        "name": func.get('name', ''),
                        "arguments": func.get('arguments', '')
                    }
                elif tc_idx in self.tool_calls_accum:
                    self.tool_calls_accum[tc_idx]["arguments"] += func.get('arguments', '')

        # Final chunk — close open content blocks, defer message_delta/stop for usage
        if finish_reason is not None:
            self.stop_reason = _FINISH_REASON_MAP.get(finish_reason, 'end_turn')

            if self.in_thinking:
                events.append({
                    "event": "content_block_stop",
                    "data": json.dumps({"type": "content_block_stop", "index": self.block_index})
                })
                self.in_thinking = False
                self.block_index += 1

            if self.in_text:
                events.append({
                    "event": "content_block_stop",
                    "data": json.dumps({"type": "content_block_stop", "index": self.block_index})
                })
                self.in_text = False
                self.block_index += 1

            for tc_idx in sorted(self.tool_calls_accum.keys()):
                tc = self.tool_calls_accum[tc_idx]
                arguments_str = tc["arguments"] or "{}"

                events.append({
                    "event": "content_block_start",
                    "data": json.dumps({
                        "type": "content_block_start",
                        "index": self.block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["name"],
                            "input": {}
                        }
                    })
                })
                # Emit the full input as a single input_json_delta so SDK
                # clients that reconstruct from deltas get the correct data
                events.append({
                    "event": "content_block_delta",
                    "data": json.dumps({
                        "type": "content_block_delta",
                        "index": self.block_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": arguments_str
                        }
                    })
                })
                events.append({
                    "event": "content_block_stop",
                    "data": json.dumps({"type": "content_block_stop", "index": self.block_index})
                })
                self.block_index += 1

            # Defer message_delta/stop — usage chunk may follow
            self._pending_finish = True

        return events

    def finish(self) -> list[dict]:
        """Emit deferred message_delta and message_stop. Safe to call multiple times."""
        if not self._pending_finish:
            return []
        self._pending_finish = False
        return [
            {
                "event": "message_delta",
                "data": json.dumps({
                    "type": "message_delta",
                    "delta": {"stop_reason": self.stop_reason, "stop_sequence": None},
                    "usage": {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens}
                })
            },
            {
                "event": "message_stop",
                "data": json.dumps({"type": "message_stop"})
            }
        ]
