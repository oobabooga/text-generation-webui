import json
import random
import re


def get_tool_call_id() -> str:
    letter_bytes = "abcdefghijklmnopqrstuvwxyz0123456789"
    b = [random.choice(letter_bytes) for _ in range(8)]
    return "call_" + "".join(b).lower()


# All known opening markers for tool calls across model formats.
TOOL_CALL_OPENING_MARKERS = [
    '<tool_call>',
    '<function_call>',
    '<minimax:tool_call>',
    '<|tool_call_begin|>',
    '<|tool_calls_section_begin|>',
    '<｜tool▁call▁begin｜>',
    '<｜tool▁calls▁begin｜>',
    '[TOOL_CALLS]',
    'to=functions.',
    '<|channel|>commentary',
]


def streaming_tool_buffer_check(text, markers=None, tool_names=None, check_bare_names=False):
    '''
    Check whether streaming output should be withheld because it may
    contain tool-call markup.

    Args:
        text: Full accumulated internal text.
        markers: Template-specific markers for partial-prefix matching.
                 If None, falls back to TOOL_CALL_OPENING_MARKERS.
        tool_names: List of tool function names.
        check_bare_names: Whether to do partial-prefix matching on tool
                          names (for models with unknown template format).
    '''
    # Full marker found in text → buffer permanently.
    # Always checks ALL known markers regardless of template (cheap safety net).
    for marker in TOOL_CALL_OPENING_MARKERS:
        if marker in text:
            return True

    # Bare function-name full match: "get_weather{...}" or "get_weather {...}"
    if tool_names:
        for name in tool_names:
            if name + '{' in text or name + ' {' in text:
                return True

    # Partial-prefix matching: only for template-specific markers.
    for marker in (markers if markers is not None else TOOL_CALL_OPENING_MARKERS):
        for prefix_len in range(min(len(marker) - 1, len(text)), 0, -1):
            if text.endswith(marker[:prefix_len]):
                return True

    # Bare-name partial matching: only when template format is unknown.
    if check_bare_names and tool_names:
        for name in tool_names:
            if text.endswith(name):
                return True
            for prefix_len in range(min(len(name) - 1, len(text)), 0, -1):
                if text.endswith(name[:prefix_len]):
                    return True

    return False


def check_and_sanitize_tool_call_candidate(candidate_dict: dict, tool_names: list[str]):
    # check if property 'function' exists and is a dictionary, otherwise adapt dict
    if 'function' not in candidate_dict and 'name' in candidate_dict and isinstance(candidate_dict['name'], str):
        candidate_dict = {"type": "function", "function": candidate_dict}
    if 'function' in candidate_dict and isinstance(candidate_dict['function'], str):
        candidate_dict['name'] = candidate_dict['function']
        del candidate_dict['function']
        candidate_dict = {"type": "function", "function": candidate_dict}
    if 'function' in candidate_dict and isinstance(candidate_dict['function'], dict):
        # check if 'name' exists within 'function' and is part of known tools
        if 'name' in candidate_dict['function'] and candidate_dict['function']['name'] in tool_names:
            candidate_dict["type"] = "function"  # ensure required property 'type' exists and has the right value
            # map property 'parameters' used by some older models to 'arguments'
            if "arguments" not in candidate_dict["function"] and "parameters" in candidate_dict["function"]:
                candidate_dict["function"]["arguments"] = candidate_dict["function"]["parameters"]
                del candidate_dict["function"]["parameters"]
            return candidate_dict
    return None


def _extract_balanced_json(text: str, start: int) -> str | None:
    """Extract a balanced JSON object from text starting at the given position.

    Walks through the string tracking brace depth and string boundaries
    to correctly handle arbitrary nesting levels.
    """
    if start >= len(text) or text[start] != '{':
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        c = text[i]
        if escape_next:
            escape_next = False
            continue
        if c == '\\' and in_string:
            escape_next = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _parse_channel_tool_calls(answer: str, tool_names: list[str]):
    """Parse channel-based tool calls used by GPT-OSS and similar models.

    Format:
        <|start|>assistant to=functions.func_name<|channel|>commentary json<|message|>{"arg": "value"}
    or:
        <|channel|>commentary to=functions.func_name <|constrain|>json<|message|>{"arg": "value"}
    """
    matches = []
    start_pos = None
    # Pattern 1: to=functions.NAME before <|channel|> (GPT-OSS primary format)
    # Pattern 2: to=functions.NAME after <|channel|> (alternative format)
    patterns = [
        r'to=functions\.([^<\s]+)\s*<\|channel\|>[^<]*<\|message\|>',
        r'<\|channel\|>\w+ to=functions\.([^<\s]+).*?<\|message\|>',
    ]
    for pattern in patterns:
        for m in re.finditer(pattern, answer):
            func_name = m.group(1).strip()
            if func_name not in tool_names:
                continue
            json_str = _extract_balanced_json(answer, m.end())
            if json_str is None:
                continue
            try:
                arguments = json.loads(json_str)
                if start_pos is None:
                    prefix = answer.rfind('<|start|>assistant', 0, m.start())
                    start_pos = prefix if prefix != -1 else m.start()
                matches.append({
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": arguments
                    }
                })
            except json.JSONDecodeError:
                pass
        if matches:
            break
    return matches, start_pos


def _parse_mistral_token_tool_calls(answer: str, tool_names: list[str]):
    """Parse Mistral/Devstral-style tool calls with [TOOL_CALLS] and [ARGS] special tokens.

    Format:
        [TOOL_CALLS]func_name[ARGS]{"arg": "value"}
    """
    matches = []
    start_pos = None
    for m in re.finditer(
        r'\[TOOL_CALLS\]\s*(\S+?)\s*\[ARGS\]\s*',
        answer
    ):
        func_name = m.group(1).strip()
        if func_name not in tool_names:
            continue
        json_str = _extract_balanced_json(answer, m.end())
        if json_str is None:
            continue
        try:
            arguments = json.loads(json_str)
            if start_pos is None:
                start_pos = m.start()
            matches.append({
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": arguments
                }
            })
        except json.JSONDecodeError:
            pass
    return matches, start_pos


def _parse_bare_name_tool_calls(answer: str, tool_names: list[str]):
    """Parse bare function-name style tool calls used by Mistral and similar models.

    Format:
        functionName{"arg": "value"}
    Multiple calls are concatenated directly or separated by whitespace.
    """
    matches = []
    start_pos = None
    # Match tool name followed by opening brace, then extract balanced JSON
    escaped_names = [re.escape(name) for name in tool_names]
    pattern = r'(?:' + '|'.join(escaped_names) + r')\s*\{'
    for match in re.finditer(pattern, answer):
        text = match.group(0)
        name = None
        for n in tool_names:
            if text.startswith(n):
                name = n
                break
        if not name:
            continue
        brace_start = match.end() - 1
        json_str = _extract_balanced_json(answer, brace_start)
        if json_str is None:
            continue
        try:
            arguments = json.loads(json_str)
            if start_pos is None:
                start_pos = match.start()
            matches.append({
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments
                }
            })
        except json.JSONDecodeError:
            pass
    return matches, start_pos


def _parse_xml_param_tool_calls(answer: str, tool_names: list[str]):
    """Parse XML-parameter style tool calls used by Qwen3.5 and similar models.

    Format:
        <tool_call>
        <function=function_name>
        <parameter=param_name>value</parameter>
        </function>
        </tool_call>
    """
    matches = []
    start_pos = None
    for tc_match in re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', answer, re.DOTALL):
        tc_content = tc_match.group(1)
        func_match = re.search(r'<function=([^>]+)>', tc_content)
        if not func_match:
            continue
        func_name = func_match.group(1).strip()
        if func_name not in tool_names:
            continue
        arguments = {}
        for param_match in re.finditer(r'<parameter=([^>]+)>\s*(.*?)\s*</parameter>', tc_content, re.DOTALL):
            param_name = param_match.group(1).strip()
            param_value = param_match.group(2).strip()
            try:
                param_value = json.loads(param_value)
            except (json.JSONDecodeError, ValueError):
                pass  # keep as string
            arguments[param_name] = param_value
        if start_pos is None:
            start_pos = tc_match.start()
        matches.append({
            "type": "function",
            "function": {
                "name": func_name,
                "arguments": arguments
            }
        })
    return matches, start_pos


def _parse_kimi_tool_calls(answer: str, tool_names: list[str]):
    """Parse Kimi-K2-style tool calls using pipe-delimited tokens.

    Format:
        <|tool_calls_section_begin|>
        <|tool_call_begin|>functions.func_name:index<|tool_call_argument_begin|>{"arg": "value"}<|tool_call_end|>
        <|tool_calls_section_end|>
    """
    matches = []
    start_pos = None
    for m in re.finditer(
        r'<\|tool_call_begin\|>\s*(?:functions\.)?(\S+?)(?::\d+)?\s*<\|tool_call_argument_begin\|>\s*',
        answer
    ):
        func_name = m.group(1).strip()
        if func_name not in tool_names:
            continue
        json_str = _extract_balanced_json(answer, m.end())
        if json_str is None:
            continue
        try:
            arguments = json.loads(json_str)
            if start_pos is None:
                # Check for section begin marker before the call marker
                section = answer.rfind('<|tool_calls_section_begin|>', 0, m.start())
                start_pos = section if section != -1 else m.start()
            matches.append({
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": arguments
                }
            })
        except json.JSONDecodeError:
            pass
    return matches, start_pos


def _parse_minimax_tool_calls(answer: str, tool_names: list[str]):
    """Parse MiniMax-style tool calls using invoke/parameter XML tags.

    Format:
        <minimax:tool_call>
        <invoke name="function_name">
        <parameter name="param_name">value</parameter>
        </invoke>
        </minimax:tool_call>
    """
    matches = []
    start_pos = None
    for tc_match in re.finditer(r'<minimax:tool_call>\s*(.*?)\s*</minimax:tool_call>', answer, re.DOTALL):
        tc_content = tc_match.group(1)
        # Split on <invoke> to handle multiple parallel calls in one block
        for invoke_match in re.finditer(r'<invoke\s+name="([^"]+)">(.*?)</invoke>', tc_content, re.DOTALL):
            func_name = invoke_match.group(1).strip()
            if func_name not in tool_names:
                continue
            invoke_body = invoke_match.group(2)
            arguments = {}
            for param_match in re.finditer(r'<parameter\s+name="([^"]+)">\s*(.*?)\s*</parameter>', invoke_body, re.DOTALL):
                param_name = param_match.group(1).strip()
                param_value = param_match.group(2).strip()
                try:
                    param_value = json.loads(param_value)
                except (json.JSONDecodeError, ValueError):
                    pass  # keep as string
                arguments[param_name] = param_value
            if start_pos is None:
                start_pos = tc_match.start()
            matches.append({
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": arguments
                }
            })
    return matches, start_pos


def _parse_deep_seek_tool_calls(answer: str, tool_names: list[str]):
    """Parse DeepSeek-style tool calls using fullwidth Unicode token delimiters.

    Format:
        <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>func_name<｜tool▁sep｜>{"arg": "value"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>
    """
    matches = []
    start_pos = None
    for m in re.finditer(
        r'<｜tool▁call▁begin｜>\s*(\S+?)\s*<｜tool▁sep｜>\s*',
        answer
    ):
        func_name = m.group(1).strip()
        if func_name not in tool_names:
            continue
        json_str = _extract_balanced_json(answer, m.end())
        if json_str is None:
            continue
        try:
            arguments = json.loads(json_str)
            if start_pos is None:
                # Check for section begin marker before the call marker
                section = answer.rfind('<｜tool▁calls▁begin｜>', 0, m.start())
                start_pos = section if section != -1 else m.start()
            matches.append({
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": arguments
                }
            })
        except json.JSONDecodeError:
            pass
    return matches, start_pos


def _parse_glm_tool_calls(answer: str, tool_names: list[str]):
    """Parse GLM-style tool calls using arg_key/arg_value XML pairs.

    Format:
        <tool_call>function_name
        <arg_key>key1</arg_key>
        <arg_value>value1</arg_value>
        </tool_call>
    """
    matches = []
    start_pos = None
    for tc_match in re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', answer, re.DOTALL):
        tc_content = tc_match.group(1)
        # First non-tag text is the function name
        name_match = re.match(r'([^<\s]+)', tc_content.strip())
        if not name_match:
            continue
        func_name = name_match.group(1).strip()
        if func_name not in tool_names:
            continue
        # Extract arg_key/arg_value pairs
        keys = [k.group(1).strip() for k in re.finditer(r'<arg_key>\s*(.*?)\s*</arg_key>', tc_content, re.DOTALL)]
        vals = [v.group(1).strip() for v in re.finditer(r'<arg_value>\s*(.*?)\s*</arg_value>', tc_content, re.DOTALL)]
        if len(keys) != len(vals):
            continue
        arguments = {}
        for k, v in zip(keys, vals):
            try:
                v = json.loads(v)
            except (json.JSONDecodeError, ValueError):
                pass  # keep as string
            arguments[k] = v
        if start_pos is None:
            start_pos = tc_match.start()
        matches.append({
            "type": "function",
            "function": {
                "name": func_name,
                "arguments": arguments
            }
        })
    return matches, start_pos


def _parse_pythonic_tool_calls(answer: str, tool_names: list[str]):
    """Parse pythonic-style tool calls used by Llama 4 and similar models.

    Format:
        [func_name(param1="value1", param2="value2"), func_name2(...)]
    """
    matches = []
    start_pos = None
    # Match a bracketed list of function calls
    bracket_match = re.search(r'\[([^\[\]]+)\]', answer)
    if not bracket_match:
        return matches, start_pos

    inner = bracket_match.group(1)

    # Build pattern for known tool names
    escaped_names = [re.escape(name) for name in tool_names]
    name_pattern = '|'.join(escaped_names)

    for call_match in re.finditer(
        r'(' + name_pattern + r')\(([^)]*)\)',
        inner
    ):
        func_name = call_match.group(1)
        params_str = call_match.group(2).strip()
        arguments = {}

        if params_str:
            # Parse key="value" pairs, handling commas inside quoted values
            for param_match in re.finditer(
                r'(\w+)\s*=\s*("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|[^,\)]+)',
                params_str
            ):
                param_name = param_match.group(1)
                param_value = param_match.group(2).strip()
                # Strip surrounding quotes
                if (param_value.startswith('"') and param_value.endswith('"')) or \
                   (param_value.startswith("'") and param_value.endswith("'")):
                    param_value = param_value[1:-1]
                # Try to parse as JSON for numeric/bool/null values
                try:
                    param_value = json.loads(param_value)
                except (json.JSONDecodeError, ValueError):
                    pass
                arguments[param_name] = param_value

        if start_pos is None:
            start_pos = bracket_match.start()
        matches.append({
            "type": "function",
            "function": {
                "name": func_name,
                "arguments": arguments
            }
        })

    return matches, start_pos


# Format registry: maps template substrings to the parser and streaming
# markers for that format.  When a format's hints are NOT found in the
# template, its parser and markers are excluded.
TOOL_CALL_FORMATS = [
    {
        'template_hints': ['tool▁call▁begin', 'tool▁calls▁begin'],
        'parser': _parse_deep_seek_tool_calls,
        'markers': ['<｜tool▁call▁begin｜>', '<｜tool▁calls▁begin｜>'],
    },
    {
        'template_hints': ['<|tool_call_begin|>', 'tool_calls_section'],
        'parser': _parse_kimi_tool_calls,
        'markers': ['<|tool_call_begin|>', '<|tool_calls_section_begin|>'],
    },
    {
        'template_hints': ['to=functions.', '<|channel|>'],
        'parser': _parse_channel_tool_calls,
        'markers': ['to=functions.', '<|channel|>commentary'],
    },
    {
        'template_hints': ['minimax:tool_call'],
        'parser': _parse_minimax_tool_calls,
        'markers': ['<minimax:tool_call>'],
    },
    {
        'template_hints': ['<arg_key>'],
        'parser': _parse_glm_tool_calls,
        'markers': ['<tool_call>'],
    },
    {
        'template_hints': ['<tool_call>'],
        'parser': _parse_xml_param_tool_calls,
        'markers': ['<tool_call>'],
    },
    {
        'template_hints': ['[TOOL_CALLS]'],
        'parser': _parse_mistral_token_tool_calls,
        'markers': ['[TOOL_CALLS]'],
    },
    {
        'template_hints': ['<function_call>'],
        'parser': None,
        'markers': ['<function_call>'],
    },
]

# Default ordered list of all specialized parsers.
ALL_PARSERS = [
    _parse_deep_seek_tool_calls,
    _parse_kimi_tool_calls,
    _parse_channel_tool_calls,
    _parse_minimax_tool_calls,
    _parse_glm_tool_calls,
    _parse_xml_param_tool_calls,
    _parse_mistral_token_tool_calls,
    _parse_bare_name_tool_calls,
    _parse_pythonic_tool_calls,
]


def detect_tool_call_format(template_str):
    """Inspect a chat/instruction template to determine which tool call
    formats are relevant.

    Uses an exclude-based approach: starts with all parsers/markers,
    then removes the ones whose hints are not found in the template.

    Returns (parsers, streaming_markers, check_bare_names).
    """
    if not template_str:
        return None, TOOL_CALL_OPENING_MARKERS, True

    matched_any = False
    exclude_parsers = []
    exclude_markers = []
    matched_markers = []

    for fmt in TOOL_CALL_FORMATS:
        if any(hint in template_str for hint in fmt['template_hints']):
            matched_any = True
            matched_markers.extend(fmt['markers'])
        else:
            if fmt['parser'] is not None:
                exclude_parsers.append(fmt['parser'])
            exclude_markers.extend(fmt['markers'])

    if not matched_any:
        return None, TOOL_CALL_OPENING_MARKERS, True

    parsers = [p for p in ALL_PARSERS if p not in exclude_parsers]
    markers = [m for m in TOOL_CALL_OPENING_MARKERS if m not in exclude_markers or m in matched_markers]

    return parsers, markers, False


def parse_tool_call(answer: str, tool_names: list[str], return_prefix: bool = False, parsers: list = None):
    matches = []
    start_pos = None

    def _return(matches, start_pos):
        if return_prefix:
            prefix = answer[:start_pos] if matches and start_pos is not None else ''
            return matches, prefix
        return matches

    # Try specialized parsers.
    for parser in (parsers if parsers is not None else ALL_PARSERS):
        matches, start_pos = parser(answer, tool_names)
        if matches:
            return _return(matches, start_pos)

    # Generic fallback: regex pattern to find the JSON content wrapped in <function>, <tools>, <tool_call>, and other tags observed from various models
    patterns = [r"(```[^\n]*)\n(.*?)```", r"<([^>]+)>(.*?)</\1>"]

    for pattern in patterns:
        for match in re.finditer(pattern, answer, re.DOTALL):
            if match.group(2) is None:
                continue
            # remove backtick wraps if present
            candidate = re.sub(r"^```(json|xml|python[^\n]*)\n", "", match.group(2).strip())
            candidate = re.sub(r"```$", "", candidate.strip())
            # unwrap inner tags
            candidate = re.sub(pattern, r"\2", candidate.strip(), flags=re.DOTALL)
            # llm might have generated multiple json objects separated by linebreaks, check for this pattern and try parsing each object individually
            if re.search(r"\}\s*\n\s*\{", candidate) is not None:
                candidate = re.sub(r"\}\s*\n\s*\{", "},\n{", candidate)
            if not candidate.strip().startswith("["):
                candidate = "[" + candidate + "]"

            candidates = []
            try:
                # parse the candidate JSON into a dictionary
                candidates = json.loads(candidate)
                if not isinstance(candidates, list):
                    candidates = [candidates]
            except json.JSONDecodeError:
                # Ignore invalid JSON silently
                continue

            for candidate_dict in candidates:
                checked_candidate = check_and_sanitize_tool_call_candidate(candidate_dict, tool_names)
                if checked_candidate is not None:
                    if start_pos is None:
                        start_pos = match.start()
                    matches.append(checked_candidate)

        # last resort if nothing has been mapped: LLM might have produced plain json tool call without xml-like tags
        if len(matches) == 0:
            try:
                candidate = answer
                # llm might have generated multiple json objects separated by linebreaks, check for this pattern and try parsing each object individually
                if re.search(r"\}\s*\n\s*\{", candidate) is not None:
                    candidate = re.sub(r"\}\s*\n\s*\{", "},\n{", candidate)
                if not candidate.strip().startswith("["):
                    candidate = "[" + candidate + "]"
                # parse the candidate JSON into a dictionary
                candidates = json.loads(candidate)
                if not isinstance(candidates, list):
                    candidates = [candidates]
                for candidate_dict in candidates:
                    checked_candidate = check_and_sanitize_tool_call_candidate(candidate_dict, tool_names)
                    if checked_candidate is not None:
                        matches.append(checked_candidate)
            except json.JSONDecodeError:
                # Ignore invalid JSON silently
                pass

    return _return(matches, start_pos)
