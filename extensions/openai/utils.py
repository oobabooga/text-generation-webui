import base64
import json
import os
import random
import re
import time
import traceback
from typing import Callable, Optional

import numpy as np


def float_list_to_base64(float_array: np.ndarray) -> str:
    # Convert the list to a float32 array that the OpenAPI client expects
    # float_array = np.array(float_list, dtype="float32")

    # Get raw bytes
    bytes_array = float_array.tobytes()

    # Encode bytes into base64
    encoded_bytes = base64.b64encode(bytes_array)

    # Turn raw base64 encoded bytes into ASCII
    ascii_string = encoded_bytes.decode('ascii')
    return ascii_string


def debug_msg(*args, **kwargs):
    from extensions.openai.script import params
    if os.environ.get("OPENEDAI_DEBUG", params.get('debug', 0)):
        print(*args, **kwargs)


def _start_cloudflared(port: int, tunnel_id: str, max_attempts: int = 3, on_start: Optional[Callable[[str], None]] = None):
    try:
        from flask_cloudflared import _run_cloudflared
    except ImportError:
        print('You should install flask_cloudflared manually')
        raise Exception(
            'flask_cloudflared not installed. Make sure you installed the requirements.txt for this extension.')

    for _ in range(max_attempts):
        try:
            if tunnel_id is not None:
                public_url = _run_cloudflared(port, port + 1, tunnel_id=tunnel_id)
            else:
                public_url = _run_cloudflared(port, port + 1)

            if on_start:
                on_start(public_url)

            return
        except Exception:
            traceback.print_exc()
            time.sleep(3)

        raise Exception('Could not start cloudflared.')


def getToolCallId() -> str:
    letter_bytes = "abcdefghijklmnopqrstuvwxyz0123456789"
    b = [random.choice(letter_bytes) for _ in range(8)]
    return "call_" + "".join(b).lower()


def checkAndSanitizeToolCallCandidate(candidate_dict: dict, tool_names: list[str]):
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


def _parseChannelToolCalls(answer: str, tool_names: list[str]):
    """Parse channel-based tool calls used by GPT-OSS and similar models.

    Format:
        <|channel|>commentary to=functions.func_name <|constrain|>json<|message|>{"arg": "value"}
    """
    matches = []
    for m in re.finditer(
        r'<\|channel\|>commentary to=functions\.([^<\s]+)\s*(?:<\|constrain\|>json)?<\|message\|>(\{[^}]*(?:\{[^}]*\}[^}]*)*\})',
        answer
    ):
        func_name = m.group(1).strip()
        if func_name not in tool_names:
            continue
        try:
            arguments = json.loads(m.group(2))
            matches.append({
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": arguments
                }
            })
        except json.JSONDecodeError:
            pass
    return matches


def _parseBareNameToolCalls(answer: str, tool_names: list[str]):
    """Parse bare function-name style tool calls used by Mistral and similar models.

    Format:
        functionName{"arg": "value"}
    Multiple calls are concatenated directly or separated by whitespace.
    """
    matches = []
    # Build pattern that matches any known tool name followed by a JSON object
    escaped_names = [re.escape(name) for name in tool_names]
    pattern = r'(?:' + '|'.join(escaped_names) + r')\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    for match in re.finditer(pattern, answer):
        text = match.group(0)
        # Split into function name and JSON arguments
        for name in tool_names:
            if text.startswith(name):
                json_str = text[len(name):].strip()
                try:
                    arguments = json.loads(json_str)
                    matches.append({
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": arguments
                        }
                    })
                except json.JSONDecodeError:
                    pass
                break
    return matches


def _parseXmlParamToolCalls(answer: str, tool_names: list[str]):
    """Parse XML-parameter style tool calls used by Qwen3.5 and similar models.

    Format:
        <tool_call>
        <function=function_name>
        <parameter=param_name>value</parameter>
        </function>
        </tool_call>
    """
    matches = []
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
        matches.append({
            "type": "function",
            "function": {
                "name": func_name,
                "arguments": arguments
            }
        })
    return matches


def parseToolCall(answer: str, tool_names: list[str]):
    matches = []

    # abort on very short answers to save computation cycles
    if len(answer) < 10:
        return matches

    # Check for channel-based tool calls (e.g. GPT-OSS format)
    matches = _parseChannelToolCalls(answer, tool_names)
    if matches:
        return matches

    # Check for XML-parameter style tool calls (e.g. Qwen3.5 format)
    matches = _parseXmlParamToolCalls(answer, tool_names)
    if matches:
        return matches

    # Check for bare function-name style tool calls (e.g. Mistral format)
    matches = _parseBareNameToolCalls(answer, tool_names)
    if matches:
        return matches

    # Define the regex pattern to find the JSON content wrapped in <function>, <tools>, <tool_call>, and other tags observed from various models
    patterns = [r"(```[^\n]*)\n(.*?)```", r"<([^>]+)>(.*?)</\1>"]

    for pattern in patterns:
        for match in re.finditer(pattern, answer, re.DOTALL):
            # print(match.group(2))
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
                checked_candidate = checkAndSanitizeToolCallCandidate(candidate_dict, tool_names)
                if checked_candidate is not None:
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
                    checked_candidate = checkAndSanitizeToolCallCandidate(candidate_dict, tool_names)
                    if checked_candidate is not None:
                        matches.append(checked_candidate)
            except json.JSONDecodeError:
                # Ignore invalid JSON silently
                pass

    return matches
