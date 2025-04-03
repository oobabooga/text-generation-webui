import base64
import os
import time
import json
import random
import re
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


def parseToolCall(answer: str, tool_names: list[str]):
    pattern = r"```(.*?)```"

    matches = []

    for match in re.finditer(pattern, answer, re.DOTALL):
        candidate = re.sub(r"^(json|python[^\n]*)\n", "", match.group(1).strip())
        try:
            # parse the candidate JSON into a dictionary
            candidates = json.loads(candidate)
            if not isinstance(candidates, list):
                candidates = [candidates]

            for candidate_dict in candidates:
                # check if property 'function' exists and is a dictionary, otherwise adapt dict
                if 'function' not in candidate_dict and 'name' in candidate_dict and isinstance(candidate_dict['name'], str):
                    candidate_dict = {"type": "function", "function": candidate_dict}
                if 'function' in candidate_dict and isinstance(candidate_dict['function'], dict):
                    # check if 'name' exists within 'function' and is part of known tools
                    if 'name' in candidate_dict['function'] and candidate_dict['function']['name'] in tool_names:
                        candidate_dict["type"] = "function"  # ensure required property 'type' exists and has the right value
                        # map property 'parameters' used by some older models to 'arguments'
                        if "arguments" not in candidate_dict["function"] and "parameters" in candidate_dict["function"]:
                            candidate_dict["function"]["arguments"] = candidate_dict["function"]["parameters"]
                            del candidate_dict["function"]["parameters"]
                        matches.append(candidate_dict)

        except json.JSONDecodeError:
            # Ignore invalid JSON silently
            continue

    return matches
