import base64
import os

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


def end_line(s):
    if s and s[-1] != '\n':
        s = s + '\n'
    return s


def debug_msg(*args, **kwargs):
    from extensions.openai.script import params
    if os.environ.get("OPENEDAI_DEBUG", params.get('debug', 0)):
        print(*args, **kwargs)


def default(dic, key, default):
    '''
    little helper to get defaults if arg is present but None and should be the same type as default.
    '''
    val = dic.get(key, default)
    if not isinstance(val, type(default)):
        # maybe it's just something like 1 instead of 1.0
        try:
            v = type(default)(val)
            if type(val)(v) == val:  # if it's the same value passed in, it's ok.
                return v
        except:
            pass

        val = default

    return val
