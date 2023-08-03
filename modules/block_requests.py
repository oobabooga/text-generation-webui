import builtins
import io

import requests

from modules.logging_colors import logger

original_open = open
original_get = requests.get


class RequestBlocker:

    def __enter__(self):
        requests.get = my_get

    def __exit__(self, exc_type, exc_value, traceback):
        requests.get = original_get


class OpenMonkeyPatch:

    def __enter__(self):
        builtins.open = my_open

    def __exit__(self, exc_type, exc_value, traceback):
        builtins.open = original_open


def my_get(url, **kwargs):
    logger.info('Unwanted HTTP request redirected to localhost :)')
    kwargs.setdefault('allow_redirects', True)
    return requests.api.request('get', 'http://127.0.0.1/', **kwargs)


# Kindly provided by our friend WizardLM-30B
def my_open(*args, **kwargs):
    filename = str(args[0])
    if filename.endswith('index.html'):
        with original_open(*args, **kwargs) as f:
            file_contents = f.read()

        file_contents = file_contents.replace(b'<script src="https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/4.3.1/iframeResizer.contentWindow.min.js"></script>', b'')
        file_contents = file_contents.replace(b'cdnjs.cloudflare.com', b'127.0.0.1')
        return io.BytesIO(file_contents)
    else:
        return original_open(*args, **kwargs)
