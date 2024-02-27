import builtins
import io

import requests

from modules.logging_colors import logger

original_open = open
original_get = requests.get
original_print = print


class RequestBlocker:

    def __enter__(self):
        requests.get = my_get

    def __exit__(self, exc_type, exc_value, traceback):
        requests.get = original_get


class OpenMonkeyPatch:

    def __enter__(self):
        builtins.open = my_open
        builtins.print = my_print

    def __exit__(self, exc_type, exc_value, traceback):
        builtins.open = original_open
        builtins.print = original_print


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

        file_contents = file_contents.replace(b'\t\t<script\n\t\t\tsrc="https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/4.3.9/iframeResizer.contentWindow.min.js"\n\t\t\tasync\n\t\t></script>', b'')
        file_contents = file_contents.replace(b'cdnjs.cloudflare.com', b'127.0.0.1')

        return io.BytesIO(file_contents)
    else:
        return original_open(*args, **kwargs)


def my_print(*args, **kwargs):
    if len(args) > 0 and 'To create a public link, set `share=True`' in args[0]:
        return
    else:
        if len(args) > 0 and 'Running on local URL' in args[0]:
            args = list(args)
            args[0] = f"\n{args[0].strip()}\n"
            args = tuple(args)

        original_print(*args, **kwargs)
