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

        if len(args) > 1 and args[1] == 'rb':
            file_contents = file_contents.decode('utf-8')

        file_contents = file_contents.replace('\t\t<script\n\t\t\tsrc="https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/4.3.9/iframeResizer.contentWindow.min.js"\n\t\t\tasync\n\t\t></script>', '')
        file_contents = file_contents.replace('cdnjs.cloudflare.com', '127.0.0.1')
        file_contents = file_contents.replace(
            '</head>',
            '\n    <script src="file/js/katex/katex.min.js"></script>'
            '\n    <script src="file/js/katex/auto-render.min.js"></script>'
            '\n    <script src="file/js/highlightjs/highlight.min.js"></script>'
            '\n    <script src="file/js/highlightjs/highlightjs-copy.min.js"></script>'
            '\n    <script>hljs.addPlugin(new CopyButtonPlugin());</script>'
            '\n  </head>'
        )

        if len(args) > 1 and args[1] == 'rb':
            file_contents = file_contents.encode('utf-8')
            return io.BytesIO(file_contents)
        else:
            return io.StringIO(file_contents)

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
