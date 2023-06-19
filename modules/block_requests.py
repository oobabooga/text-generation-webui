import requests

from modules.logging_colors import logger


class RequestBlocker:

    def __enter__(self):
        self.original_get = requests.get
        requests.get = my_get

    def __exit__(self, exc_type, exc_value, traceback):
        requests.get = self.original_get


def my_get(url, **kwargs):
    logger.info('Unwanted HTTP request redirected to localhost :)')
    kwargs.setdefault('allow_redirects', True)
    return requests.api.request('get', 'http://127.0.0.1/', **kwargs)
