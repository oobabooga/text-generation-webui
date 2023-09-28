import concurrent.futures
import requests
import re

from bs4 import BeautifulSoup

import extensions.superboogav2.parameters as parameters

from .data_processor import process_and_add_to_collector
from .utils import create_metadata_source

def _download_single(url):
    response = requests.get(url, timeout=5)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception("Failed to download URL")


def _download_urls(urls, threads=1):
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        futures = []
        for url in urls:
            future = executor.submit(_download_single, url)
            futures.append(future)

        results = []
        i = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                i += 1
                yield f"{i}/{len(urls)}", results
            except Exception:
                pass

        yield "Done", results


def feed_url_into_collector(urls, collector):
    all_text = ''
    cumulative = ''

    urls = urls.strip().split('\n')
    cumulative += f'Loading {len(urls)} URLs with {parameters.get_num_threads()} threads...\n\n'
    yield cumulative
    for update, contents in _download_urls(urls, threads=parameters.get_num_threads()):
        yield cumulative + update

    cumulative += 'Processing the HTML sources...'
    yield cumulative
    for content in contents:
        soup = BeautifulSoup(content, features="lxml")
        for script in soup(["script", "style"]):
            script.extract()

        strings = soup.stripped_strings
        if parameters.get_is_strong_cleanup():
            strings = [s for s in strings if re.search("[A-Za-z] ", s)]

        text = '\n'.join([s.strip() for s in strings])
        all_text += text

    process_and_add_to_collector(all_text, collector, False, create_metadata_source('url-download'))