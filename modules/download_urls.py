import concurrent.futures

import requests


def download_single(url):
    response = requests.get(url, timeout=5)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception("Failed to download URL")


def download_urls(urls, threads=1):
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        futures = []
        for url in urls:
            future = executor.submit(download_single, url)
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
