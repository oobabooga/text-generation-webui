import concurrent.futures

import requests


def download_single(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    response = requests.get(url, headers=headers, timeout=5)
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
