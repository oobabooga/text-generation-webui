import concurrent.futures
from concurrent.futures import as_completed
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

from modules.logging_colors import logger


def get_current_timestamp():
    """Returns the current time in 24-hour format"""
    return datetime.now().strftime('%b %d, %Y %H:%M')


def download_web_page(url, timeout=5):
    """Download and extract text from a web page"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text and clean it up
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        return text
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return f"[Error downloading content from {url}: {str(e)}]"


def perform_web_search(query, num_pages=3, max_workers=5):
    """Perform web search and return results with content"""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_pages))

        # Prepare download tasks
        download_tasks = []
        for i, result in enumerate(results):
            url = result.get('href', '')
            title = result.get('title', f'Search Result {i+1}')
            download_tasks.append((url, title, i))

        search_results = [None] * len(download_tasks)  # Pre-allocate to maintain order

        # Download pages in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            future_to_task = {
                executor.submit(download_web_page, task[0]): task
                for task in download_tasks
            }

            # Collect results as they complete
            for future in as_completed(future_to_task):
                url, title, index = future_to_task[future]
                try:
                    content = future.result()
                    search_results[index] = {
                        'title': title,
                        'url': url,
                        'content': content
                    }
                except Exception as e:
                    logger.error(f"Error downloading {url}: {e}")
                    # Include failed downloads with empty content
                    search_results[index] = {
                        'title': title,
                        'url': url,
                        'content': ''
                    }

        return search_results

    except Exception as e:
        logger.error(f"Error performing web search: {e}")
        return []


def add_web_search_attachments(history, row_idx, user_message, search_query, state):
    """Perform web search and add results as attachments"""
    if not search_query:
        logger.warning("No search query provided")
        return

    try:
        logger.info(f"Using search query: {search_query}")

        # Perform web search
        num_pages = int(state.get('web_search_pages', 3))
        search_results = perform_web_search(search_query, num_pages)

        if not search_results:
            logger.warning("No search results found")
            return

        # Add search results as attachments
        key = f"user_{row_idx}"
        if key not in history['metadata']:
            history['metadata'][key] = {"timestamp": get_current_timestamp()}
        if "attachments" not in history['metadata'][key]:
            history['metadata'][key]["attachments"] = []

        for result in search_results:
            attachment = {
                "name": result['title'],
                "type": "text/html",
                "url": result['url'],
                "content": result['content']
            }
            history['metadata'][key]["attachments"].append(attachment)

        logger.info(f"Added {len(search_results)} web search results as attachments")

    except Exception as e:
        logger.error(f"Error in web search: {e}")
