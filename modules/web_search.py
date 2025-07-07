import concurrent.futures
import html
import re
from concurrent.futures import as_completed
from datetime import datetime
from urllib.parse import quote_plus

import requests

from modules import shared
from modules.logging_colors import logger


def get_current_timestamp():
    """Returns the current time in 24-hour format"""
    return datetime.now().strftime('%b %d, %Y %H:%M')


def download_web_page(url, timeout=10):
    """
    Download a web page and convert its HTML content to structured Markdown text.
    """
    import html2text

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Initialize the HTML to Markdown converter
        h = html2text.HTML2Text()
        h.body_width = 0
        h.ignore_images = True
        h.ignore_links = True

        # Convert the HTML to Markdown
        markdown_text = h.handle(response.text)

        return markdown_text
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading {url}: {e}")
        return ""
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return ""


def perform_web_search(query, num_pages=3, max_workers=5, timeout=10):
    """Perform web search and return results with content"""
    try:
        # Use DuckDuckGo HTML search endpoint
        search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

        response = requests.get(search_url, headers=headers, timeout=timeout)
        response.raise_for_status()

        # Extract results with regex
        titles = re.findall(r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*>(.*?)</a>', response.text, re.DOTALL)
        urls = re.findall(r'<a[^>]*class="[^"]*result__url[^"]*"[^>]*>(.*?)</a>', response.text, re.DOTALL)

        # Prepare download tasks
        download_tasks = []
        for i in range(min(len(titles), len(urls), num_pages)):
            url = f"https://{urls[i].strip()}"
            title = re.sub(r'<[^>]+>', '', titles[i]).strip()
            title = html.unescape(title)
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
                except Exception:
                    search_results[index] = {
                        'title': title,
                        'url': url,
                        'content': ''
                    }

        return search_results

    except Exception as e:
        logger.error(f"Error performing web search: {e}")
        return []


def truncate_content_by_tokens(content, max_tokens=8192):
    """Truncate content to fit within token limit using binary search"""
    if len(shared.tokenizer.encode(content)) <= max_tokens:
        return content

    left, right = 0, len(content)
    while left < right:
        mid = (left + right + 1) // 2
        if len(shared.tokenizer.encode(content[:mid])) <= max_tokens:
            left = mid
        else:
            right = mid - 1

    return content[:left]


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

        # Filter out failed downloads before adding attachments
        successful_results = [result for result in search_results if result['content'].strip()]

        if not successful_results:
            logger.warning("No successful downloads to add as attachments")
            return

        # Add search results as attachments
        key = f"user_{row_idx}"
        if key not in history['metadata']:
            history['metadata'][key] = {"timestamp": get_current_timestamp()}
        if "attachments" not in history['metadata'][key]:
            history['metadata'][key]["attachments"] = []

        for result in successful_results:
            attachment = {
                "name": result['title'],
                "type": "text/html",
                "url": result['url'],
                "content": truncate_content_by_tokens(result['content'])
            }
            history['metadata'][key]["attachments"].append(attachment)

        logger.info(f"Added {len(successful_results)} successful web search results as attachments.")

    except Exception as e:
        logger.error(f"Error in web search: {e}")
