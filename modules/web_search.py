import concurrent.futures
import html
import ipaddress
import random
import re
import socket
from concurrent.futures import as_completed
from datetime import datetime
from urllib.parse import parse_qs, quote_plus, urljoin, urlparse

import requests

from modules import shared
from modules.logging_colors import logger


def _validate_url(url):
    """Validate that a URL is safe to fetch (not targeting private/internal networks)."""
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https'):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("No hostname in URL")

    # Resolve hostname and check all returned addresses
    try:
        for family, _, _, _, sockaddr in socket.getaddrinfo(hostname, None):
            ip = ipaddress.ip_address(sockaddr[0])
            if not ip.is_global:
                raise ValueError(f"Access to non-public address {ip} is blocked")
    except socket.gaierror:
        raise ValueError(f"Could not resolve hostname: {hostname}")


def get_current_timestamp():
    """Returns the current time in 24-hour format"""
    return datetime.now().strftime('%b %d, %Y %H:%M')


def download_web_page(url, timeout=10, include_links=False):
    """
    Download a web page and extract its main content as Markdown text.
    """
    import trafilatura

    try:
        _validate_url(url)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        max_redirects = 5
        for _ in range(max_redirects):
            response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=False)
            if response.is_redirect and 'Location' in response.headers:
                url = urljoin(url, response.headers['Location'])
                _validate_url(url)
            else:
                break

        response.raise_for_status()

        result = trafilatura.extract(
            response.text,
            include_links=include_links,
            output_format='markdown',
            url=url
        )
        return result or ""
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading {url}: {e}")
        return ""
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return ""


def perform_web_search(query, num_pages=3, max_workers=5, timeout=10, fetch_content=True):
    """Perform web search and return results, optionally with page content"""
    try:
        search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"

        agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        ]

        response = requests.get(search_url, headers={'User-Agent': random.choice(agents)}, timeout=timeout)
        response.raise_for_status()
        response_text = response.text

        # Extract results - title and URL come from the same <a class="result__a"> element
        result_links = re.findall(r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*>(.*?)</a>', response_text, re.DOTALL)
        result_tags = re.findall(r'<a([^>]*class="[^"]*result__a[^"]*"[^>]*)>', response_text, re.DOTALL)

        # Prepare download tasks
        download_tasks = []
        for i, (tag_attrs, raw_title) in enumerate(zip(result_tags, result_links)):
            if num_pages is not None and i >= num_pages:
                break
            # Extract href and resolve the actual URL from DuckDuckGo's redirect link
            href_match = re.search(r'href="([^"]*)"', tag_attrs)
            if not href_match:
                continue
            uddg = parse_qs(urlparse(html.unescape(href_match.group(1))).query).get('uddg', [''])[0]
            if not uddg:
                continue
            title = html.unescape(re.sub(r'<[^>]+>', '', raw_title).strip())
            download_tasks.append((uddg, title, len(download_tasks)))

        search_results = [None] * len(download_tasks)  # Pre-allocate to maintain order

        if not fetch_content:
            for url, title, index in download_tasks:
                search_results[index] = {
                    'title': title,
                    'url': url,
                    'content': ''
                }

            return search_results

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
