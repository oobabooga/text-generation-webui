import concurrent.futures
import html
import random
import re
import urllib.request
from concurrent.futures import as_completed
from datetime import datetime
from urllib.parse import quote_plus, urlparse, parse_qs, unquote

import requests

from modules import shared
from modules.logging_colors import logger


def get_current_timestamp():
    """Returns the current time in 24-hour format"""
    return datetime.now().strftime('%b %d, %Y %H:%M')


def download_web_page(url, timeout=10):
    """
    Download a web page and convert its HTML content to Markdown text,
    handling Brotli/gzip and non-HTML content robustly.
    """
    logger.info(f"download_web_page {url}")

    # --- soft deps
    try:
        import html2text
    except Exception:
        logger.exception("html2text import failed")
        html2text = None

    try:
        from readability import Document
    except Exception:
        Document = None

    try:
        import brotli as _brotli
        have_brotli = True
    except Exception:
        _brotli = None
        have_brotli = False

    import gzip, zlib, re, html as _html

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        # IMPORTANT: only advertise br if brotli is installed
        "Accept-Encoding": "gzip, deflate" + (", br" if have_brotli else ""),
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()

        # --- bail out early if it's not HTML
        ctype = resp.headers.get("Content-Type", "").lower()
        if not any(t in ctype for t in ("text/html", "application/xhtml+xml")):
            logger.warning("Non-HTML content-type %r at %s", ctype, url)
            return ""

        # --- get raw bytes then decompress if server didn't/requests couldn't
        raw = resp.content  # bytes
        enc_hdr = resp.headers.get("Content-Encoding", "").lower()

        # If requests didn't decode (it normally does gzip/deflate), handle manually.
        if "br" in enc_hdr and have_brotli:
            try:
                raw = _brotli.decompress(raw)
            except Exception:
                # it may already be decoded; ignore
                pass
        elif "gzip" in enc_hdr:
            try:
                raw = gzip.decompress(raw)
            except Exception:
                pass
        elif "deflate" in enc_hdr:
            try:
                raw = zlib.decompress(raw, -zlib.MAX_WBITS)
            except Exception:
                pass

        # --- decode text with a robust charset guess
        # use HTTP charset if present
        charset = None
        if "charset=" in ctype:
            charset = ctype.split("charset=")[-1].split(";")[0].strip()
        if not charset:
            # requests’ detector
            charset = resp.apparent_encoding or "utf-8"
        try:
            html_text = raw.decode(charset, errors="replace")
        except Exception:
            html_text = raw.decode("utf-8", errors="replace")

        # anti-bot shells (avoid empty output surprises)
        if re.search(r"(cf-chl|Just a moment|enable JavaScript)", html_text, re.I):
            logger.warning("Possible anti-bot/challenge page at %s", url)

        # --- extract readable text (readability -> html2text -> fallback)
        md_readability = ""
        if Document is not None:
            try:
                doc = Document(html_text)
                title = (doc.short_title() or "").strip()
                main_html = doc.summary(html_partial=True)
                main_text = re.sub(r"<[^>]+>", " ", main_html, flags=re.S)
                main_text = re.sub(r"\s+", " ", main_text).strip()
                if title:
                    md_readability = f"# {title}\n\n{main_text}".strip()
                else:
                    md_readability = main_text
            except Exception:
                logger.exception("readability failed on %s", url)

        md_html2text = ""
        if html2text is not None:
            try:
                h = html2text.HTML2Text()
                h.body_width = 0
                h.ignore_images = True
                h.ignore_links = True
                h.single_line_break = True
                md_html2text = (h.handle(html_text) or "").strip()
            except Exception:
                logger.exception("html2text failed on %s", url)

        def _clean(s):
            s = re.sub(r"<[^>]+>", " ", s, flags=re.S)
            return _html.unescape(re.sub(r"\s+", " ", s)).strip()

        # fallback: meta/title/headers/paragraphs + noscript
        parts = []
        t = re.search(r"<title[^>]*>(.*?)</title>", html_text, re.I | re.S)
        if t:
            parts.append(f"# {_clean(t.group(1))}")

        for pat in [
            r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\'](.*?)["\']',
            r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\'](.*?)["\']',
            r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']',
            r'<meta[^>]+name=["\']twitter:description["\'][^>]+content=["\'](.*?)["\']',
        ]:
            m = re.search(pat, html_text, re.I | re.S)
            if m:
                parts.append(_clean(m.group(1)))

        parts += [f"## {_clean(h)}" for h in re.findall(r"<h[1-3][^>]*>(.*?)</h[1-3]>", html_text, re.I | re.S)[:4] if _clean(h)]
        parts += [_clean(p) for p in re.findall(r"<p[^>]*>(.*?)</p>", html_text, re.I | re.S)[:8] if _clean(p)]
        for n in re.findall(r"<noscript[^>]*>(.*?)</noscript>", html_text, re.I | re.S):
            c = _clean(n)
            if c:
                parts.append(c)
        md_fallback = "\n\n".join([p for p in parts if p]).strip()

        best = max([md_readability, md_html2text, md_fallback], key=lambda s: len(s or ""))
        if not best.strip():
            logger.warning("Empty content extracted from %s", url)
        return best

    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading {url}: {e}")
        return ""
    except Exception:
        logger.exception("Unexpected error while downloading %s", url)
        return ""



def _extract_results_from_duckduckgo(response_text, num_pages):
    # 1) Grab the title anchors (they carry the real clickable href)
    #    We capture both the inner text (title) and href.
    anchor_pattern = re.compile(
        r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        re.DOTALL | re.IGNORECASE
    )
    matches = anchor_pattern.findall(response_text)

    results = []
    for href, title_html in matches:
        # 2) Resolve DuckDuckGo redirect: ?uddg=<encoded_target>
        parsed = urlparse(href)
        target_url = href
        if parsed.netloc.endswith("duckduckgo.com"):
            qs = parse_qs(parsed.query)
            if "uddg" in qs and qs["uddg"]:
                target_url = unquote(qs["uddg"][0])

        # 3) Clean title
        title_text = re.sub(r'<[^>]+>', '', title_html).strip()
        title_text = html.unescape(title_text)

        # 4) Basic normalization: add scheme if missing
        if target_url.startswith("//"):
            target_url = "https:" + target_url
        elif not re.match(r'^https?://', target_url, flags=re.I):
            target_url = "https://" + target_url

        results.append((target_url, title_text))

        if len(results) >= num_pages:
            break

    return results

def perform_web_search(query, num_pages=3, max_workers=5, timeout=10):
    """Perform web search and return results with content"""
    try:
        search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"

        agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        ]

        req = urllib.request.Request(search_url, headers={'User-Agent': random.choice(agents)})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            response_text = response.read().decode('utf-8', errors='replace')

        # Extract (url, title) pairs from the proper anchors
        download_tasks = _extract_results_from_duckduckgo(response_text, num_pages)

        if not download_tasks:
            return []

        search_results = [None] * len(download_tasks)

        # Download pages in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(download_web_page, url, timeout): (i, url, title)
                for i, (url, title) in enumerate(download_tasks)
            }

            for future in as_completed(future_to_index):
                i, url, title = future_to_index[future]
                try:
                    content = future.result()
                except Exception:
                    content = ""

                search_results[i] = {
                    "title": title,
                    "url": url,
                    "content": content or ""
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
    logger.info(f"add_web_search_attachments")
    if not search_query:
        logger.warning("No search query provided")
        return

    try:
        logger.info(f"Add Web Search - Using search query: {search_query}")

        # Perform web search
        num_pages = int(state.get('web_search_pages', 3))
        search_results = perform_web_search(search_query, num_pages)

        if not search_results:
            logger.warning("No search results found")
            return

        # Filter out failed downloads before adding attachments
        #  logger.info(f"search_results {search_results}")
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
