import requests
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
from datetime import datetime

from modules.text_generation import generate_reply
from modules.logging_colors import logger


def get_current_timestamp():
    """Returns the current time in 24-hour format"""
    return datetime.now().strftime('%b %d, %Y %H:%M')


def generate_search_query(user_message, state):
    """Generate a search query from user message using the LLM"""
    search_prompt = f"{user_message}\n\n=====\n\nPlease turn the message above into a short web search query in the same language as the message. Respond with only the search query, nothing else."
    
    # Use a minimal state for search query generation
    search_state = state.copy()
    search_state['max_new_tokens'] = 50
    search_state['temperature'] = 0.1
    
    query = ""
    for reply in generate_reply(search_prompt, search_state, stopping_strings=[], is_chat=False):
        query = reply.strip()
    
    return query


def download_web_page(url, timeout=10):
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
        
        # Limit text length to prevent overwhelming the context
        if len(text) > 5000:
            text = text[:5000] + "... [content truncated]"
        
        return text
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return f"[Error downloading content from {url}: {str(e)}]"


def perform_web_search(query, num_pages=3):
    """Perform web search and return results with content"""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_pages))
        
        search_results = []
        for i, result in enumerate(results):
            url = result.get('href', '')
            title = result.get('title', f'Search Result {i+1}')
            
            # Download page content
            content = download_web_page(url)
            
            search_results.append({
                'title': title,
                'url': url,
                'content': content
            })
        
        return search_results
    except Exception as e:
        logger.error(f"Error performing web search: {e}")
        return []


def add_web_search_attachments(history, row_idx, user_message, state):
    """Perform web search and add results as attachments"""
    if not state.get('enable_web_search', False):
        return
    
    try:
        # Generate search query
        search_query = generate_search_query(user_message, state)
        if not search_query:
            logger.warning("Failed to generate search query")
            return
        
        logger.info(f"Generated search query: {search_query}")
        
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
                "name": f"Web Search: {result['title']}",
                "type": "text/html",
                "content": f"URL: {result['url']}\n\n{result['content']}"
            }
            history['metadata'][key]["attachments"].append(attachment)
        
        logger.info(f"Added {len(search_results)} web search results as attachments")
        
    except Exception as e:
        logger.error(f"Error in web search: {e}")
