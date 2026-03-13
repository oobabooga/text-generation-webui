from modules.web_search import download_web_page, truncate_content_by_tokens

tool = {
    "type": "function",
    "function": {
        "name": "fetch_webpage",
        "description": "Fetch and read the contents of a web page given its URL. Returns the page content as plain text.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL of the web page to fetch."},
            },
            "required": ["url"]
        }
    }
}


def execute(arguments):
    url = arguments.get("url", "")
    if not url:
        return {"error": "No URL provided."}

    content = download_web_page(url, include_links=True)
    if not content or not content.strip():
        return {"error": f"Failed to fetch content from {url}"}

    return {"url": url, "content": truncate_content_by_tokens(content)}
