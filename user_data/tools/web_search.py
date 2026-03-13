from modules.web_search import perform_web_search, truncate_content_by_tokens

tool = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web using DuckDuckGo and return page contents.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "num_pages": {"type": "integer", "description": "Number of search result pages to fetch (default: 3)."},
                "max_tokens": {"type": "integer", "description": "Maximum number of tokens per page result (default: 2048)."},
            },
            "required": ["query"]
        }
    }
}


def execute(arguments):
    query = arguments.get("query", "")
    num_pages = arguments.get("num_pages", 3)
    max_tokens = arguments.get("max_tokens", 2048)
    results = perform_web_search(query, num_pages=num_pages)
    output = []
    for r in results:
        if r and r["content"].strip():
            output.append({"title": r["title"], "url": r["url"], "content": truncate_content_by_tokens(r["content"], max_tokens=max_tokens)})

    return output if output else [{"error": "No results found."}]
