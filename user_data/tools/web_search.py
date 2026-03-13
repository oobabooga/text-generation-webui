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
            },
            "required": ["query"]
        }
    }
}


def execute(arguments):
    query = arguments.get("query", "")
    results = perform_web_search(query, num_pages=3)
    output = []
    for r in results:
        if r and r["content"].strip():
            output.append({"title": r["title"], "url": r["url"], "content": truncate_content_by_tokens(r["content"])})

    return output if output else [{"error": "No results found."}]
