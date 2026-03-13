from modules.web_search import perform_web_search

tool = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web using DuckDuckGo and return a list of result titles and URLs. Use fetch_webpage to read the contents of a specific result.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "num_pages": {"type": "integer", "description": "Number of search results to return (default: 3)."},
            },
            "required": ["query"]
        }
    }
}


def execute(arguments):
    query = arguments.get("query", "")
    num_pages = arguments.get("num_pages", 3)
    results = perform_web_search(query, num_pages=num_pages, fetch_content=False)
    output = []
    for r in results:
        if r:
            output.append({"title": r["title"], "url": r["url"]})

    return output if output else [{"error": "No results found."}]
