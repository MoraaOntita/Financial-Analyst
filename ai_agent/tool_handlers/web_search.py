from tavily import TavilyClient
import os

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def search_web(query: str):
    response = tavily.search(query=query, max_results=5)

    cleaned_results = []

    for r in response.get("results", []):
        cleaned_results.append({
            "title": r.get("title"),
            "summary": r.get("content")[:200],  # shorten content
            "url": r.get("url")
        })

    return {
        "results": cleaned_results
    }