"""
Natural language summarization of tool outputs.

Converts raw tool results into human-readable text summaries for different
tool types (aggregation, query, web search).
"""

from typing import Any, Dict, List


def summarize_to_text(tool_name: str, purpose: str, observation: Dict[str, Any]) -> str:
    """
    Convert step observation into readable natural language summary.
    
    Routes to specialized summarizers based on tool type and dispatches
    appropriate formatting logic.
    
    Args:
        tool_name: Name of the tool (query, aggregate, web_search)
        purpose: Purpose of the step from the plan
        observation: Normalized observation from tool execution
    
    Returns:
        Human-readable text summary of the step result
    """
    if not observation or observation.get("error"):
        error_msg = observation.get("message", "Unknown error") if observation else "No output"
        return f"Error executing step: {error_msg}"

    final_answer = observation.get("final_answer", [])
    result_type = observation.get("type", tool_name)

    # Handle aggregation results
    if tool_name == "aggregate" or result_type == "aggregation":
        return summarize_aggregation(purpose, final_answer)

    # Handle query results
    if tool_name == "query" or result_type == "record_set" or result_type == "records":
        return summarize_query(purpose, final_answer)

    # Handle web search results
    if tool_name == "web_search" or result_type == "search_results":
        return summarize_search(purpose, final_answer)

    # Generic fallback
    count = len(final_answer) if isinstance(final_answer, list) else 1
    return f"{purpose}: Found {count} result(s)."


def summarize_aggregation(purpose: str, results: List[Any]) -> str:
    """
    Summarize aggregation results into readable text.
    
    Handles count/sum/avg/min/max aggregations with proper number formatting.
    Provides group summaries for multiple results.
    
    Args:
        purpose: Purpose of the aggregation from the plan
        results: List of aggregation results
    
    Returns:
        Formatted text summary
    """
    if not results:
        return f"{purpose}: No results found."

    if len(results) == 1:
        item = results[0]
        if isinstance(item, dict):
            # Build summary from dict keys/values
            pairs = []
            for key, value in item.items():
                if isinstance(value, (int, float)):
                    value = f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
                pairs.append(f"{key} = {value}")
            return f"{purpose}: {', '.join(pairs)}."
        else:
            return f"{purpose}: {str(item)}"

    # Multiple results - summarize with counts
    if isinstance(results[0], dict):
        # Check for common key patterns
        keys = results[0].keys()
        
        # Try to identify what's being counted or aggregated
        if "count" in keys:
            total = sum(r.get("count", 0) for r in results if isinstance(r, dict))
            return f"{purpose}: Total count is {total:,} across {len(results)} groups."
        elif "selling_price" in keys or "price" in keys or "avg" in keys:
            return f"{purpose}: Analyzed {len(results)} items with various metrics."
        else:
            return f"{purpose}: Aggregated {len(results)} distinct items."
    
    return f"{purpose}: Aggregated data with {len(results)} results."


def summarize_query(purpose: str, results: List[Any]) -> str:
    """
    Summarize query results into readable text while preserving the most relevant
    row-level details instead of collapsing everything to a generic placeholder.
    """
    if not results:
        return f"{purpose}: No records found matching the criteria."

    if len(results) == 1:
        item = results[0]
        if isinstance(item, dict):
            name = item.get("car_name") or item.get("name") or "Record"
            year = item.get("year")
            price = item.get("selling_price") or item.get("price")
            fuel = item.get("fuel_type")
            transmission = item.get("transmission")

            details = [name]
            if year:
                details.append(f"year {year}")
            if price is not None:
                details.append(f"price {price}")
            if fuel:
                details.append(f"fuel {fuel}")
            if transmission:
                details.append(f"transmission {transmission}")

            return f"{purpose}: Found {' '.join(details)}."
        return f"{purpose}: Retrieved one record."

    if isinstance(results[0], dict):
        preview_items = []
        for item in results[:3]:
            name = item.get("car_name") or item.get("name") or "Unknown"
            price = item.get("selling_price") or item.get("price")
            if price is not None:
                preview_items.append(f"{name} ({price})")
            else:
                preview_items.append(name)

        shown = ", ".join(preview_items)
        remaining = f" and {len(results) - 3} more" if len(results) > 3 else ""
        return f"{purpose}: Found {len(results)} records{remaining}. Top results: {shown}."

    return f"{purpose}: Retrieved {len(results)} records."


def summarize_search(purpose: str, results: List[Any]) -> str:
    """
    Summarize web search results into readable text.
    
    Extracts titles and summaries from search results with truncation
    for large result sets.
    
    Args:
        purpose: Purpose of the search from the plan
        results: List of search results
    
    Returns:
        Formatted text summary
    """
    if not results:
        return f"{purpose}: No search results found."

    if len(results) == 1:
        item = results[0]
        if isinstance(item, dict):
            title = item.get("title", "Result")
            summary = item.get("summary", "")[:100]  # First 100 chars
            return f"{purpose}: {title}. {summary}..."
        else:
            return f"{purpose}: {str(item)[:150]}"

    # Multiple results
    titles = []
    for item in results[:3]:
        if isinstance(item, dict):
            title = item.get("title", "Result")
            titles.append(title)
    
    shown = ", ".join(titles)
    remaining = f" and {len(results) - 3} more" if len(results) > 3 else ""
    return f"{purpose}: Found {len(results)} sources{remaining}. Top results: {shown}."


def transform_output(tool_name: str, raw_output: Any) -> Dict[str, Any]:
    """
    Normalize any tool output into a standard structure for chaining.
    
    Ensures all tool outputs have consistent structure for downstream
    processing and dependency resolution.
    
    Args:
        tool_name: Name of the tool that produced this output
        raw_output: Raw output from the tool
    
    Returns:
        Normalized structure: {"type": ..., "result": {...}, "final_answer": [...]}
    """
    normalized = {
        "type": tool_name,
        "result": {"results": []},
        "final_answer": []
    }

    if isinstance(raw_output, dict):
        if tool_name == "query" and "rows" in raw_output:
            rows = raw_output.get("rows") or []
            if isinstance(rows, list):
                normalized["result"]["results"] = rows
                normalized["final_answer"] = rows
                return normalized

        # If aggregation returns a list of dicts under "results" or "data"
        if "results" in raw_output and isinstance(raw_output["results"], list):
            normalized["result"]["results"] = raw_output["results"]
        elif "data" in raw_output and isinstance(raw_output["data"], list):
            normalized["result"]["results"] = raw_output["data"]
        else:
            # fallback: wrap dict in list
            normalized["result"]["results"] = [raw_output]

        # For convenience, final_answer duplicates results
        normalized["final_answer"] = normalized["result"]["results"]

    elif isinstance(raw_output, list):
        normalized["result"]["results"] = raw_output
        normalized["final_answer"] = raw_output

    else:
        # scalar: wrap in a list
        normalized["result"]["results"] = [raw_output]
        normalized["final_answer"] = [raw_output]

    return normalized
