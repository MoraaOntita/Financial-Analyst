# ai_agent/agent.py
import asyncio
import json
import os
import re
from dotenv import load_dotenv
import httpx
from fastmcp import Client

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MCP_SERVER_URL = "http://127.0.0.1:8000/mcp"

# -------------------------------
# Tool wrappers calling MCP server
# -------------------------------
async def query_tool(columns, distinct=False, filters=None, limit=50):
    """
    Calls the flexible query_used_cars_data tool with structured parameters.
    """
    async with Client(MCP_SERVER_URL) as client:
        payload = {
            "params": {
                "columns": columns,
                "distinct": distinct,
                "filters": filters or {},
                "limit": limit
            }
        }
        result = await client.call_tool("query_used_cars_data", payload)
        return result.data

async def aggregation_tool(metric: str, column: str, group_by: str = None, filters: dict = None):
    async with Client(MCP_SERVER_URL) as client:
        result = await client.call_tool(
            "aggregate_used_car_data",
            {
                "params": {
                    "metric": metric,
                    "column": column,
                    "group_by": group_by,
                    "filters": filters
                }
            }
        )
        return result.data

async def web_search_tool(query: str):
    async with Client(MCP_SERVER_URL) as client:
        result = await client.call_tool("search_web", {"query": query})
        return result.data

tool_mapping = {
    "query": query_tool,
    "aggregate": aggregation_tool,
    "web_search": web_search_tool
}

# -------------------------------
# Call LLM via Groq API
# -------------------------------
async def call_llm(prompt: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 512
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

        return result["choices"][0]["message"]["content"]

# -------------------------------
# JSON extractor
# -------------------------------
def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found in LLM response")
    return json.loads(match.group())

# -------------------------------
# Load reasoning prompt from file
# -------------------------------
def load_prompt():
    prompt_path = os.path.join("ai_agent", "prompts", "reasoning_prompt.txt")
    with open(prompt_path, "r") as f:
        return f.read()

REASONING_PROMPT = load_prompt()



# -------------------------------
# Multi-step reasoning loop with JSON validation
# -------------------------------

def summarize_result(result, max_items=10):
    """
    Reduce large result sets to a small preview to avoid 413 errors.
    """
    if isinstance(result, list):
        return result[:max_items]
    if isinstance(result, dict):
        if "data" in result and isinstance(result["data"], list):
            return result["data"][:max_items]
    return result


async def agent_loop(user_query: str, max_steps: int = 3, max_json_attempts: int = 3):
    history = []

    for step in range(max_steps):
        prompt = f"""
{REASONING_PROMPT}

User question: {user_query}

Previous steps:
{json.dumps(history, indent=2)}
"""

        # Try to get valid JSON up to max_json_attempts times
        step_output = None
        for attempt in range(max_json_attempts):
            response_text = await call_llm(prompt)
            await asyncio.sleep(5)  # 1-second delay between calls to avoid rate limit
            try:
                step_output = extract_json(response_text)
                break  # success
            except Exception:
                # Add a feedback hint for the LLM
                prompt += "\n\nREMINDER: Respond ONLY in valid JSON as per instructions."
                if attempt == max_json_attempts - 1:
                    return {"error": "Invalid JSON from LLM after multiple attempts", "raw": response_text}

        thought = step_output.get("thought")
        action = step_output.get("action")
        parameters = step_output.get("parameters", {})

        print(f"\n[Step {step+1}]")
        print("Thought:", thought)
        print("Action:", action)
        print("Params:", parameters)

        # Final answer
        if action == "final_answer":
            return {
                "answer": step_output.get("final_answer"),
                "steps": history
            }
        
        

        # Tool execution
        if action not in tool_mapping:
            return {"error": f"Invalid action: {action}"}
        try:
            tool_result = await tool_mapping[action](**parameters)
        except Exception as e:
            tool_result = {"error": str(e)}

        history.append({
            "thought": thought,
            "action": action,
            "parameters": parameters,
            "observation": summarize_result(tool_result)
        })

    return {
        "answer": "Max steps reached without final answer.",
        "steps": history
    }

# -------------------------------
# Terminal interactive loop
# -------------------------------
if __name__ == "__main__":
    async def main_loop():
        print("AI Agent ready! Type 'exit' to quit.")
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            try:
                result = await agent_loop(user_input)

                print("\nFinal Answer:", result.get("answer"))

                # Optional: debug steps
                # print(json.dumps(result.get("steps"), indent=2))

            except Exception as e:
                print("Error:", e)

    asyncio.run(main_loop())