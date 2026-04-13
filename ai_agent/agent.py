# ai_agent/agent.py
import asyncio
import json
import os
import re
from dotenv import load_dotenv
import httpx
from fastmcp import Client
from ai_agent.memory_store import load_conversation, save_message, load_state, save_state
import uuid

# -------------------------------
# Load environment variables
# -------------------------------
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path, override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MCP_SERVER_URL = "http://127.0.0.1:8000/mcp"

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not loaded. Check your .env file.")

print("DEBUG GROQ KEY:", GROQ_API_KEY[:10] if GROQ_API_KEY else None)

# -------------------------------
# Tool wrappers calling MCP server
# -------------------------------
async def query_tool(columns, distinct=False, filters=None, limit=50, order_by=None):
    """
    Calls the flexible query_used_cars_data tool with structured parameters.
    """
    async with Client(MCP_SERVER_URL) as client:
        payload = {
            "params": {
                "columns": columns,
                "distinct": distinct,
                "filters": filters or {},
                "limit": limit,
                "order_by": order_by
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
async def call_llm(prompt: str, retries=5) -> str:  # increase retries
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256  # reduce tokens to avoid large requests
    }

    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                wait_time = 2 ** attempt  # exponential backoff: 1,2,4,8,16s
                print(f"Rate limited by Groq API, retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                raise

        except httpx.RequestError as e:
            print(f"Network error: {e}, retrying in 2s...")
            await asyncio.sleep(2)
            
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


async def agent_loop(
    user_query: str,
    session_id: str,
    max_steps: int = 5,
    max_json_attempts: int = 3
):
    """
    Multi-step reasoning agent loop with persistent memory.
    - Automatically unwraps 'params' for tools
    - Handles multiple actions in a single user query
    - Stops looping if all actions are completed
    """
    history = load_conversation(session_id)
    state = load_state(session_id)
    steps = []

    save_message(session_id, "user", user_query)

    for step in range(max_steps):

        # inject last_observation
        last_observation = steps[-1]["observation"] if steps else None

        prompt = f"""
{REASONING_PROMPT}

User question: {user_query}

Conversation history:
{json.dumps(history, indent=2)}

Agent state:
{json.dumps(state, indent=2)}

Tool interaction history:
{json.dumps(steps, indent=2)}

Latest tool result:
{json.dumps(last_observation, indent=2) if last_observation else "None"}

IMPORTANT:
- The "observation" field contains the result of the previous tool call
- You MUST use ONLY this observation to decide the next step
- DO NOT use prior knowledge or assumptions

CRITICAL RULE:
- You MUST ONLY use data returned in tool observations
- If observation is empty or contains an error, respond with:
  "No matching records found"
- Never invent car names, prices, or statistics
"""

        if user_query.lower() in ["hi", "hello", "morning"]:
            save_message(session_id, "assistant", "Hello! How can I help you today?")
            return {"answer": "Hello! How can I help you today?", "steps": []}
        
        step_output = None
        for attempt in range(max_json_attempts):
            response_text = await call_llm(prompt)
            await asyncio.sleep(1)
            try:
                step_output = extract_json(response_text)
                break
            except Exception:
                prompt += "\n\nREMINDER: Respond ONLY in valid JSON."
                if attempt == max_json_attempts - 1:
                    return {"error": "Invalid JSON after multiple attempts", "raw": response_text}

        thought = step_output.get("thought")
        action = step_output.get("action")
        parameters = step_output.get("parameters", {})

        print(f"\n[Step {step+1}]")
        print("Thought:", thought)
        print("Action:", action)
        print("Params:", parameters)

        final_answer_text = step_output.get("final_answer", "")

        if action == "final_answer":
            if not final_answer_text:
                final_answer_text = "I couldn't compute a final answer based on available data."

            save_message(session_id, "assistant", final_answer_text)
            return {"answer": final_answer_text, "steps": steps}

        # unwrap 'params' for query/aggregate ---
        tool_args = parameters
        if action in ["query", "aggregate"] and "params" in parameters:
            tool_args = parameters["params"]

        if action not in tool_mapping:
            return {"error": f"Invalid action: {action}"}

        try:
            tool_result = await tool_mapping[action](**tool_args)
        except Exception as e:
            tool_result = {"error": True, "message": str(e)}

        # debug tool output
        print("Tool Result:", tool_result)

        tool_result = summarize_result(tool_result)

        # handle tool failure early
        if isinstance(tool_result, dict) and tool_result.get("error"):
            print("Tool Error:", tool_result)

            steps.append({
                "thought": thought,
                "action": action,
                "parameters": parameters,
                "observation": tool_result
            })

            # pass error as observation
            last_observation = tool_result

            continue

        steps.append({
            "thought": thought,
            "action": action,
            "parameters": parameters,
            "observation": tool_result
        })

        last_observation = steps[-1]["observation"] if steps else None

        if len(steps) >= 2:
            if (
                steps[-1]["action"] == steps[-2]["action"]
                and steps[-1]["parameters"] == steps[-2]["parameters"]
            ):
                return {"error": "Detected repeated identical tool call"}

        # save_message(session_id, "assistant", thought)

        # Update structured state
        state["last_action"] = action
        state["last_intent"] = thought
        save_state(session_id, state)

        # if tool produced final answer inside observation, break early ---
        if isinstance(tool_result, dict) and "final_answer" in tool_result:
            save_message(session_id, "assistant", tool_result["final_answer"])
            return {"answer": tool_result["final_answer"], "steps": steps}

    return {"answer": "Max steps reached without final answer.", "steps": steps}

# -------------------------------
# Terminal interactive loop
# -------------------------------
if __name__ == "__main__":
    async def main_loop():
        print("AI Agent ready! Type 'exit' to quit.")

        # In a real multi-user app, this session_id comes from the frontend/user auth
        session_id = str(uuid.uuid4())
        print(f"Session ID: {session_id}")

        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                break

            try:
                result = await agent_loop(user_input, session_id)
                print("\nFinal Answer:", result.get("answer"))

            except Exception as e:
                print("Error:", e)

    asyncio.run(main_loop())