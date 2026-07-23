# ai_agent/agent.py
import asyncio
import json
import os
import re
from dotenv import load_dotenv
import httpx
from fastmcp import Client
from ai_agent.memory_store import load_conversation, save_message, load_state, save_state
from ai_agent.tool_chaining.executor import Executor
from ai_agent.tool_cache import ToolCache
from ai_agent.llm import call_llm
from ai_agent.tool_chaining.planner import Planner
import uuid
import traceback

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

# call_llm is provided by ai_agent.llm (kept as separate module for reuse)


def is_conversational_query(user_query: str) -> bool:
    """Return True for simple conversational prompts that do not need tool use."""
    text = (user_query or "").strip().lower()
    if not text:
        return False

    conversational_phrases = [
        "good morning",
        "good afternoon",
        "good evening",
        "hello",
        "hi",
        "hey",
        "how are you",
        "how are you doing",
        "how's it going",
        "thanks",
        "thank you",
        "what's up",
        "nice to meet you",
        "bye",
        "goodbye",
    ]

    return any(phrase in text for phrase in conversational_phrases)


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

    if is_conversational_query(user_query):
        prompt = (
            "You are a helpful assistant. Respond naturally and briefly to the user's message.\n"
            f"User message: {user_query}"
        )
        final_answer = await call_llm(prompt)
        save_message(session_id, "assistant", final_answer)
        return {"answer": final_answer, "steps": []}

    # --- Planner: create an initial execution plan
    try:
        print("\n>>> Calling Planner.create_plan()")

        plan_doc = await Planner.create_plan(
            user_query,
            history,
            state,
            allowed_tools=list(tool_mapping.keys()),
        )

        print("\n>>> Planner returned:")
        print(json.dumps(plan_doc, indent=2))

    except Exception as e:
        print("\n" + "=" * 80)
        print("PLANNER FAILED")
        print("=" * 80)

        traceback.print_exc()

        print("\nException type:", type(e).__name__)
        print("Exception:", repr(e))

        return {
            "error": "Planner failed",
            "details": str(e)
        }


    # --- Executor: run planned steps deterministically
    tool_cache = ToolCache()
    executor = Executor(tool_mapping, tool_cache=tool_cache)

    try:
        exec_result = await executor.execute_plan(plan_doc, session_id, state)
        print("[Executor] Completed steps:", json.dumps(exec_result.get("completed_steps", []), indent=2))
    except Exception as e:
        print(f"[Executor] Execution failed: {e}")
        return {"error": "Executor failed", "details": str(e)}

    # -------------------------------------------------------
    # unified result extraction (no ReAct loop anymore)
    # -------------------------------------------------------

    completed_steps = exec_result.get("completed_steps", [])

    # Try to find final answer from tool outputs (if any tool provides it)
    final_answer = None

    for step in completed_steps:
        obs = step.get("observation")

        if isinstance(obs, dict) and "final_answer" in obs:
            final_answer = obs["final_answer"]
            break

    # fallback behavior
    if not final_answer:
        final_answer = "Execution completed. No explicit final answer was produced."

    save_message(session_id, "assistant", final_answer)

    return {
        "answer": final_answer,
        "steps": completed_steps
    }

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