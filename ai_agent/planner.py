import json
import os
import re
from typing import Dict, Any
from .llm import call_llm


def load_prompt() -> str:
    prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "reasoning_prompt.txt")
    if not os.path.exists(prompt_path):
        # fallback to workspace relative path
        prompt_path = os.path.join("ai_agent", "prompts", "reasoning_prompt.txt")
    with open(prompt_path, "r") as f:
        return f.read()


def extract_json(text: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found in LLM response")
    return json.loads(match.group())


class Planner:
    """
    Planner creates a minimal, dependency-aware execution plan given a user query and state.

    For the first incremental refactor this module returns a JSON plan produced by the LLM.
    Later this will be enhanced with deterministic decomposition logic.
    """

    @staticmethod
    async def create_plan(user_query: str, history: list, state: dict, max_steps: int = 5) -> Dict:
        prompt = load_prompt() + "\n\n"
        prompt += "You are a planner. Given the user question, conversation history, and agent state, produce a minimal JSON plan. "
        prompt += (
            "Return ONLY valid JSON with a top-level 'plan' array. Each step must include: 'step' (int), 'tool' (one of: query, aggregate, web_search), "
            "'purpose' (short string), and optional 'parameters' object. Keep plans minimal and avoid unnecessary steps.\n\n"
        )

        prompt += f"User question: {user_query}\n"
        prompt += f"Conversation history: {json.dumps(history)}\n"
        prompt += f"Agent state: {json.dumps(state)}\n"
        prompt += f"Max steps: {max_steps}\n"

        response_text = await call_llm(prompt)
        plan_json = extract_json(response_text)

        # Basic validation and normalization
        if "plan" not in plan_json or not isinstance(plan_json["plan"], list):
            raise ValueError("Planner returned invalid plan structure")

        return plan_json
