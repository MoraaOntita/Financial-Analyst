"""
Planner module for creating execution plans.

Generates minimal DAG-based execution plans for multi-step tool chains
using an LLM to reason about dependencies and tool choices.
"""

import json
from typing import Dict, Any, List, Optional

from ...llm import call_llm
from .utils import load_prompt, extract_json


class Planner:
    """
    Planner creates minimal, dependency-aware execution plans.
    
    Given a user query, conversation history, and agent state, the Planner
    uses an LLM to decompose the query into steps that invoke appropriate tools
    and declare dependencies between steps for tool chaining.
    
    Each step includes:
    - step (int): Unique step identifier
    - tool (str): One of registered tools (query, aggregate, web_search)
    - purpose (str): Description of what this step accomplishes
    - parameters (dict): Tool-specific parameters
    - depends_on (list): IDs of steps this step depends on (default: [])
    
    Example:
        plan_doc = await Planner.create_plan(
            user_query="Get top 5 cars by price",
            history=conversation_history,
            state=agent_state
        )
        print(plan_doc)
        # {
        #   "plan": [
        #     {
        #       "step": 1,
        #       "tool": "query",
        #       "purpose": "Get top 5 cars by price",
        #       "parameters": {"columns": [...], "limit": 5, "order_by": "selling_price DESC"},
        #       "depends_on": []
        #     }
        #   ]
        # }
    """

    @staticmethod
    async def create_plan(
        user_query: str,
        history: List[Dict[str, Any]],
        state: Dict[str, Any],
        max_steps: int = 5
    ) -> Dict[str, Any]:
        """
        Create an execution plan for a user query.
        
        Uses LLM reasoning with the provided context to generate a minimal,
        dependency-aware plan for multi-step tool execution.
        
        Args:
            user_query: The user's natural language query
            history: Conversation history (list of user/assistant messages)
            state: Current agent state with any previous results
            max_steps: Maximum number of steps to plan (default 5)
        
        Returns:
            Plan document: {"plan": [step1, step2, ...]}
        
        Raises:
            ValueError: If planner response is invalid JSON or malformed plan
        """
        print(">>> ENTERED Planner.create_plan")
        prompt = load_prompt() + "\n\n"
        prompt += "You are a planner. Given the user question, conversation history, and agent state, produce a minimal JSON plan. "
        prompt += (
            "Return ONLY valid JSON with a top-level 'plan' array.\n\n"
            "Each step must include:\n"
            "- step (int)\n"
            "- tool (one of: query, aggregate, web_search)\n"
            "- purpose (short string)\n"
            "- optional parameters (object)\n"
            "- optional depends_on (array of step numbers this step depends on)\n\n"
            "CRITICAL RULES:\n"
            "- If a step uses results from a previous step, it MUST declare depends_on\n"
            "- If using previous results, reference them as: step_<n>.output\n"
            "- Keep plans minimal and logically chained\n"
        )

        prompt += f"User question: {user_query}\n"
        prompt += f"Conversation history: {json.dumps(history)}\n"
        prompt += f"Agent state: {json.dumps(state)}\n"
        prompt += f"Max steps: {max_steps}\n"

        print(">>> Calling LLM...")

        response_text = await call_llm(prompt)

        print("\n=== RAW PLANNER RESPONSE ===")
        print(response_text)
        print("============================\n")

        plan_json = extract_json(response_text)

        print(">>> Parsed JSON:")

        if not isinstance(plan_json, dict):
            raise ValueError(
                f"Planner returned {type(plan_json).__name__}, expected dict."
            )

        if "plan" not in plan_json:
            raise ValueError(
                f"Planner JSON missing 'plan': {plan_json}"
            )

        if not isinstance(plan_json["plan"], list):
            raise ValueError(
                f"'plan' should be a list, got {type(plan_json['plan'])}"
            )

        for step in plan_json["plan"]:
            step.setdefault("depends_on", [])

        return plan_json
