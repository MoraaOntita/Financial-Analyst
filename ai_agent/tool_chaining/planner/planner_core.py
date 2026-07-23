"""
Planner module for creating execution plans.

Generates minimal DAG-based execution plans for multi-step tool chains
using an LLM to reason about dependencies and tool choices.
"""

import json
import re
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
    def validate_plan_payload(
        plan_doc: Optional[Dict[str, Any]],
        allowed_tools: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Validate the parsed plan shape and ensure tools are registered."""
        if not isinstance(plan_doc, dict):
            raise ValueError(f"Planner returned {type(plan_doc).__name__}, expected dict.")

        if "plan" not in plan_doc:
            raise ValueError(f"Planner JSON missing 'plan': {plan_doc}")

        if not isinstance(plan_doc["plan"], list):
            raise ValueError(f"'plan' should be a list, got {type(plan_doc['plan'])}")

        allowed = {tool for tool in (allowed_tools or []) if isinstance(tool, str) and tool.strip()}
        if not allowed:
            allowed = {"query", "aggregate", "web_search"}

        validated_plan: List[Dict[str, Any]] = []

        for step in plan_doc["plan"]:
            if not isinstance(step, dict):
                raise ValueError("Each plan step must be an object.")

            if not isinstance(step.get("step"), int):
                raise ValueError("Each plan step must include an integer 'step'.")

            tool_name = step.get("tool")
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise ValueError(f"Step {step.get('step')} is missing a valid tool name.")

            if tool_name not in allowed:
                raise ValueError(
                    f"Step {step.get('step')} uses unregistered tool '{tool_name}'. "
                    f"Allowed tools: {sorted(allowed)}"
                )

            purpose = step.get("purpose")
            if not isinstance(purpose, str) or not purpose.strip():
                raise ValueError(f"Step {step.get('step')} is missing a valid purpose.")

            parameters = step.get("parameters", {}) or {}
            if not isinstance(parameters, dict):
                raise ValueError(f"Step {step.get('step')} has invalid parameters; expected an object.")

            depends_on = step.get("depends_on", []) or []
            if not isinstance(depends_on, list) or any(not isinstance(dep, int) for dep in depends_on):
                raise ValueError(f"Step {step.get('step')} has invalid depends_on; expected a list of step numbers.")

            step_number = step.get("step")
            parameter_text = json.dumps(parameters, default=str)
            referenced_dependencies = {
                dep for dep in depends_on if dep > 0 and dep < step_number and re.search(rf"step_{dep}\.output", parameter_text)
            }
            cleaned_dependencies = sorted(referenced_dependencies)

            normalized_step = dict(step)
            normalized_step.setdefault("depends_on", [])
            normalized_step["parameters"] = parameters
            normalized_step["depends_on"] = cleaned_dependencies
            validated_plan.append(normalized_step)

        return {"plan": validated_plan}

    @staticmethod
    async def create_plan(
        user_query: str,
        history: List[Dict[str, Any]],
        state: Dict[str, Any],
        max_steps: int = 5,
        allowed_tools: Optional[List[str]] = None,
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
        prompt += "You are a planner, not an answer generator. Given the user question, conversation history, and agent state, produce a minimal JSON plan for tool execution only. "
        prompt += (
            "Return ONLY valid JSON with a top-level 'plan' array.\n\n"
            "Each step must include:\n"
            "- step (int)\n"
            "- tool (one of: query, aggregate, web_search)\n"
            "- purpose (short string)\n"
            "- optional parameters (object)\n"
            "- optional depends_on (array of step numbers this step depends on)\n\n"
            "CRITICAL RULES:\n"
            "- Do NOT generate answer-only steps such as final_answer or response steps.\n"
            "- Do NOT create tool names outside the allowed list.\n"
            "- Use query for row retrieval and matching records.\n"
            "- Use aggregate for averages, counts, minimums, maximums, totals, or grouping.\n"
            "- Use web_search only for external explanations or information not available in the dataset.\n"
            "- Only add depends_on when a later step actually uses the output of an earlier step. If a step is independent, leave depends_on as [].\n"
            "- If using previous results, reference them as: step_<n>.output.\n"
            "- Keep plans minimal and logically chained.\n"
            "- If the user request is conversational or does not require any tool, return an empty plan: {'plan': []}.\n\n"
            "FEW-SHOT EXAMPLES:\n"
            "Example 1:\n"
            "User: Show me the average selling price by fuel type\n"
            "Output: {\"plan\": [{\"step\": 1, \"tool\": \"aggregate\", \"purpose\": \"Compute average selling price by fuel type\", \"parameters\": {\"metric\": \"avg\", \"column\": \"selling_price\", \"group_by\": \"fuel_type\"}, \"depends_on\": []}]}\n\n"
            "Example 2:\n"
            "User: Compare diesel vs petrol average prices\n"
            "Output: {\"plan\": [{\"step\": 1, \"tool\": \"aggregate\", \"purpose\": \"Compute average selling price for diesel cars\", \"parameters\": {\"metric\": \"avg\", \"column\": \"selling_price\", \"filters\": {\"fuel_type\": \"diesel\"}}, \"depends_on\": []}, {\"step\": 2, \"tool\": \"aggregate\", \"purpose\": \"Compute average selling price for petrol cars\", \"parameters\": {\"metric\": \"avg\", \"column\": \"selling_price\", \"filters\": {\"fuel_type\": \"petrol\"}}, \"depends_on\": []}]}\n\n"
            "Example 3:\n"
            "User: Good morning, how are you?\n"
            "Output: {\"plan\": []}\n\n"
            "Example 4:\n"
            "User: List some cars from the dataset\n"
            "Output: {\"plan\": [{\"step\": 1, \"tool\": \"query\", \"purpose\": \"Retrieve a sample of used cars\", \"parameters\": {\"columns\": [\"car_name\", \"selling_price\"], \"limit\": 10}, \"depends_on\": []}]}\n"
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

        plan_json = Planner.validate_plan_payload(plan_json, allowed_tools=allowed_tools)

        return plan_json
