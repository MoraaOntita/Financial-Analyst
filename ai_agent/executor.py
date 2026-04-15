import asyncio
import inspect
from typing import Dict, Any, List
from .memory_store import load_state, save_state, save_message


class Executor:
    """
    Executor runs a provided plan sequentially, one step at a time.

    - Does NOT replan or call the LLM.
    - Accepts a tool mapping (tool name -> async callable).
    - Uses an optional tool_cache with `get` and `set` methods.
    - Persists completed steps and partial results to state via `save_state`.
    """

    def __init__(self, tool_mapping: Dict[str, Any], tool_cache=None):
        self.tool_mapping = tool_mapping
        self.tool_cache = tool_cache

    async def _call_tool(self, tool_name: str, params: Dict[str, Any]):
        if tool_name not in self.tool_mapping:
            raise ValueError(f"Tool not registered: {tool_name}")

        # check cache
        if self.tool_cache:
            cached = self.tool_cache.get(tool_name, params)
            if cached is not None:
                return cached

        func = self.tool_mapping[tool_name]
        # support both coroutine functions and regular callables
        if inspect.iscoroutinefunction(func):
            result = await func(**(params or {}))
        else:
            # run blocking call in threadpool
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: func(**(params or {})))

        if self.tool_cache:
            self.tool_cache.set(tool_name, params, result)

        return result

    async def execute_plan(self, plan_doc: Dict[str, Any], session_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
        plan = plan_doc.get("plan", []) if plan_doc else []
        completed_steps: List[Dict[str, Any]] = state.get("completed_steps", []) or []

        steps_log = []

        for step in plan:
            step_idx = step.get("step")
            tool = step.get("tool")
            purpose = step.get("purpose")
            params = step.get("parameters", {})

            # Normalize common wrapper structure: allow 'params' inside parameters
            if params and "params" in params and isinstance(params["params"], dict):
                params = params["params"]

            try:
                observation = await self._call_tool(tool, params)
            except Exception as e:
                observation = {"error": True, "message": str(e)}

            # Persist partial result to state
            completed_entry = {
                "step": step_idx,
                "tool": tool,
                "purpose": purpose,
                "parameters": params,
                "observation": observation,
            }

            completed_steps.append(completed_entry)
            steps_log.append(completed_entry)

            # update persistent state after each step
            state["completed_steps"] = completed_steps
            # store a lightweight preview to avoid storing huge datasets
            try:
                # if observation is list or dict, store small preview
                if isinstance(observation, list):
                    state.setdefault("partial_results", {})[f"step_{step_idx}"] = observation[:10]
                else:
                    state.setdefault("partial_results", {})[f"step_{step_idx}"] = observation
            except Exception:
                state.setdefault("partial_results", {})[f"step_{step_idx}"] = "<unserializable>"

            save_state(session_id, state)

        return {"completed_steps": steps_log, "state": state}
