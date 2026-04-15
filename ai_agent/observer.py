import asyncio
from typing import Any, Dict, Optional
from .memory_store import load_conversation, save_message, load_state, save_state


def _preview(observation: Any, max_items: int = 10):
    try:
        if isinstance(observation, list):
            return observation[:max_items]
        if isinstance(observation, dict):
            # if contains 'data' list, preview it
            if "data" in observation and isinstance(observation["data"], list):
                return observation["data"][:max_items]
            return observation
    except Exception:
        return "<unserializable>"
    return observation


class Observer:
    """
    Observer evaluates tool results, updates memory/state, and determines whether replanning is needed.

    Methods are async to allow optional planner invocation for replanning.
    """

    @staticmethod
    async def observe_step(
        session_id: str,
        step: Dict[str, Any],
        observation: Any,
        state: Dict[str, Any],
        history: Optional[list] = None,
        planner=None,
        user_query: Optional[str] = None,
        max_steps: int = 5,
    ) -> Dict[str, Any]:
        """
        Assess a single step observation and update persistent state.

        Returns a dict with keys: 'complete', 'error', 'needs_replan', 'replan' (optional plan)
        """

        error = False
        needs_replan = False
        complete = False

        # detect explicit error
        if isinstance(observation, dict) and observation.get("error"):
            error = True
            needs_replan = True

        # detect empty results
        if observation is None:
            needs_replan = True

        if isinstance(observation, (list, dict)):
            # empty collection
            try:
                if isinstance(observation, list) and len(observation) == 0:
                    needs_replan = True
                if isinstance(observation, dict) and len(observation) == 0:
                    needs_replan = True
            except Exception:
                pass

        # completion detection: tool returned a final answer
        if isinstance(observation, dict) and "final_answer" in observation:
            complete = True
            final_text = observation.get("final_answer")
            if final_text:
                save_message(session_id, "assistant", final_text)

        # update structured state with preview and last_observation
        state.setdefault("last_observation", {})
        state["last_observation"][f"step_{step.get('step')}"] = _preview(observation)

        # propagate filters if present in parameters (helps chaining)
        try:
            params = step.get("parameters", {}) or {}
            if isinstance(params, dict) and "filters" in params:
                # merge filters shallowly
                state.setdefault("filters", {})
                state["filters"].update(params.get("filters") or {})
        except Exception:
            pass

        save_state(session_id, state)

        replan_doc = None
        if needs_replan and planner is not None and user_query is not None:
            # call planner to get a revised plan
            try:
                conv_history = history if history is not None else load_conversation(session_id)
                replan_doc = await planner.create_plan(user_query, conv_history, state, max_steps=max_steps)
            except Exception:
                replan_doc = None

        return {
            "complete": complete,
            "error": error,
            "needs_replan": needs_replan,
            "replan": replan_doc,
        }
