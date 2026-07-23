"""
Planner utilities for JSON extraction and prompt loading.

Helper functions for loading reasoning prompts and extracting JSON
from LLM responses.
"""

import json
import os
import re
from typing import Dict, Any


def load_prompt() -> str:
    """
    Load the reasoning prompt from the prompts directory.
    
    Tries absolute path first, then falls back to workspace-relative path
    for flexibility in different execution contexts.
    
    Returns:
        Reasoning prompt text
    
    Raises:
        FileNotFoundError: If prompt file cannot be found
    """
    prompt_path = os.path.join(os.path.dirname(__file__), "..", "..", "prompts", "reasoning_prompt.txt")
    if not os.path.exists(prompt_path):
        # fallback to workspace relative path
        prompt_path = os.path.join("ai_agent", "prompts", "reasoning_prompt.txt")
    with open(prompt_path, "r") as f:
        return f.read()


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()

    # Remove Markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)

    if not match:
        raise ValueError(
            f"No JSON object found.\nLLM response:\n{text}"
        )

    return json.loads(match.group(0))