# ai_agent/memory_store.py

import os
from supabase import create_client, Client
from typing import List, Dict

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase credentials are not set in environment variables.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------------
# Conversation Memory
# -------------------------------

def load_conversation(session_id: str, limit: int = 5) -> List[Dict]:
    """
    Load recent conversation history for a session.
    Returns last N messages in chronological order.
    """
    response = (
        supabase.table("conversations")
        .select("role, content, created_at")
        .eq("session_id", session_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )

    messages = response.data or []

    # Reverse to chronological order
    messages = messages[::-1]

    formatted = []
    for msg in messages:
        if msg["role"] == "user":
            formatted.append({"user": msg["content"]})
        else:
            formatted.append({"assistant": msg["content"]})

    return formatted


def save_message(session_id: str, role: str, content: str):
    """
    Save a single message (user or assistant).
    """
    supabase.table("conversations").insert({
        "session_id": session_id,
        "role": role,
        "content": content
    }).execute()


# -------------------------------
# Structured State Memory
# -------------------------------

def load_state(session_id: str) -> Dict:
    """
    Load structured agent state.
    If none exists, return default state.
    """
    response = (
        supabase.table("agent_state")
        .select("*")
        .eq("session_id", session_id)
        .execute()
    )

    data = response.data

    if data:
        return data[0]

    # Default state
    return {
        "session_id": session_id,
        "filters": {},
        "last_intent": None,
        "last_action": None
    }


def save_state(session_id: str, state: Dict):
    """
    Upsert (insert or update) structured state.
    """
    supabase.table("agent_state").upsert({
        "session_id": session_id,
        "filters": state.get("filters", {}),
        "last_intent": state.get("last_intent"),
        "last_action": state.get("last_action")
    }).execute()