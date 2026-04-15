import hashlib
import json
from typing import Any, Optional


class ToolCache:
    """Simple in-memory hash-based cache for tool results.

    Key is sha256 of tool name + sorted JSON parameters.
    """

    def __init__(self):
        self._store = {}

    @staticmethod
    def _make_key(tool_name: str, params: Any) -> str:
        params_json = json.dumps(params or {}, sort_keys=True, separators=(",", ":"))
        key_source = f"{tool_name}|{params_json}"
        return hashlib.sha256(key_source.encode()).hexdigest()

    def get(self, tool_name: str, params: Any) -> Optional[Any]:
        key = self._make_key(tool_name, params)
        return self._store.get(key)

    def set(self, tool_name: str, params: Any, value: Any):
        key = self._make_key(tool_name, params)
        self._store[key] = value
