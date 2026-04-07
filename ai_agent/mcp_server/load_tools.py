import json

def load_tool_registry(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    return data["tools"]