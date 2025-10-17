from typing import TypedDict, Any, Dict, List

class AgentState(TypedDict, total=False):
    input: str
    target_length: str
    style: str
    n: int
    mode: str
    plan: List[Dict[str, Any]]
    context: List[Dict[str, Any]]
    draft: str
    final: str
    deps: Dict[str, Any]

    