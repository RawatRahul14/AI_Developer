# === Python Modules ===
from typing import TypedDict, Dict

# === Agent State ===
class AgentState(TypedDict):
    """
    State of the Agent that can be passed between calls.
    """
    ## === User Query ===
    user_query: str
    
    ## === Rephrased Question ===
    rephrased_question: str | None

    ## === History ===
    conversation: Dict[int, Dict[str, str]]

    ## === Tool Flag ===
    tool_flag: bool = False

    ## === Retrieved Documents ===
    documents: list | None
    proceed_to_generate: bool = False

    ## === Answer Generation ===
    generated_answer: str | None