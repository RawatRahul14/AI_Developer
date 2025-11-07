# === Agent State ===
from Task3.agent_state import AgentState

## === No Document Retrieved ===
def no_relevant_docs(
        state: AgentState
) -> str:
    """
    Route when no relevant documents are retrieved.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        str: The next action to take.
    """
    proceed = state.get("proceed_to_generate", False)

    if proceed:
        return "generate_answer"
    else:
        return "fallback"