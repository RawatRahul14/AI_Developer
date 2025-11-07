# === Agent State ===
import asyncio
from Task3.agent_state import AgentState

# === Main Retriever Agent ===
async def doc_retriever(
        state: AgentState,
        config
) -> AgentState:
    """
    Main function to retrieve documents based on the current state and configuration.
    Args:
        state (AgentState): The current state of the agent.
        config: Configuration parameters for the retrieval process.

    Returns:
        AgentState: Updated state after document retrieval.
    """
    ## === Getting the retriever and query ===
    retriever = config.get("configurable", {}).get("retriever", None)
    query = state.get("rephrased_question", "")

    # === Retrieving documents ===
    documents = await asyncio.to_thread(retriever.invoke, query)

    state["documents"] = documents

    return state