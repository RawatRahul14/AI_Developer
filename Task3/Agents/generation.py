# === Python Modules ===
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# === Agent State ===
from Task3.agent_state import AgentState

# === Schema ===
from Task3.schema.schemas import AnswerGeneration

# === Utils ===
from Task3.utils import update_recent_chats

# === Answer Generation Agent ===
async def answer_generation(
        state: AgentState
) -> AgentState:
    """
    Generates the final answer based on the rephrased question and relevant documents.

    Args:
        state (AgentState): The current state of the agent containing the rephrased question and documents.

    Returns:
        AgentState: The updated state with the generated answer.
    """
    ## === Render Prompt ===
    prompt = """
    ROLE
        - You are an expert AI answer generator for a medical Retrieval-Augmented Generation system.
        - Your job is to create accurate, clear, and context-grounded answers based on retrieved documents.

    TASK
        - Generate outputs:
            1. **answer**: A well-written, human-friendly response to display to the user.

    RULES
        - Base your answer **only** on the given documents; do not hallucinate missing details.
        - Use neutral, professional language.
        - Avoid repetition and unnecessary elaboration.

    USER QUESTION:
    {user_query}

    RELEVANT DOCUMENTS:
    {documents}
    """.format(
        user_query = state.get("rephrased_question"),
        documents = state.get("documents")
    )

    # === Initialize Language Model ===
    llm = ChatOpenAI(
        model_name = "gpt-4o-mini",
        temperature = 0.0
    ).with_structured_output(AnswerGeneration)

    message = [
        SystemMessage(content = prompt),
        HumanMessage(content = "Return the result strictly following the JSON schema.")
    ]

    response = await llm.ainvoke(
        message
    )

    ## === Outputs ===
    state["generated_answer"] = response.answer
    state["conversation"] = update_recent_chats(
        recent_chats = state.get("messages", []),
        latest_question = state.get("rephrased_question"),
        answer = response.answer
    )

    return state