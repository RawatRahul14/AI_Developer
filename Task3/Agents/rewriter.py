# === Python Modules ===
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# === Agent State ===
from Task3.agent_state import AgentState

# === Schema ===
from Task3.schema.schemas import QueryRewrite


# === Main Agent Body ===
async def query_rewriter(
        state: AgentState
) -> AgentState:
    """
    Rewrites query and also tells whether to use Tool Node or Not.
    """

    ## === Rephrased Question ===
    state["rephrased_question"] = None

    ## === Conversation ===
    if "conversation" not in state:
        state["conversation"] = {}

    ## === Tool Flag ===
    state["tool_flag"] = False

    ## === Generated Answer ===
    state["generated_answer"] = None

    ## === Prompt ===
    prompt = """
    You are an intelligent query interpreter for a medical retrieval augmented system.
    Your job is to:
        1. Analyze the user question.
        2. Decide if the query needs to use code:
            - "tool" -> if the question needs filtering, counting, comparison, or listing  (e.g. "which patients", "how many", "most frequent", "list all", "find who")
        3. Rewrite the question in a clear and specific way for downstream nodes.
            - Expand pronouns or vague references using memory if provided.
            - Keep the meaning identical.

    You will be given:
        - The user's question
        - Optional memory context from the last few chats.
    ---

    Memory Context:
    {memory_context}

    User Question:
    {user_query}
    """.format(
        memory_context = state.get("conversation", {}),
        user_query = state.get("user_query")
    )

    ## === LLM Model Call ===
    model = ChatOpenAI(
        model = "gpt-4o-mini",
        temperature = 0
    ).with_structured_output(QueryRewrite)

    ## === Invoke the Model ===
    try:
        response = await model.ainvoke([
            SystemMessage(content = prompt),
            HumanMessage(content = "Return the result strictly following the JSON schema.")
        ])

        # === Update State ===
        state["rephrased_question"] = response.rephrased_question
        state["tool_flag"] = response.tool_flag

    except Exception as e:
        print(f"Query Rewriter Error: {e}")
        # Fallback: default to RAG mode with same query
        state["rephrased_question"] = state.get("user_query", "")
        state["tool_flag"] = False

    ## === Return Updated State ===
    return state