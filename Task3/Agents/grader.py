# === Python Modules ===
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# === Agent State ===
from Task3.agent_state import AgentState

# === Schema ===
from Task3.schema.schemas import DocGrader

# === Main Agent Body ===
async def doc_grader(
        state: AgentState
) -> AgentState:
    """
    Grades the relevance of retrieved documents to the user's query.

    Args:
        state (AgentState): The current state of the agent, including the user's question and retrieved documents.

    Returns:
        AgentState: The updated state with graded documents.
    """
    relavant_docs: list = []
    for document in state.get("documents"):
        ## === Prompt ===
        prompt = """
        Evaluate whether a retrieved document is relevant to the user's question.
        The grader determines if the document helps answer the question or not.

        ROLE
            - You are a retrieval grader that assesses the relevance of a document to a user's question.

        INPUTS
            - "question": The user's query.
            - "document": The retrieved document content.

        TASK
            - Determine if the document contains information that directly answers or helps answer the question.

        RULES
            - If the document is relevant, respond **"Yes"**.
            - If not, respond **"No"**.
            - Respond with exactly one word: **"Yes"** or **"No"**.
            - Do not include any explanations or reasoning.

        USER QUESTION:
            {question}

        DOCUMENT CONTENT:
            {document}
        """.format(
            question = state.get("rephrased_question"),
            document = document
        )

        ## === LLM ===
        llm = ChatOpenAI(
            model = "gpt-4o-mini",
            temperature = 0.0
        ).with_structured_output(DocGrader)

        messages = [
            SystemMessage(content = prompt),
            HumanMessage(content = "Return the result strictly following the JSON schema.")
        ]

        response = await llm.ainvoke(
            messages
        )

        if response.score.strip().lower() == "yes":
            relavant_docs.append(document)
    
    ## === Output ===
    state["documents"] = relavant_docs
    state["proceed_to_generate"] = len(relavant_docs) > 0

    return state