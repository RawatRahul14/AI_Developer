# === Python Modules ===
import os
from langgraph.checkpoint.mongodb import AsyncMongoDBSaver
from pymongo import AsyncMongoClient
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

# === Import Agents ===
from Task3.Agents.rewriter import query_rewriter
from Task3.Agents.retriever import doc_retriever
from Task3.Agents.grader import doc_grader
from Task3.Agents.generation import answer_generation
from Task3.Agents.fallback import fallback_agent

# === Import Router and State ===
from Task3.router.routes import no_relevant_docs
from Task3.agent_state import AgentState

# === Load Environment Variables ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")


# === Graph Flow ===
def create_graph() -> StateGraph:
    """
    Creates a LangGraph-based flow for the Mini RAG Agent.

    Node sequence:
        1. query_rewriter     -> rewrites & classifies the user query.
        2. doc_retriever      -> retrieves relevant documents from FAISS.
        3. doc_grader         -> checks document relevance.
        4. answer_generation  -> generates the final answer if docs are relevant.
        5. fallback_agent     -> used if docs are not relevant.
    """

    ## === Async MongoDb Connection for Maintaining the state ===
    mongo_client = AsyncMongoClient(MONGODB_URI)
    checkpointer = AsyncMongoDBSaver(
        client = mongo_client,
        db_name = DB_NAME,
        checkpoint_collection_name = COLLECTION_NAME
    )

    # === Initialize the Graph ===
    graph = StateGraph(AgentState)

    # === Add Nodes ===
    graph.add_node(
        "query_rewriter",
        RunnableLambda(query_rewriter).with_config(
            {
                "run_async": True
            }
        )
    )
    graph.add_node(
        "doc_retriever",
        RunnableLambda(doc_retriever).with_config(
            {
                "run_async": True
            }
        )
    )
    graph.add_node(
        "doc_grader",
        RunnableLambda(doc_grader).with_config(
            {
                "run_async": True
            }
        )
    )
    graph.add_node(
        "answer_generation",
        RunnableLambda(answer_generation).with_config(
            {
                "run_async": True
            }
        )
    )
    graph.add_node(
        "fallback_agent",
        RunnableLambda(fallback_agent).with_config(
            {
                "run_async": True
            }
        )
    )

    # === Define Conditional Routing ===
    # Router: decides if relevant docs exist or fallback is needed
    graph.add_conditional_edges(
        "doc_grader",
        no_relevant_docs,
        {
            "generate_answer": "answer_generation",
            "fallback": "fallback_agent"
        }
    )

    # === Define Edges ===
    graph.add_edge(START, "query_rewriter")
    graph.add_edge("query_rewriter", "doc_retriever")
    graph.add_edge("doc_retriever", "doc_grader")
    graph.add_edge("answer_generation", END)
    graph.add_edge("fallback_agent", END)

    # === Compile and Return Graph ===
    return graph.compile(
        checkpointer = checkpointer
    )