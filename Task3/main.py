# === Python Modules ===
import os
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# === LangGraph + Components ===
from Task3.graph import create_graph
from Task3.agent_state import AgentState
from Task3.components.retriever.faiss_retriever import FAISSRetriever

# === Load Environment ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# === Lifespan Manager ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initializes retriever and LangGraph on startup,
    releases resources gracefully on shutdown.
    """
    print("🚀 Starting Mini RAG Agent API...")

    # --- Load FAISS Retriever ---
    retriever_obj = FAISSRetriever()
    retriever = retriever_obj.load_index()

    # --- Initialize LangGraph ---
    graph = create_graph()

    # --- Store globally ---
    app.state.graph = graph
    app.state.retriever = retriever

    print("✅ Graph and Retriever initialized successfully.")
    yield
    print("🛑 Shutting down Mini RAG Agent API...")


# === FastAPI App ===
app = FastAPI(
    title = "Mini RAG Agent API",
    version = "1.0.0",
    lifespan = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

# === Request Model ===
class QueryRequest(BaseModel):
    unique_id: str
    query: str

# === Root ===
@app.get("/")
def root():
    return {"message": "Mini RAG Agent API is live 🚀"}

# === Ask Endpoint ===
@app.post("/generate")
async def ask_query(payload: QueryRequest):
    """
    Executes LangGraph pipeline using user UUID (thread_id) and query.
    Returns only the generated answer.
    """
    try:
        graph = app.state.graph
        retriever = app.state.retriever

        # === Build initial agent state ===
        state = {
            "user_query": payload.query
        }

        # === Run LangGraph ===
        result = await graph.ainvoke(
            state,
            config = {
                "configurable": {
                    "retriever": retriever,
                    "thread_id": payload.unique_id
                }
            }
        )

        # === Extract Only the Final Answer ===
        answer = result.get("generated_answer", "No answer generated.")

        return {"answer": answer}

    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail = str(e)
        )