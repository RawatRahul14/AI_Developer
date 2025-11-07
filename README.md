# 🧠 MedRAG - From Medical Intelligence to RAG Automation

Transforming unstructured medical data into structured, queryable intelligence through AWS, LangGraph, and OpenAI.
Built as a complete, end-to-end system that demonstrates mastery in cloud-based NLP, multimodal data processing, and agentic AI workflows.

## 🧭 Project Overview

MedRAG is a full-stack medical data intelligence pipeline that turns raw clinical images into structured, searchable knowledge.
By integrating AWS Textract, Comprehend Medical, and a LangGraph-powered RAG agent, it creates a seamless flow from OCR extraction -> entity recognition -> conversational intelligence.

| Task  | Description |
|-------|--------------|
| **Task 1** | Extract and process medical entities using **AWS Textract** + **Comprehend Medical**. |
| **Task 2** | Summarize structured data into clinical notes using **LangChain** + **GPT**. |
| **Task 3** | Deploy a **Mini RAG Agent** (**FastAPI** + **LangGraph** + **Streamlit**) for conversational queries. |

## ⚡ Key Features

- 🧠 Multi-Stage AI Workflow: From raw medical images to structured knowledge — integrates AWS Textract, Comprehend Medical, and LangChain in one pipeline.
- 🧬 Automated Entity Extraction: Uses AWS Textract for OCR and Comprehend Medical to detect medical conditions, treatments, and tests.
- 🩺 LLM-Based Summarization: Converts extracted medical data into clean, validated structured notes using OpenAI’s GPT models.
- 📚 Retrieval-Augmented Generation (RAG): FAISS vector search combined with LangGraph agents enables accurate, document-grounded answers.
- 🔁 Memory & Checkpointing: MongoDB asynchronous checkpointing preserves chat context and conversation state.
- 💬 Streamlit Chat UI: Lightweight, interactive chat interface for querying medical data in natural language.
- 🐳 End-to-End Dockerization: Ready-to-run container setup for seamless local or cloud deployment.
- ⚙️ Modular Architecture: Each component (Extraction, Summarization, RAG) is independently executable and reusable.
- 🚀 Cloud-Ready Design: Fully compatible with AWS ECS, GCP Cloud Run, or Render for one-click deployment.

## 📂 Directory Overview
```bash
Directory structure:
└── rawatrahul14-ai_developer/
    ├── README.md
    ├── Dockerfile
    ├── LICENSE
    ├── requirements.txt
    ├── data/
    │   ├── processed_images/
    │   │   └── processed_text.json
    │   ├── processed_medical_data/
    │   │   ├── image_1.jpg_summary.csv
    │   │   ├── image_2.jpg_summary.csv
    │   │   ├── image_3.jpg_summary.csv
    │   │   └── image_4.jpg_summary.csv
    │   └── structured_json/
    │       ├── image_1.jpg_summary.json
    │       ├── image_2.jpg_summary.json
    │       └── image_4.jpg_summary.json
    ├── Task1/
    │   ├── __init__.py
    │   ├── data_search.py
    │   ├── main.py
    │   ├── components/
    │   │   ├── __init__.py
    │   │   ├── comprehend.py
    │   │   └── extraction.py
    │   ├── pipelines/
    │   │   ├── __init__.py
    │   │   ├── comprehend_pipeline.py
    │   │   └── textract_pipeline.py
    │   └── utils/
    │       ├── __init__.py
    │       ├── common.py
    │       └── summary.py
    ├── Task2/
    │   ├── main.py
    │   ├── model/
    │   │   ├── __init__.py
    │   │   └── agent.py
    │   ├── pipelines/
    │   │   ├── __init__.py
    │   │   └── summarizer_pipeline.py
    │   ├── schema/
    │   │   ├── __init__.py
    │   │   └── schema.py
    │   └── utils/
    │       ├── __init__.py
    │       └── common.py
    └── Task3/
        ├── agent_state.py
        ├── app.py
        ├── graph.py
        ├── main.py
        ├── Agents/
        │   ├── __init__.py
        │   ├── fallback.py
        │   ├── generation.py
        │   ├── grader.py
        │   ├── retriever.py
        │   └── rewriter.py
        ├── components/
        │   ├── __init__.py
        │   └── retriever/
        │       ├── __init__.py
        │       └── faiss_retriever.py
        ├── data/
        │   └── faiss_index/
        │       ├── index.faiss
        │       └── index.pkl
        ├── pipelines/
        │   ├── __init__.py
        │   └── build_retriever.py
        ├── router/
        │   ├── __init__.py
        │   └── routes.py
        ├── schema/
        │   └── schemas.py
        └── utils/
            └── __init__.py
```

## 🧠 LangGraph Node Flow

| **Node**            | **Role**                                   |
| ------------------- | ------------------------------------------ |
| `query_rewriter`    | Refines and classifies user queries.       |
| `doc_retriever`     | Retrieves top-k documents from FAISS.      |
| `doc_grader`        | Grades and filters document relevance.     |
| `answer_generation` | Synthesizes the final contextual response. |
| `fallback_agent`    | Handles empty or off-topic queries.        |

## 🧾 Data Management Overview

| **Data Type**        | **Location**                   | **Purpose**            |
| -------------------- | ------------------------------ | ---------------------- |
| Extracted Text       | `data/processed_images/`       | Textract OCR output    |
| Medical Entities     | `data/processed_medical_data/` | Comprehend CSVs        |
| Structured Summaries | `data/structured_json/`        | GPT-Generated JSON     |
| FAISS Index          | `Task3/data/faiss_index/`      | Vector Store           |
| MongoDB Checkpoints  | Cloud                          | Saves LangGraph states |

## ⚙️ Setup & Configuration

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/RawatRahul14/AI_Developer.git
cd AI_Developer
```

### 2️⃣ Create Virtual Environment & Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 3️⃣ Configure Environment Variables
Create a .env file in the root directory:
```bash
OPENAI_API_KEY=your_openai_key
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
MONGODB_URI=your_mongodb_uri
DB_NAME=rag_db
COLLECTION_NAME=checkpoints
```

## 🧰 Tech Stack Summary
| **Category**         | **Technologies Used**            |
| -------------------- | -------------------------------- |
| **Language**         | Python 3.11                      |
| **Backend**          | FastAPI                          |
| **Frontend**         | Streamlit                        |
| **Orchestration**    | LangGraph + LangChain            |
| **Vector DB**        | FAISS                            |
| **Storage**          | MongoDB (Async Checkpoint Saver) |
| **Cloud APIs**       | AWS Textract, Comprehend Medical |
| **Containerization** | Docker                           |