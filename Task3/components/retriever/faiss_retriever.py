# === Python Modules ===
import os
import re
import json
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# === FAISS Retriever Class ===
class FAISSRetriever:
    """
    Handles FAISS vector store creation, loading, and retrieval
    using LangChain Document objects.
    """

    def __init__(
        self,
        index_path: str = "Task3/data/faiss_index",
        embedding_model: str = "text-embedding-3-small"
    ):
        self.index_path = index_path
        self.embeddings = OpenAIEmbeddings(model = embedding_model)
        self.db = None

        ## === Creating the directory if not exist ===
        os.makedirs(
            index_path,
            exist_ok = True
        )

    # === Helper: Convert JSON/Text dict into LangChain Documents ===
    @staticmethod
    def create_documents_from_json(
        data_dict: Dict[str, Dict[str, str]],
        data: Dict[str, str]
    ) -> List[Document]:
        """
        Converts a dict of filename:text into a list of LangChain Document objects with metadata.

        Args:
            data_dict (Dict[str, Dict[str, str]]): e.g. {"'image_1.jpg_summary.json': {'patient': 'Anupama Joshi', ..."}
            data (Dict[str, str]): e.g. {"image_1.json": "Patient details text...", ...}

        Returns:
            List[Document]: list of LangChain Document objects
        """
        documents = []
        for filename, content in data_dict.items():

            ## === Getting metadata ===
            name = content.get("patient")

            metadata = {
                "source_file": filename,
                "patient_name": name
            }

            doc = Document(
                page_content = data[filename],
                metadata = metadata
            )
            documents.append(doc)

        print(f"Created {len(documents)} Document objects with metadata.")
        return documents

    # === Build index from Document objects ===
    def build_index(
            self,
            documents: List[Document]
    ) -> None:
        """
        Builds and saves a FAISS index from LangChain Document objects.
        """
        print("Building FAISS index from documents...")

        if not documents:
            raise ValueError("No documents provided to build the index.")

        self.db = FAISS.from_documents(
            documents = documents,
            embedding = self.embeddings
        )

        self.db.save_local(self.index_path)
        print(f"FAISS index saved at: {os.path.abspath(self.index_path)}")

    # === Load existing FAISS index ===
    def load_index(
            self
    ) -> None:
        """
        Loads a previously saved FAISS index.
        """
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"No FAISS index found at {self.index_path}")

        self.db = FAISS.load_local(
            folder_path = self.index_path,
            embeddings = self.embeddings,
            allow_dangerous_deserialization = True
        )
        print(f"✅ Loaded FAISS index from {os.path.abspath(self.index_path)}")

    # === Retrieve relevant documents ===
    def retrieve(
            self,
            query: str,
            top_k: int = 1
    ) -> List[Document]:
        """
        Retrieve top-k similar documents for a given query.
        """
        if self.db is None:
            self.load_index()

        print(f"Retrieving top-{top_k} results for: '{query}'")
        results = self.db.similarity_search(
            query,
            k = top_k
        )
        return results