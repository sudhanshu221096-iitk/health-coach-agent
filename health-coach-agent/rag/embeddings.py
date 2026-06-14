"""
rag/embeddings.py

ChromaDB EmbeddingFunction backed by google-genai text-embedding-004.
Imports _compat first to patch sqlite3 for ChromaDB.
"""

from __future__ import annotations

import os
from typing import List

import rag._compat  # noqa: F401 — patches sqlite3 before chromadb loads
import google.genai as genai
from chromadb import EmbeddingFunction, Embeddings
from dotenv import load_dotenv

load_dotenv()


class GeminiEmbeddingFunction(EmbeddingFunction):
    """ChromaDB-compatible embedding function using Gemini embedding model."""

    def __init__(
        self,
        model: str = "models/gemini-embedding-001",
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> None:
        self.model = model
        self.task_type = task_type
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def __call__(self, input: List[str]) -> Embeddings:  # noqa: A002
        embeddings: Embeddings = []
        for text in input:
            result = self._client.models.embed_content(
                model=self.model,
                contents=text,
                config={"task_type": self.task_type},
            )
            embeddings.append(result.embeddings[0].values)
        return embeddings


class GeminiQueryEmbeddingFunction(GeminiEmbeddingFunction):
    """Query-task embedding function for retrieval-time embeddings."""

    def __init__(self, model: str = "models/gemini-embedding-001") -> None:
        super().__init__(model=model, task_type="RETRIEVAL_QUERY")
