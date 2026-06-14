"""
rag/retriever.py

Performs semantic similarity search over the wellness protocol ChromaDB collection.
"""

from __future__ import annotations

import logging
import os
from typing import List

import rag._compat  # noqa: F401 — patches sqlite3 before chromadb loads
import chromadb
from dotenv import load_dotenv

from rag.embeddings import GeminiQueryEmbeddingFunction
from rag.ingest import COLLECTION_NAME, get_chroma_client

load_dotenv()
logger = logging.getLogger(__name__)

_query_embed_fn = None  # lazy-initialised singleton


def _get_query_embed_fn() -> GeminiQueryEmbeddingFunction:
    global _query_embed_fn
    if _query_embed_fn is None:
        _query_embed_fn = GeminiQueryEmbeddingFunction()
    return _query_embed_fn


def retrieve_context(query: str, n_results: int = 5) -> List[str]:
    """
    Retrieve the top-n most relevant protocol chunks for a given query.

    Args:
        query: The patient's question.
        n_results: Number of chunks to return.

    Returns:
        List of text strings (empty list if collection is empty or unavailable).
    """
    if not query.strip():
        return []

    try:
        client = get_chroma_client()
        embed_fn = _get_query_embed_fn()

        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn,
        )

        if collection.count() == 0:
            logger.warning("ChromaDB collection is empty — has ingestion been run?")
            return []

        query_embedding = embed_fn([query])[0]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, collection.count()),
        )

        documents = results.get("documents", [[]])[0]
        return [doc for doc in documents if doc]

    except Exception as e:
        logger.exception("retrieve_context failed for query=%r: %s", query, e)
        return []
