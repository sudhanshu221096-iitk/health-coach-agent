"""
tests/test_rag.py — Sanity checks for the RAG pipeline.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from rag.ingest import _chunk_text


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("CHROMA_DB_PATH", "/tmp/test_chroma_hca")
    monkeypatch.setenv("WELLNESS_PDF_PATH", "./data/wellness_protocol.pdf")


class TestChunking:

    def test_small_text_one_chunk(self):
        assert len(_chunk_text("Short text.", 600, 100)) == 1

    def test_long_text_multiple_chunks(self):
        assert len(_chunk_text("A" * 2000, 600, 100)) > 1

    def test_chunk_size_respected(self):
        for c in _chunk_text("B" * 3000, 600, 100):
            assert len(c) <= 600

    def test_empty_text_no_chunks(self):
        assert _chunk_text("", 600, 100) == []

    def test_whitespace_only_excluded(self):
        chunks = _chunk_text(" " * 1000, 600, 100)
        assert all(c.strip() for c in chunks)

    def test_single_char_boundary(self):
        chunks = _chunk_text("X" * 1200, 600, 100)
        assert len(chunks) >= 2


class TestRetriever:

    def test_empty_query_returns_empty(self):
        from rag.retriever import retrieve_context
        assert retrieve_context("") == []

    def test_whitespace_query_returns_empty(self):
        from rag.retriever import retrieve_context
        assert retrieve_context("   ") == []

    def test_collection_error_returns_empty(self):
        with patch("rag.retriever.get_chroma_client") as mc:
            mc.return_value.get_collection.side_effect = Exception("not found")
            from rag.retriever import retrieve_context
            assert retrieve_context("What should I eat?") == []


class TestRagAnswererNode:

    def _state(self, q: str) -> dict:
        from agent.state import AgentState
        return AgentState(
            session_id="t1", mode="ask",
            patient_profile={"name": "Priya"},
            day_number=1, session_history=[],
            current_input=q, rag_context="", response="", error="",
        )

    def test_grounded_answer_when_context_found(self):
        mock_resp = MagicMock()
        mock_resp.text = "Drink 2.5 litres of water daily."
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_resp

        with (
            patch("agent.nodes.rag_answerer._get_client", return_value=mock_client),
            patch("agent.nodes.rag_answerer.retrieve_context",
                  return_value=["You should drink 2.5 L of water daily."]),
        ):
            from agent.nodes.rag_answerer import rag_answerer_node
            result = rag_answerer_node(self._state("How much water should I drink?"))

        assert "water" in result["response"].lower()
        assert result["error"] == ""

    def test_not_in_protocol_when_no_context(self):
        with patch("agent.nodes.rag_answerer.retrieve_context", return_value=[]):
            from agent.nodes.rag_answerer import rag_answerer_node
            result = rag_answerer_node(self._state("What is the meaning of life?"))
        assert "protocol" in result["response"].lower()

    def test_gemini_failure_returns_fallback(self):
        with (
            patch("agent.nodes.rag_answerer.retrieve_context", return_value=["context"]),
            patch("agent.nodes.rag_answerer._get_client", side_effect=Exception("API down")),
        ):
            from agent.nodes.rag_answerer import rag_answerer_node
            result = rag_answerer_node(self._state("Tell me about sleep."))
        assert result["error"]
        assert result["response"]
