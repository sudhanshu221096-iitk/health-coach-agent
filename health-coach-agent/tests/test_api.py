"""
tests/test_api.py

Integration-level sanity checks for all API endpoints.
The agent graph is fully mocked so no Gemini API key is required.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from agent.state import AgentState


# ── Minimal mock agent graph ─────────────────────────────────────────────────

def _fake_graph_invoke(state: AgentState) -> AgentState:
    mode = state.get("mode", "ask")
    if mode == "onboard":
        return {
            **state,
            "patient_profile": {"name": "Priya", "primary_goals": ["sleep better"]},
            "response": "Welcome, Priya!",
            "error": "",
        }
    elif mode == "checkin":
        return {**state, "response": "How did you sleep last night?", "error": ""}
    else:
        return {**state, "response": "Drink 2.5 L of water daily.", "error": ""}


@pytest.fixture()
def client():
    # Patch the graph and RAG ingestion before importing the app
    with (
        patch("agent.graph.agent_graph") as mock_graph,
        patch("rag.ingest.ingest_documents", return_value=10),
    ):
        mock_graph.invoke.side_effect = _fake_graph_invoke
        # Patch after imports resolve
        import api.routes.onboard as onboard_mod
        import api.routes.checkin as checkin_mod
        import api.routes.ask as ask_mod
        onboard_mod.agent_graph = mock_graph
        checkin_mod.agent_graph = mock_graph
        ask_mod.agent_graph = mock_graph

        from api.main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_returns_ok_status(self, client):
        r = client.get("/health")
        assert r.json()["status"] == "ok"


# ── POST /api/onboard ─────────────────────────────────────────────────────────

class TestOnboardEndpoint:
    def test_onboard_creates_session(self, client):
        r = client.post("/api/onboard", json={"profile_text": "I'm Priya, 32, want to sleep better."})
        assert r.status_code == 200
        data = r.json()
        assert "session_id" in data
        assert data["session_id"]

    def test_onboard_returns_response(self, client):
        r = client.post("/api/onboard", json={"profile_text": "My name is Priya, I want to sleep better."})
        assert r.json()["response"]

    def test_onboard_returns_correct_mode(self, client):
        r = client.post("/api/onboard", json={"profile_text": "I'm Alex, I want more energy."})
        assert r.json()["mode"] == "onboard"

    def test_onboard_with_explicit_session_id(self, client):
        r = client.post("/api/onboard", json={
            "session_id": "my-custom-id",
            "profile_text": "I'm Sam, I want to reduce stress.",
        })
        assert r.json()["session_id"] == "my-custom-id"

    def test_onboard_rejects_short_profile_text(self, client):
        r = client.post("/api/onboard", json={"profile_text": "Hi"})
        assert r.status_code == 422


# ── POST /api/checkin ─────────────────────────────────────────────────────────

class TestCheckinEndpoint:
    def _create_session(self, client) -> str:
        r = client.post("/api/onboard", json={"profile_text": "I'm Priya, I want to sleep better."})
        return r.json()["session_id"]

    def test_checkin_returns_200(self, client):
        sid = self._create_session(client)
        r = client.post("/api/checkin", json={"session_id": sid, "day_number": 1})
        assert r.status_code == 200

    def test_checkin_returns_question(self, client):
        sid = self._create_session(client)
        r = client.post("/api/checkin", json={"session_id": sid, "day_number": 1})
        assert r.json()["response"]

    def test_checkin_reflects_day_number(self, client):
        sid = self._create_session(client)
        r = client.post("/api/checkin", json={"session_id": sid, "day_number": 5})
        assert r.json()["day_number"] == 5

    def test_checkin_unknown_session_returns_404(self, client):
        r = client.post("/api/checkin", json={"session_id": "nonexistent", "day_number": 1})
        assert r.status_code == 404

    def test_checkin_with_user_response_accepted(self, client):
        sid = self._create_session(client)
        r = client.post("/api/checkin", json={
            "session_id": sid,
            "day_number": 2,
            "user_response": "I slept 7 hours and felt rested.",
        })
        assert r.status_code == 200


# ── POST /api/ask ─────────────────────────────────────────────────────────────

class TestAskEndpoint:
    def _create_session(self, client) -> str:
        r = client.post("/api/onboard", json={"profile_text": "I'm Priya, I want to sleep better."})
        return r.json()["session_id"]

    def test_ask_returns_200(self, client):
        sid = self._create_session(client)
        r = client.post("/api/ask", json={"session_id": sid, "question": "How much water should I drink?"})
        assert r.status_code == 200

    def test_ask_returns_non_empty_response(self, client):
        sid = self._create_session(client)
        r = client.post("/api/ask", json={"session_id": sid, "question": "What should I eat for breakfast?"})
        assert r.json()["response"]

    def test_ask_unknown_session_returns_404(self, client):
        r = client.post("/api/ask", json={"session_id": "ghost", "question": "What should I eat?"})
        assert r.status_code == 404

    def test_ask_short_question_rejected(self, client):
        sid = self._create_session(client)
        r = client.post("/api/ask", json={"session_id": sid, "question": "Hi"})
        assert r.status_code == 422


# ── GET /api/state/{session_id} ───────────────────────────────────────────────

class TestStateEndpoint:
    def test_state_returns_200_for_existing_session(self, client):
        r = client.post("/api/onboard", json={"profile_text": "I'm Priya, I want to sleep better."})
        sid = r.json()["session_id"]
        r2 = client.get(f"/api/state/{sid}")
        assert r2.status_code == 200

    def test_state_returns_404_for_unknown_session(self, client):
        r = client.get("/api/state/does-not-exist")
        assert r.status_code == 404

    def test_state_contains_session_id(self, client):
        r = client.post("/api/onboard", json={"session_id": "abc123", "profile_text": "I'm Priya, 32, want to sleep better."})
        r2 = client.get("/api/state/abc123")
        assert r2.json()["session_id"] == "abc123"
