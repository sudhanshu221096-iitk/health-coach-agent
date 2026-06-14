"""
tests/test_day_router.py — Sanity checks for the day_router node.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from agent.nodes.day_router import _DAY_TEMPLATES, get_day_bucket
from agent.state import AgentState


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")


def _make_state(day: int, profile: dict | None = None) -> AgentState:
    return AgentState(
        session_id="t1", mode="checkin",
        patient_profile=profile or {"name": "Priya", "primary_goals": ["sleep better"],
                                    "activity_level": "lightly_active",
                                    "sleep_quality": "fair", "motivation": "energy"},
        day_number=day, session_history=[], current_input="",
        rag_context="", response="", error="",
    )


def _run_node(day: int, llm_response: str = "How are you today?") -> dict:
    mock_resp = MagicMock()
    mock_resp.text = llm_response
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_resp

    with patch("agent.nodes.day_router._get_client", return_value=mock_client):
        from agent.nodes.day_router import day_router_node
        return day_router_node(_make_state(day))


class TestDayBucketSelection:

    def test_day_1(self): assert get_day_bucket(1) == "day_1"
    def test_day_2(self): assert get_day_bucket(2) == "day_2_3"
    def test_day_3(self): assert get_day_bucket(3) == "day_2_3"
    def test_day_4(self): assert get_day_bucket(4) == "day_4_5"
    def test_day_5(self): assert get_day_bucket(5) == "day_4_5"
    def test_day_6(self): assert get_day_bucket(6) == "day_6_7"
    def test_day_7(self): assert get_day_bucket(7) == "day_6_7"
    def test_day_8(self): assert get_day_bucket(8) == "day_8_plus"
    def test_day_100(self): assert get_day_bucket(100) == "day_8_plus"

    def test_all_buckets_non_empty(self):
        for bucket, templates in _DAY_TEMPLATES.items():
            assert len(templates) >= 1


class TestDayRouterNode:

    def test_returns_response_day_1(self):
        result = _run_node(1, "Welcome! How are you feeling today?")
        assert result["response"]
        assert result["error"] == ""

    def test_returns_response_day_5(self):
        result = _run_node(5, "Halfway through -- how's your energy?")
        assert result["response"]

    def test_returns_response_day_7(self):
        result = _run_node(7, "Day 7 -- incredible journey!")
        assert result["response"]

    def test_gemini_failure_returns_fallback(self):
        with patch("agent.nodes.day_router._get_client", side_effect=Exception("API down")):
            from agent.nodes.day_router import day_router_node
            result = day_router_node(_make_state(3))
        assert "check-in" in result["response"].lower()
        assert result["error"]
