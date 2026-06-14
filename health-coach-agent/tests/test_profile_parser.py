"""
tests/test_profile_parser.py — Sanity checks for profile_parser node.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import SAMPLE_PROFILE_JSON, SAMPLE_WELCOME


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")


def _mock_client(responses: list[str]) -> MagicMock:
    """Build a mock genai client whose generate_content cycles through responses."""
    resp_iter = iter(responses)

    def gen_content(**kwargs):
        r = MagicMock()
        r.text = next(resp_iter)
        return r

    client = MagicMock()
    client.models.generate_content.side_effect = gen_content
    return client


def _run(text: str, extraction: str, welcome: str) -> dict:
    from agent.state import AgentState
    state = AgentState(
        session_id="t1", mode="onboard", patient_profile={}, day_number=1,
        session_history=[], current_input=text, rag_context="", response="", error="",
    )
    mock = _mock_client([extraction, welcome])
    # Patch the patchable factory directly on the module
    with patch("agent.nodes.profile_parser._get_client", return_value=mock):
        from agent.nodes.profile_parser import profile_parser_node
        return profile_parser_node(state)


class TestProfileParser:

    def test_valid_full_profile_parsed_correctly(self):
        result = _run(
            "I'm Priya, 32. I want to lose weight and sleep better.",
            SAMPLE_PROFILE_JSON, SAMPLE_WELCOME,
        )
        assert result["patient_profile"]["name"] == "Priya"
        assert result["patient_profile"]["age"] == 32
        assert "lose weight" in result["patient_profile"]["primary_goals"]
        assert result["response"] == SAMPLE_WELCOME
        assert result["error"] == ""

    def test_partial_profile_uses_defaults(self):
        partial = json.dumps({
            "name": "Alex", "age": None, "primary_goals": ["feel better"],
            "sleep_hours": None, "sleep_quality": None, "activity_level": None,
            "dietary_restrictions": [], "health_conditions": [], "motivation": "",
        })
        result = _run("I'm Alex.", partial, "Hi Alex!")
        assert result["patient_profile"]["name"] == "Alex"
        assert result["error"] == ""

    def test_malformed_json_returns_fallback(self):
        result = _run("Some text", "NOT VALID JSON {{{{", "Welcome!")
        assert result["patient_profile"]["name"] == "Friend"
        assert "JSON parse error" in result["error"]
        assert result["response"]

    def test_empty_profile_text_handled(self):
        empty_json = json.dumps({
            "name": "Friend", "age": None, "primary_goals": [],
            "sleep_hours": None, "sleep_quality": None, "activity_level": None,
            "dietary_restrictions": [], "health_conditions": [], "motivation": "",
        })
        result = _run("", empty_json, "Welcome!")
        assert result["patient_profile"] is not None
        assert result["response"]

    def test_welcome_message_non_empty(self):
        result = _run("I'm Sam, 45.", SAMPLE_PROFILE_JSON, "Sam, so glad you're here!")
        assert len(result["response"]) > 10
