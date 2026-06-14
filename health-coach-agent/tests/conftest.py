"""
tests/conftest.py

Shared fixtures and Gemini API mock setup.
All tests mock genai.GenerativeModel and genai.embed_content so
no real API key is needed to run the test suite.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


# ── Fake Gemini response factory ─────────────────────────────────────────────

def make_fake_model(response_text: str):
    """Return a mock GenerativeModel that always replies with `response_text`."""
    mock_response = MagicMock()
    mock_response.text = response_text

    mock_chat = MagicMock()
    mock_chat.send_message.return_value = mock_response

    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response
    mock_model.start_chat.return_value = mock_chat
    return mock_model


SAMPLE_PROFILE_JSON = json.dumps({
    "name": "Priya",
    "age": 32,
    "primary_goals": ["lose weight", "sleep better"],
    "sleep_hours": 6.0,
    "sleep_quality": "fair",
    "activity_level": "lightly_active",
    "dietary_restrictions": ["vegetarian"],
    "health_conditions": [],
    "motivation": "To have energy for my kids and feel confident in my body",
})

SAMPLE_WELCOME = "Welcome, Priya! I'm so glad you're here."

SAMPLE_PROFILE = json.loads(SAMPLE_PROFILE_JSON)


@pytest.fixture()
def sample_profile():
    return SAMPLE_PROFILE.copy()


@pytest.fixture()
def base_state(sample_profile):
    from agent.state import AgentState
    return AgentState(
        session_id="test-session",
        mode="onboard",
        patient_profile=sample_profile,
        day_number=1,
        session_history=[],
        current_input="",
        rag_context="",
        response="",
        error="",
    )
