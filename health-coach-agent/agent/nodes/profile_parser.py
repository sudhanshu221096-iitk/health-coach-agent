"""
agent/nodes/profile_parser.py

Parses free-form patient onboarding text into a structured PatientProfile dict.
Uses google-genai SDK (google.genai).
"""

from __future__ import annotations

import json
import logging
import os
import re

import google.genai as genai
from dotenv import load_dotenv

from agent.state import AgentState

load_dotenv()
logger = logging.getLogger(__name__)

_EXTRACTION_PROMPT = """\
You are a patient onboarding assistant. Extract structured information from the \
patient's text below.

Patient text:
\"\"\"{text}\"\"\"

Return ONLY a valid JSON object with these exact keys (use null for missing fields):
{{
  "name": "<string, default 'Friend' if not mentioned>",
  "age": <integer or null>,
  "primary_goals": ["<goal1>", "<goal2>"],
  "sleep_hours": <float or null>,
  "sleep_quality": "<poor|fair|good|excellent or null>",
  "activity_level": "<sedentary|lightly_active|moderately_active|very_active or null>",
  "dietary_restrictions": ["<item>"],
  "health_conditions": ["<condition>"],
  "motivation": "<one sentence describing their core motivation>"
}}

No markdown fences, no explanation -- raw JSON only.
"""

_CONFIRMATION_PROMPT = """\
You are a warm, encouraging health coach. A patient just shared their profile:
{profile_json}

Write a 2-3 sentence warm welcome message that:
1. Addresses them by name
2. Reflects back their primary goal(s)
3. Expresses genuine enthusiasm for supporting them

Tone: warm, clear, not clinical, not fluffy. No bullet points.
"""


def _get_client():
    """Patchable factory — override in tests with a mock client."""
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def profile_parser_node(state: AgentState) -> dict:
    """LangGraph node: parse onboarding text -> structured profile + welcome message."""
    try:
        import agent.nodes.profile_parser as _self
        client = _self._get_client()

        # Step 1: Extract structured profile
        extraction = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=_EXTRACTION_PROMPT.format(text=state["current_input"]),
        )
        raw = extraction.text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        profile = json.loads(raw)

        # Step 2: Generate warm confirmation
        confirmation = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=_CONFIRMATION_PROMPT.format(profile_json=json.dumps(profile, indent=2)),
        )

        return {
            "patient_profile": profile,
            "response": confirmation.text.strip(),
            "error": "",
        }

    except json.JSONDecodeError as e:
        logger.error("Profile JSON parse failed: %s", e)
        return {
            "patient_profile": {"name": "Friend", "primary_goals": [], "motivation": ""},
            "response": (
                "Thanks for sharing! I wasn't able to capture all the details perfectly, "
                "but I'm here to support your wellness journey."
            ),
            "error": f"JSON parse error: {e}",
        }
    except Exception as e:
        logger.exception("profile_parser_node failed")
        return {"response": "Something went wrong during onboarding.", "error": str(e)}
