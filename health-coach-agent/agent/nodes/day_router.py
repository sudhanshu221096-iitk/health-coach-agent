"""
agent/nodes/day_router.py

Generates adaptive daily check-in questions based on protocol day number.
Uses google-genai SDK.
"""

from __future__ import annotations

import logging
import os
from typing import List

import google.genai as genai
from dotenv import load_dotenv

from agent.state import AgentState

load_dotenv()
logger = logging.getLogger(__name__)

_DAY_TEMPLATES: dict[str, List[str]] = {
    "day_1": [
        "How are you feeling right now, physically and emotionally?",
        "What's your biggest hope for this wellness journey?",
        "On a scale of 1-10, how would you rate your energy levels today?",
    ],
    "day_2_3": [
        "How did Day 1 go? What was easier than expected -- and what was harder?",
        "Were you able to follow any of yesterday's goals? Tell me about it.",
        "What's one small win from yesterday you can feel good about?",
    ],
    "day_4_5": [
        "You're nearly halfway through the first week! How are your energy and mood evolving?",
        "Which habit has started to feel a little more automatic -- even slightly?",
        "Is there anything from the protocol you'd like to revisit or understand better?",
    ],
    "day_6_7": [
        "You've made it almost a full week -- how does that feel?",
        "What surprised you most about yourself during this first week?",
        "Looking back, which habit made the biggest difference, and which needs more work?",
    ],
    "day_8_plus": [
        "How are things going as you continue your journey past the first week?",
        "What feels sustainable long-term, and what still feels like a stretch?",
        "What's one thing you'd like to focus on or improve in the days ahead?",
    ],
}

_PERSONALISE_PROMPT = """\
You are a warm, empathetic health coach conducting a daily check-in.

Patient profile:
- Name: {name}
- Primary goals: {goals}
- Activity level: {activity}
- Sleep quality: {sleep}
- Motivation: {motivation}

Today is Day {day} of their protocol.

Base question to personalise:
"{question}"

Previous responses in this session (for context):
{history}

Rewrite the base question to feel personal and specific to THIS patient. \
Keep it to 1-2 sentences max. Warm, conversational, not clinical. \
Do NOT add bullet lists or multiple questions -- just one focused question.
"""


def _select_template(day: int) -> str:
    import random
    bucket = get_day_bucket(day)
    return random.choice(_DAY_TEMPLATES[bucket])


def get_day_bucket(day: int) -> str:
    """Return template bucket key for a given day (used in tests)."""
    if day == 1:
        return "day_1"
    elif day <= 3:
        return "day_2_3"
    elif day <= 5:
        return "day_4_5"
    elif day <= 7:
        return "day_6_7"
    return "day_8_plus"



def _get_client():
    """Patchable factory."""
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def day_router_node(state: AgentState) -> dict:
    """LangGraph node: select + personalise a day-appropriate check-in question."""
    try:
        import agent.nodes.day_router as _self
        client = _self._get_client()
        profile = state.get("patient_profile", {})
        day = state.get("day_number", 1)
        history = state.get("session_history", [])
        base_question = _select_template(day)

        recent = "\n".join(
            f"  [{t['mode']} Day {t['day']}] {t['role']}: {t['content']}"
            for t in history[-3:]
        ) or "  (no prior check-ins yet)"

        result = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=_PERSONALISE_PROMPT.format(
                name=profile.get("name", "Friend"),
                goals=", ".join(profile.get("primary_goals", [])) or "general wellness",
                activity=profile.get("activity_level", "unknown"),
                sleep=profile.get("sleep_quality", "unknown"),
                motivation=profile.get("motivation", "improving health"),
                day=day,
                question=base_question,
                history=recent,
            ),
        )

        return {"response": result.text.strip(), "error": ""}

    except Exception as e:
        logger.exception("day_router_node failed")
        return {
            "response": f"Day {state.get('day_number', 1)} check-in: How are you feeling today?",
            "error": str(e),
        }
