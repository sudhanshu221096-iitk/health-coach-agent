"""
api/routes/onboard.py — POST /onboard
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from agent.graph import agent_graph
from agent.state import AgentState
from api.models import AgentResponse, OnboardRequest
from api.session_store import session_store

router = APIRouter()


@router.post("/onboard", response_model=AgentResponse, tags=["Agent"])
async def onboard(req: OnboardRequest) -> AgentResponse:
    """
    Parse a free-form patient profile and initialise the session.

    - Creates a new session if `session_id` is omitted.
    - Returns the structured welcome message and the assigned session_id.
    """
    sid = session_store.create(req.session_id)
    state = session_store.get(sid)

    # Patch state for this turn
    state["mode"] = "onboard"
    state["current_input"] = req.profile_text

    # Run the agent graph
    result: AgentState = agent_graph.invoke(state)

    # Persist updated state
    session_store.update(sid, result)

    return AgentResponse(
        session_id=sid,
        response=result["response"],
        mode="onboard",
        day_number=result.get("day_number", 1),
        patient_name=result.get("patient_profile", {}).get("name"),
        error=result.get("error") or None,
    )
