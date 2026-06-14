"""
api/routes/checkin.py — POST /checkin
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from agent.graph import agent_graph
from agent.state import AgentState, HistoryTurn
from api.models import AgentResponse, CheckInRequest
from api.session_store import session_store

router = APIRouter()


@router.post("/checkin", response_model=AgentResponse, tags=["Agent"])
async def checkin(req: CheckInRequest) -> AgentResponse:
    """
    Run a day-adaptive daily check-in.

    - If `user_response` is provided, it is stored as the patient's reply to the
      previous agent question before generating the next question.
    - `day_number` advances the protocol timeline and drives question selection.
    """
    if not session_store.exists(req.session_id):
        raise HTTPException(status_code=404, detail="Session not found. Please onboard first.")

    state = session_store.get(req.session_id)

    # Store any user reply to the previous question
    if req.user_response:
        state["session_history"].append(
            HistoryTurn(
                role="user",
                content=req.user_response,
                day=req.day_number,
                mode="checkin",
            )
        )

    state["mode"] = "checkin"
    state["day_number"] = req.day_number
    state["current_input"] = req.user_response or ""

    result: AgentState = agent_graph.invoke(state)
    session_store.update(req.session_id, result)

    return AgentResponse(
        session_id=req.session_id,
        response=result["response"],
        mode="checkin",
        day_number=req.day_number,
        patient_name=result.get("patient_profile", {}).get("name"),
        error=result.get("error") or None,
    )
