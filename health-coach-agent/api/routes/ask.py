"""
api/routes/ask.py — POST /ask
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from agent.graph import agent_graph
from agent.state import AgentState
from api.models import AgentResponse, AskRequest
from api.session_store import session_store

router = APIRouter()


@router.post("/ask", response_model=AgentResponse, tags=["Agent"])
async def ask(req: AskRequest) -> AgentResponse:
    """
    Answer a protocol question using RAG — grounded strictly in the wellness PDF.

    Responses are bounded by the retrieved context; if the answer isn't in the
    protocol, the agent says so clearly rather than hallucinating.
    """
    if not session_store.exists(req.session_id):
        raise HTTPException(status_code=404, detail="Session not found. Please onboard first.")

    state = session_store.get(req.session_id)
    state["mode"] = "ask"
    state["current_input"] = req.question

    result: AgentState = agent_graph.invoke(state)
    session_store.update(req.session_id, result)

    return AgentResponse(
        session_id=req.session_id,
        response=result["response"],
        mode="ask",
        day_number=result.get("day_number", 1),
        patient_name=result.get("patient_profile", {}).get("name"),
        error=result.get("error") or None,
    )
