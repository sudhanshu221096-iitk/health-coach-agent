"""
api/models.py — Pydantic request and response schemas.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Requests ─────────────────────────────────────────────────────────────────

class OnboardRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Omit to create a new session")
    profile_text: str = Field(..., min_length=10, description="Free-form patient onboarding text")


class CheckInRequest(BaseModel):
    session_id: str
    user_response: Optional[str] = Field(
        None, description="Patient's reply to the previous check-in question (if any)"
    )
    day_number: int = Field(1, ge=1, description="Current day of the protocol")


class AskRequest(BaseModel):
    session_id: str
    question: str = Field(..., min_length=3, description="Patient's protocol question")


# ── Responses ─────────────────────────────────────────────────────────────────

class AgentResponse(BaseModel):
    session_id: str
    response: str
    mode: str
    day_number: int
    patient_name: Optional[str] = None
    error: Optional[str] = None


class SessionStateResponse(BaseModel):
    session_id: str
    patient_profile: Dict[str, Any]
    day_number: int
    history_length: int
    history: List[Dict[str, Any]]
