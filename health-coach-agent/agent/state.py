"""
Shared TypedDict state that flows through every node in the LangGraph agent.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional
import operator
from typing_extensions import TypedDict


class PatientProfile(TypedDict, total=False):
    name: str
    age: Optional[int]
    primary_goals: List[str]
    sleep_hours: Optional[float]
    sleep_quality: Optional[str]      # poor | fair | good | excellent
    activity_level: Optional[str]     # sedentary | lightly_active | moderately_active | very_active
    dietary_restrictions: List[str]
    health_conditions: List[str]
    motivation: str


class HistoryTurn(TypedDict):
    role: str          # "agent" | "user"
    content: str
    day: int
    mode: str          # "onboard" | "checkin" | "ask"


class AgentState(TypedDict):
    session_id: str
    mode: str                                          # onboard | checkin | ask
    patient_profile: Dict[str, Any]
    day_number: int
    session_history: Annotated[List[HistoryTurn], operator.add]   # accumulates
    current_input: str
    rag_context: str
    response: str
    error: str
