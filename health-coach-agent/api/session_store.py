"""
api/session_store.py

Thread-safe in-memory session store. For the MVP, sessions live for the
lifetime of the server process. A Redis/DB backend can slot in here later.
"""

from __future__ import annotations

import threading
import uuid
from typing import Dict, Optional

from agent.state import AgentState


def _empty_state(session_id: str) -> AgentState:
    return AgentState(
        session_id=session_id,
        mode="onboard",
        patient_profile={},
        day_number=1,
        session_history=[],
        current_input="",
        rag_context="",
        response="",
        error="",
    )


class SessionStore:
    """Simple thread-safe dict-backed session store."""

    def __init__(self) -> None:
        self._store: Dict[str, AgentState] = {}
        self._lock = threading.Lock()

    def create(self, session_id: Optional[str] = None) -> str:
        sid = session_id or str(uuid.uuid4())
        with self._lock:
            if sid not in self._store:
                self._store[sid] = _empty_state(sid)
        return sid

    def get(self, session_id: str) -> Optional[AgentState]:
        with self._lock:
            return self._store.get(session_id)

    def update(self, session_id: str, new_state: AgentState) -> None:
        with self._lock:
            self._store[session_id] = new_state

    def exists(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._store

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


# Module-level singleton
session_store = SessionStore()
