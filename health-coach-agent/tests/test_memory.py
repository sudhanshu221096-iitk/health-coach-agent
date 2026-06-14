"""
tests/test_memory.py

Sanity checks for the memory_updater node and SessionStore:
- New session starts with empty history
- memory_updater appends both user and agent turns
- Session isolation: two sessions don't share history
- Session store is idempotent on re-create with same ID
- get() returns None for unknown session
- History accumulates across multiple turns
"""

from __future__ import annotations

import pytest

from agent.nodes.memory_updater import memory_updater_node
from agent.state import AgentState
from api.session_store import SessionStore


def make_state(**kwargs) -> AgentState:
    defaults = dict(
        session_id="s1", mode="checkin", patient_profile={},
        day_number=1, session_history=[], current_input="",
        rag_context="", response="", error="",
    )
    defaults.update(kwargs)
    return AgentState(**defaults)


class TestSessionStore:

    def test_new_session_has_empty_history(self):
        store = SessionStore()
        sid = store.create()
        state = store.get(sid)
        assert state["session_history"] == []

    def test_create_returns_unique_ids(self):
        store = SessionStore()
        ids = {store.create() for _ in range(10)}
        assert len(ids) == 10

    def test_create_with_same_id_is_idempotent(self):
        store = SessionStore()
        sid = store.create("fixed-id")
        sid2 = store.create("fixed-id")
        assert sid == sid2 == "fixed-id"
        # Should not reset existing session
        state = store.get(sid)
        state["day_number"] = 5
        store.update(sid, state)
        store.create("fixed-id")  # re-create
        assert store.get("fixed-id")["day_number"] == 5

    def test_get_returns_none_for_unknown_session(self):
        store = SessionStore()
        assert store.get("does-not-exist") is None

    def test_exists_returns_false_for_unknown(self):
        store = SessionStore()
        assert not store.exists("ghost")

    def test_exists_returns_true_after_create(self):
        store = SessionStore()
        sid = store.create()
        assert store.exists(sid)

    def test_update_persists_state(self):
        store = SessionStore()
        sid = store.create()
        state = store.get(sid)
        state["day_number"] = 7
        store.update(sid, state)
        assert store.get(sid)["day_number"] == 7

    def test_two_sessions_are_isolated(self):
        store = SessionStore()
        sid1 = store.create()
        sid2 = store.create()
        s1 = store.get(sid1)
        s1["day_number"] = 3
        store.update(sid1, s1)
        assert store.get(sid2)["day_number"] == 1


class TestMemoryUpdaterNode:

    def test_appends_user_and_agent_turns(self):
        state = make_state(
            current_input="I slept 7 hours",
            response="That's great — quality sleep is key!",
            mode="checkin",
            day_number=2,
        )
        result = memory_updater_node(state)
        new_turns = result["session_history"]
        assert len(new_turns) == 2
        roles = [t["role"] for t in new_turns]
        assert "user" in roles
        assert "agent" in roles

    def test_appends_only_agent_turn_when_no_input(self):
        state = make_state(current_input="", response="How are you today?", day_number=1)
        result = memory_updater_node(state)
        assert len(result["session_history"]) == 1
        assert result["session_history"][0]["role"] == "agent"

    def test_appends_nothing_when_both_empty(self):
        state = make_state(current_input="", response="")
        result = memory_updater_node(state)
        assert result["session_history"] == []

    def test_turn_contains_correct_day_and_mode(self):
        state = make_state(
            current_input="Feeling great",
            response="Wonderful!",
            day_number=5,
            mode="checkin",
        )
        result = memory_updater_node(state)
        for turn in result["session_history"]:
            assert turn["day"] == 5
            assert turn["mode"] == "checkin"

    def test_history_accumulates_across_turns(self):
        """Simulate operator.add behaviour over 3 consecutive turns."""
        accumulated = []
        for i in range(3):
            state = make_state(
                current_input=f"User turn {i}",
                response=f"Agent turn {i}",
                session_history=accumulated,
                day_number=i + 1,
            )
            result = memory_updater_node(state)
            accumulated = accumulated + result["session_history"]
        assert len(accumulated) == 6  # 2 turns per iteration × 3
