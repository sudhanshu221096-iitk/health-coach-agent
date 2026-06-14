"""
agent/nodes/memory_updater.py

Appends the latest agent response and user input to session_history.
Always runs as the final node before END.
"""

from __future__ import annotations

from datetime import datetime, timezone

from agent.state import AgentState, HistoryTurn


def memory_updater_node(state: AgentState) -> dict:
    """LangGraph node: persist the current turn into session_history."""
    day = state.get("day_number", 1)
    mode = state.get("mode", "ask")
    user_input = state.get("current_input", "")
    agent_response = state.get("response", "")

    new_turns: list[HistoryTurn] = []

    if user_input:
        new_turns.append(
            HistoryTurn(
                role="user",
                content=user_input,
                day=day,
                mode=mode,
            )
        )

    if agent_response:
        new_turns.append(
            HistoryTurn(
                role="agent",
                content=agent_response,
                day=day,
                mode=mode,
            )
        )

    # Annotated[list, operator.add] means we RETURN new items;
    # LangGraph will concatenate them to the existing list.
    return {"session_history": new_turns}
