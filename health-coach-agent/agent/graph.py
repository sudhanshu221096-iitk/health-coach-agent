"""
agent/graph.py

Defines and compiles the LangGraph StateGraph for the health coach agent.

Flow:
  START → dispatcher → [profile_parser | day_router | rag_answerer]
                              ↓
                       memory_updater → END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent.nodes.day_router import day_router_node
from agent.nodes.memory_updater import memory_updater_node
from agent.nodes.profile_parser import profile_parser_node
from agent.nodes.rag_answerer import rag_answerer_node
from agent.state import AgentState


def _dispatcher(state: AgentState) -> AgentState:
    """Pass-through dispatcher; routing logic lives in the conditional edge."""
    return {}


def _route_by_mode(state: AgentState) -> str:
    """Conditional edge: route to the right processing node based on mode."""
    mode = state.get("mode", "ask")
    if mode == "onboard":
        return "profile_parser"
    elif mode == "checkin":
        return "day_router"
    else:
        return "rag_answerer"


def build_graph() -> StateGraph:
    """Build and compile the LangGraph agent graph."""
    g = StateGraph(AgentState)

    # Register nodes
    g.add_node("dispatcher", _dispatcher)
    g.add_node("profile_parser", profile_parser_node)
    g.add_node("day_router", day_router_node)
    g.add_node("rag_answerer", rag_answerer_node)
    g.add_node("memory_updater", memory_updater_node)

    # Entry point
    g.add_edge(START, "dispatcher")

    # Conditional routing
    g.add_conditional_edges(
        "dispatcher",
        _route_by_mode,
        {
            "profile_parser": "profile_parser",
            "day_router": "day_router",
            "rag_answerer": "rag_answerer",
        },
    )

    # All processing nodes flow into memory_updater, then end
    for node in ("profile_parser", "day_router", "rag_answerer"):
        g.add_edge(node, "memory_updater")

    g.add_edge("memory_updater", END)

    return g.compile()


# Compiled singleton — import this in the API layer
agent_graph = build_graph()
