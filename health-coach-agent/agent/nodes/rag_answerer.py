"""
agent/nodes/rag_answerer.py

RAG-grounded protocol Q&A using google-genai SDK.
"""

from __future__ import annotations

import logging
import os

import google.genai as genai
from dotenv import load_dotenv

from agent.state import AgentState
from rag.retriever import retrieve_context

load_dotenv()
logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a wellness protocol specialist and health coach.

Answer the patient's question ONLY using the protocol excerpts provided below.

Rules:
1. If the answer is clearly in the context, give a warm, clear, helpful response.
2. If the answer is NOT in the context, say exactly:
   "I don't have specific guidance on that in your current protocol. \
Please consult your healthcare provider for personalised advice."
3. Never invent facts, statistics, or advice not present in the context.
4. Keep your answer concise -- 2-4 sentences unless more detail is truly needed.
5. Tone: warm, clear, not clinical, not fluffy.

Protocol excerpts:
{context}

Patient's name: {name}
Patient's question: {question}
"""


def _get_client():
    """Patchable factory."""
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def rag_answerer_node(state: AgentState) -> dict:
    """LangGraph node: retrieve relevant protocol chunks -> grounded Gemini answer."""
    try:
        import agent.nodes.rag_answerer as _self
        client = _self._get_client()
        query = state["current_input"]
        profile = state.get("patient_profile", {})
        name = profile.get("name", "Friend")

        context_chunks = retrieve_context(query, n_results=5)
        if not context_chunks:
            return {
                "rag_context": "",
                "response": (
                    "I don't have specific guidance on that in your current protocol. "
                    "Please consult your healthcare provider for personalised advice."
                ),
                "error": "",
            }

        context_text = "\n\n---\n\n".join(context_chunks)
        result = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=_SYSTEM_PROMPT.format(context=context_text, name=name, question=query),
        )

        return {
            "rag_context": context_text,
            "response": result.text.strip(),
            "error": "",
        }

    except Exception as e:
        logger.exception("rag_answerer_node failed")
        return {
            "rag_context": "",
            "response": "I'm having trouble accessing the protocol right now. Please try again.",
            "error": str(e),
        }
