"""
Single-file Health Coach Agent for Render deployment.
All modules merged to avoid Python 3.14 package import issues.
"""
from __future__ import annotations

import json
import logging
import operator
import os
import random
import re
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

# ── Patch sqlite3 before chromadb loads ──────────────────────────────────────
try:
    import pysqlite3
    import sys
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

import chromadb
import google.genai as genai
from chromadb import EmbeddingFunction, Embeddings
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from pypdf import PdfReader
from typing_extensions import TypedDict

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ── State ─────────────────────────────────────────────────────────────────────

class HistoryTurn(TypedDict):
    role: str
    content: str
    day: int
    mode: str

class AgentState(TypedDict):
    session_id: str
    mode: str
    patient_profile: Dict[str, Any]
    day_number: int
    session_history: Annotated[List[HistoryTurn], operator.add]
    current_input: str
    rag_context: str
    response: str
    error: str

# ── Pydantic models ───────────────────────────────────────────────────────────

class OnboardRequest(BaseModel):
    session_id: Optional[str] = Field(None)
    profile_text: str = Field(..., min_length=10)

class CheckInRequest(BaseModel):
    session_id: str
    user_response: Optional[str] = Field(None)
    day_number: int = Field(1, ge=1)

class AskRequest(BaseModel):
    session_id: str
    question: str = Field(..., min_length=3)

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

# ── Session Store ─────────────────────────────────────────────────────────────

def _empty_state(session_id: str) -> AgentState:
    return AgentState(
        session_id=session_id, mode="onboard", patient_profile={},
        day_number=1, session_history=[], current_input="",
        rag_context="", response="", error="",
    )

class SessionStore:
    def __init__(self):
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

    def update(self, session_id: str, new_state: AgentState):
        with self._lock:
            self._store[session_id] = new_state

    def exists(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._store

    def __len__(self):
        with self._lock:
            return len(self._store)

session_store = SessionStore()

# ── Embeddings ────────────────────────────────────────────────────────────────

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model="models/gemini-embedding-001", task_type="RETRIEVAL_DOCUMENT"):
        self.model = model
        self.task_type = task_type
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def __call__(self, input: List[str]) -> Embeddings:
        embeddings = []
        for text in input:
            result = self._client.models.embed_content(
                model=self.model, contents=text,
                config={"task_type": self.task_type},
            )
            embeddings.append(result.embeddings[0].values)
        return embeddings

class GeminiQueryEmbeddingFunction(GeminiEmbeddingFunction):
    def __init__(self, model="models/gemini-embedding-001"):
        super().__init__(model=model, task_type="RETRIEVAL_QUERY")

# ── RAG ───────────────────────────────────────────────────────────────────────

COLLECTION_NAME = "wellness_protocol"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

def get_chroma_client():
    db_path = os.environ.get("CHROMA_DB_PATH", "./data/chroma_db")
    Path(db_path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=db_path)

def _load_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)

def _chunk_text(text: str) -> List[str]:
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start:start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def ingest_documents(force: bool = False) -> int:
    pdf_path = os.environ.get("WELLNESS_PDF_PATH", "./data/wellness_protocol.pdf")
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"Wellness protocol PDF not found: {pdf_path}")
    client = get_chroma_client()
    embed_fn = GeminiEmbeddingFunction()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    if not force and collection.count() > 0:
        return collection.count()
    text = _load_pdf_text(pdf_path)
    chunks = _chunk_text(text)
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)
    return len(chunks)

_query_embed_fn = None

def retrieve_context(query: str, n_results: int = 5) -> List[str]:
    global _query_embed_fn
    if not query.strip():
        return []
    try:
        if _query_embed_fn is None:
            _query_embed_fn = GeminiQueryEmbeddingFunction()
        client = get_chroma_client()
        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=_query_embed_fn)
        if collection.count() == 0:
            return []
        query_embedding = _query_embed_fn([query])[0]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, collection.count()),
        )
        return [doc for doc in results.get("documents", [[]])[0] if doc]
    except Exception as e:
        logger.exception("retrieve_context failed: %s", e)
        return []

# ── Agent Nodes ───────────────────────────────────────────────────────────────

_EXTRACTION_PROMPT = """\
You are a patient onboarding assistant. Extract structured information from the patient's text below.

Patient text:
\"\"\"{text}\"\"\"

Return ONLY a valid JSON object with these exact keys (use null for missing fields):
{{
  "name": "<string, default 'Friend' if not mentioned>",
  "age": <integer or null>,
  "primary_goals": ["<goal1>", "<goal2>"],
  "sleep_hours": <float or null>,
  "sleep_quality": "<poor|fair|good|excellent or null>",
  "activity_level": "<sedentary|lightly_active|moderately_active|very_active or null>",
  "dietary_restrictions": ["<item>"],
  "health_conditions": ["<condition>"],
  "motivation": "<one sentence describing their core motivation>"
}}

No markdown fences, no explanation -- raw JSON only.
"""

_CONFIRMATION_PROMPT = """\
You are a warm, encouraging health coach. A patient just shared their profile:
{profile_json}

Write a 2-3 sentence warm welcome message that:
1. Addresses them by name
2. Reflects back their primary goal(s)
3. Expresses genuine enthusiasm for supporting them

Tone: warm, clear, not clinical, not fluffy. No bullet points.
"""

_RAG_SYSTEM_PROMPT = """\
You are a wellness protocol specialist and health coach.
Answer the patient's question ONLY using the protocol excerpts provided below.

Rules:
1. If the answer is clearly in the context, give a warm, clear, helpful response.
2. If the answer is NOT in the context, say: "I don't have specific guidance on that in your current protocol. Please consult your healthcare provider for personalised advice."
3. Never invent facts not present in the context.
4. Keep your answer concise -- 2-4 sentences.
5. Tone: warm, clear, not clinical.

Protocol excerpts:
{context}

Patient's name: {name}
Patient's question: {question}
"""

_DAY_TEMPLATES = {
    "day_1": ["How are you feeling right now, physically and emotionally?", "What's your biggest hope for this wellness journey?", "On a scale of 1-10, how would you rate your energy levels today?"],
    "day_2_3": ["How did Day 1 go? What was easier than expected -- and what was harder?", "Were you able to follow any of yesterday's goals? Tell me about it.", "What's one small win from yesterday you can feel good about?"],
    "day_4_5": ["You're nearly halfway through the first week! How are your energy and mood evolving?", "Which habit has started to feel a little more automatic?", "Is there anything from the protocol you'd like to revisit?"],
    "day_6_7": ["You've made it almost a full week -- how does that feel?", "What surprised you most about yourself during this first week?", "Which habit made the biggest difference, and which needs more work?"],
    "day_8_plus": ["How are things going as you continue your journey?", "What feels sustainable long-term, and what still feels like a stretch?", "What's one thing you'd like to focus on in the days ahead?"],
}

_PERSONALISE_PROMPT = """\
You are a warm, empathetic health coach conducting a daily check-in.

Patient profile:
- Name: {name}
- Primary goals: {goals}
- Activity level: {activity}
- Sleep quality: {sleep}
- Motivation: {motivation}

Today is Day {day} of their protocol.
Base question to personalise: "{question}"
Previous responses: {history}

Rewrite the base question to feel personal. Keep it to 1-2 sentences. Warm, conversational. One focused question only.
"""

def _gemini_client():
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def profile_parser_node(state: AgentState) -> dict:
    try:
        client = _gemini_client()
        extraction = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=_EXTRACTION_PROMPT.format(text=state["current_input"]),
        )
        raw = re.sub(r"^```(?:json)?\s*", "", extraction.text.strip())
        raw = re.sub(r"\s*```$", "", raw)
        profile = json.loads(raw)
        confirmation = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=_CONFIRMATION_PROMPT.format(profile_json=json.dumps(profile, indent=2)),
        )
        return {"patient_profile": profile, "response": confirmation.text.strip(), "error": ""}
    except Exception as e:
        logger.exception("profile_parser_node failed")
        return {"patient_profile": {"name": "Friend", "primary_goals": [], "motivation": ""}, "response": "Thanks for sharing! I'm here to support your wellness journey.", "error": str(e)}

def day_router_node(state: AgentState) -> dict:
    try:
        client = _gemini_client()
        day = state.get("day_number", 1)
        profile = state.get("patient_profile", {})
        history = state.get("session_history", [])
        if day == 1: bucket = "day_1"
        elif day <= 3: bucket = "day_2_3"
        elif day <= 5: bucket = "day_4_5"
        elif day <= 7: bucket = "day_6_7"
        else: bucket = "day_8_plus"
        base_question = random.choice(_DAY_TEMPLATES[bucket])
        recent = "\n".join(f"  [{t['mode']} Day {t['day']}] {t['role']}: {t['content']}" for t in history[-3:]) or "  (no prior check-ins yet)"
        result = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=_PERSONALISE_PROMPT.format(
                name=profile.get("name", "Friend"),
                goals=", ".join(profile.get("primary_goals", [])) or "general wellness",
                activity=profile.get("activity_level", "unknown"),
                sleep=profile.get("sleep_quality", "unknown"),
                motivation=profile.get("motivation", "improving health"),
                day=day, question=base_question, history=recent,
            ),
        )
        return {"response": result.text.strip(), "error": ""}
    except Exception as e:
        logger.exception("day_router_node failed")
        return {"response": f"Day {state.get('day_number', 1)} check-in: How are you feeling today?", "error": str(e)}

def rag_answerer_node(state: AgentState) -> dict:
    try:
        client = _gemini_client()
        query = state["current_input"]
        name = state.get("patient_profile", {}).get("name", "Friend")
        context_chunks = retrieve_context(query, n_results=5)
        if not context_chunks:
            return {"rag_context": "", "response": "I don't have specific guidance on that in your current protocol. Please consult your healthcare provider for personalised advice.", "error": ""}
        context_text = "\n\n---\n\n".join(context_chunks)
        result = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=_RAG_SYSTEM_PROMPT.format(context=context_text, name=name, question=query),
        )
        return {"rag_context": context_text, "response": result.text.strip(), "error": ""}
    except Exception as e:
        logger.exception("rag_answerer_node failed")
        return {"rag_context": "", "response": "I'm having trouble accessing the protocol right now. Please try again.", "error": str(e)}

def memory_updater_node(state: AgentState) -> dict:
    day = state.get("day_number", 1)
    mode = state.get("mode", "ask")
    new_turns = []
    if state.get("current_input"):
        new_turns.append(HistoryTurn(role="user", content=state["current_input"], day=day, mode=mode))
    if state.get("response"):
        new_turns.append(HistoryTurn(role="agent", content=state["response"], day=day, mode=mode))
    return {"session_history": new_turns}

# ── Agent Graph ───────────────────────────────────────────────────────────────

def _dispatcher(state: AgentState) -> AgentState:
    return {}

def _route_by_mode(state: AgentState) -> str:
    mode = state.get("mode", "ask")
    if mode == "onboard": return "profile_parser"
    elif mode == "checkin": return "day_router"
    return "rag_answerer"

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("dispatcher", _dispatcher)
    g.add_node("profile_parser", profile_parser_node)
    g.add_node("day_router", day_router_node)
    g.add_node("rag_answerer", rag_answerer_node)
    g.add_node("memory_updater", memory_updater_node)
    g.add_edge(START, "dispatcher")
    g.add_conditional_edges("dispatcher", _route_by_mode, {"profile_parser": "profile_parser", "day_router": "day_router", "rag_answerer": "rag_answerer"})
    for node in ("profile_parser", "day_router", "rag_answerer"):
        g.add_edge(node, "memory_updater")
    g.add_edge("memory_updater", END)
    return g.compile()

agent_graph = build_graph()

# ── FastAPI App ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        n = ingest_documents()
        logger.info("RAG ready — %d protocol chunks in ChromaDB.", n)
    except FileNotFoundError:
        logger.warning("Wellness PDF not found.")
    except Exception as e:
        logger.error("RAG ingestion failed: %s", e)
    yield

app = FastAPI(title="Health Coach Agent", version="1.0.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/api/onboard", response_model=AgentResponse, tags=["Agent"])
async def onboard(req: OnboardRequest):
    sid = session_store.create(req.session_id)
    state = session_store.get(sid)
    state["mode"] = "onboard"
    state["current_input"] = req.profile_text
    result = agent_graph.invoke(state)
    session_store.update(sid, result)
    return AgentResponse(session_id=sid, response=result["response"], mode="onboard", day_number=result.get("day_number", 1), patient_name=result.get("patient_profile", {}).get("name"), error=result.get("error") or None)

@app.post("/api/checkin", response_model=AgentResponse, tags=["Agent"])
async def checkin(req: CheckInRequest):
    if not session_store.exists(req.session_id):
        raise HTTPException(status_code=404, detail="Session not found. Please onboard first.")
    state = session_store.get(req.session_id)
    if req.user_response:
        state["session_history"].append(HistoryTurn(role="user", content=req.user_response, day=req.day_number, mode="checkin"))
    state["mode"] = "checkin"
    state["day_number"] = req.day_number
    state["current_input"] = req.user_response or ""
    result = agent_graph.invoke(state)
    session_store.update(req.session_id, result)
    return AgentResponse(session_id=req.session_id, response=result["response"], mode="checkin", day_number=req.day_number, patient_name=result.get("patient_profile", {}).get("name"), error=result.get("error") or None)

@app.post("/api/ask", response_model=AgentResponse, tags=["Agent"])
async def ask(req: AskRequest):
    if not session_store.exists(req.session_id):
        raise HTTPException(status_code=404, detail="Session not found. Please onboard first.")
    state = session_store.get(req.session_id)
    state["mode"] = "ask"
    state["current_input"] = req.question
    result = agent_graph.invoke(state)
    session_store.update(req.session_id, result)
    return AgentResponse(session_id=req.session_id, response=result["response"], mode="ask", day_number=result.get("day_number", 1), patient_name=result.get("patient_profile", {}).get("name"), error=result.get("error") or None)

@app.get("/api/state/{session_id}", response_model=SessionStateResponse, tags=["Session"])
async def get_state(session_id: str):
    state = session_store.get(session_id)
    if state is None:
        return JSONResponse(status_code=404, content={"detail": "Session not found."})
    return SessionStateResponse(session_id=session_id, patient_profile=state.get("patient_profile", {}), day_number=state.get("day_number", 1), history_length=len(state.get("session_history", [])), history=state.get("session_history", []))

@app.get("/health", tags=["Meta"])
async def health():
    return {"status": "ok", "sessions": len(session_store)}

_frontend = Path(__file__).parent / "frontend"
if _frontend.exists():
    app.mount("/static", StaticFiles(directory=str(_frontend)), name="static")

    @app.get("/", include_in_schema=False)
    async def root():
        return FileResponse(str(_frontend / "index.html"))
