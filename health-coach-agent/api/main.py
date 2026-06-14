"""
api/main.py — FastAPI application entrypoint.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.models import SessionStateResponse
from api.routes.ask import router as ask_router
from api.routes.checkin import router as checkin_router
from api.routes.onboard import router as onboard_router
from api.session_store import session_store

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ingest the wellness protocol PDF into ChromaDB on startup if needed."""
    try:
        from rag.ingest import ingest_documents
        n = ingest_documents()
        logger.info("RAG ready — %d protocol chunks in ChromaDB.", n)
    except FileNotFoundError:
        logger.warning("Wellness PDF not found — run scripts/generate_pdf.py first.")
    except Exception as e:
        logger.error("RAG ingestion failed: %s", e)
    yield


app = FastAPI(
    title="Health Coach Agent",
    description="An AI health coach that personalises daily check-ins and answers protocol questions.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(onboard_router, prefix="/api")
app.include_router(checkin_router, prefix="/api")
app.include_router(ask_router,     prefix="/api")


# ── Session state endpoint ────────────────────────────────────────────────────
@app.get("/api/state/{session_id}", response_model=SessionStateResponse, tags=["Session"])
async def get_state(session_id: str):
    """Retrieve the full state (profile + history) for a session."""
    state = session_store.get(session_id)
    if state is None:
        return JSONResponse(status_code=404, content={"detail": "Session not found."})
    return SessionStateResponse(
        session_id=session_id,
        patient_profile=state.get("patient_profile", {}),
        day_number=state.get("day_number", 1),
        history_length=len(state.get("session_history", [])),
        history=state.get("session_history", []),
    )


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["Meta"])
async def health():
    return {"status": "ok", "sessions": len(session_store)}


# ── Frontend ──────────────────────────────────────────────────────────────────
_frontend = Path(__file__).parent.parent / "frontend"
if _frontend.exists():
    app.mount("/static", StaticFiles(directory=str(_frontend)), name="static")

    @app.get("/", include_in_schema=False)
    async def root():
        return FileResponse(str(_frontend / "index.html"))
