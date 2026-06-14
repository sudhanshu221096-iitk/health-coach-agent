# Health Coach Agent 🌿

An AI-powered personal health coach built as an agentic pipeline. It onboards patients, runs adaptive daily check-ins, and answers protocol questions using RAG — all grounded in a reference wellness document.

**Live demo:** `https://health-coach-agent.onrender.com/?session_id=demo`

---

## Features

| Feature | Description |
|---|---|
| **Patient Onboarding** | Parses free-form text → structured profile (Gemini extraction) |
| **Adaptive Check-ins** | Questions change by day (Day 1 intro → Day 7 reflection) |
| **Protocol Q&A (RAG)** | Answers grounded strictly in wellness PDF — no hallucination |
| **Session Memory** | Full conversation history maintained within a session |
| **Shareable URL** | `?session_id=<id>` resumes any session |

---

## Tech Stack & Rationale

| Layer | Choice | Why |
|---|---|---|
| **LLM** | Google Gemini 1.5 Flash | Fast, free tier, 1M context window |
| **Agent Framework** | LangGraph | Stateful graph with conditional routing; perfect for multi-node adaptive flow |
| **Embeddings** | Gemini `text-embedding-004` | Native ecosystem, asymmetric retrieval (doc vs query task types) |
| **Vector Store** | ChromaDB | Zero-config local persistence, cosine similarity |
| **PDF Parsing** | pypdf | Lightweight, no Java dependency |
| **Backend** | FastAPI | Async, Pydantic validation, automatic OpenAPI docs |
| **Frontend** | Vanilla HTML/CSS/JS | Zero dependencies, query-param shareable, instant load |
| **Deployment** | Render.com | Free Python web service, simple `render.yaml` config |

---

## Architecture

```
Browser (?session_id=xxx)
    │
    ▼
FastAPI  (/api/onboard · /api/checkin · /api/ask · /api/state)
    │
    ▼
LangGraph Agent Graph
    START → dispatcher ──┬── profile_parser  (mode=onboard)
                         ├── day_router      (mode=checkin)
                         └── rag_answerer    (mode=ask)
                                ↓
                         memory_updater → END
    │
    ▼
ChromaDB  (wellness_protocol.pdf chunks, cosine similarity)
```

---

## Quick Start

```bash
# 1. Clone & enter
git clone https://github.com/YOUR_USERNAME/health-coach-agent
cd health-coach-agent

# 2. Create virtual environment
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Gemini API key
cp .env.example .env
# Edit .env and add: GEMINI_API_KEY=your_key_here

# 5. Generate the wellness protocol PDF
python scripts/generate_pdf.py

# 6. Start the server (ingestion runs automatically on startup)
uvicorn api.main:app --reload

# 7. Open the app
# http://localhost:8000
```

---

## Running Tests

```bash
pytest -v
```

All tests mock the Gemini API — **no API key required** to run the test suite.

**Test coverage:**
- `test_profile_parser.py` — 5 checks (valid, partial, malformed JSON, empty input, welcome message)
- `test_day_router.py` — 10 checks (all day buckets, LLM integration, failure fallback)
- `test_rag.py` — 12 checks (chunking, retrieval, RAG node integration)
- `test_memory.py` — 13 checks (session store CRUD, memory_updater accumulation)
- `test_api.py` — 22 checks (all endpoints, 404s, validation errors)

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/onboard` | Parse patient profile, create session |
| `POST` | `/api/checkin` | Day-adaptive check-in question |
| `POST` | `/api/ask` | RAG-grounded protocol Q&A |
| `GET` | `/api/state/{session_id}` | Full session state & history |
| `GET` | `/health` | Health check |

Interactive docs: `http://localhost:8000/docs`

---

## Deploy to Render.com

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New → Web Service → Connect repo
3. Render auto-detects `render.yaml`
4. Set env var `GEMINI_API_KEY` in the Render dashboard
5. Deploy — the PDF is generated and ingested at build/startup time

---

## Project Structure

```
health-coach-agent/
├── agent/
│   ├── graph.py            # LangGraph StateGraph (compiled singleton)
│   ├── state.py            # AgentState TypedDict
│   └── nodes/
│       ├── profile_parser.py   # Unstructured text → structured profile
│       ├── day_router.py       # Day-adaptive check-in question
│       ├── rag_answerer.py     # RAG-grounded protocol Q&A
│       └── memory_updater.py   # Session history management
├── rag/
│   ├── embeddings.py       # Gemini ChromaDB embedding function
│   ├── ingest.py           # PDF → ChromaDB ingestion
│   └── retriever.py        # Semantic similarity retrieval
├── api/
│   ├── main.py             # FastAPI app + lifespan ingestion
│   ├── models.py           # Pydantic request/response models
│   ├── session_store.py    # Thread-safe in-memory sessions
│   └── routes/             # onboard / checkin / ask
├── frontend/
│   └── index.html          # Single-page chat UI
├── scripts/
│   └── generate_pdf.py     # Generates wellness_protocol.pdf
├── data/
│   └── wellness_protocol.pdf
├── tests/                  # 62 sanity checks
├── render.yaml
└── requirements.txt
```
