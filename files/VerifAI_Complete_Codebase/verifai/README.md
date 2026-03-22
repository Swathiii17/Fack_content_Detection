# VerifAI — Agentic Fake Content Detection

A multi-agent AI system for journalists and researchers to detect fake news, fake reviews, manipulated images, and audio deepfakes.

---

## Architecture

```
User Input
    │
    ▼
Orchestrator Agent          ← Decides which agents to run
    │
    ├──────────────────────────────────────┐
    ▼              ▼            ▼           ▼
News Agent    Review Agent  Image Agent  Audio Agent
(LLM + web   (LLM + NLP    (Vision +    (Librosa +
 search)       heuristics)   ELA + EXIF)  LLM)
    │              │            │           │
    └──────────────┴────────────┴───────────┘
                        │
                        ▼
              Verdict Aggregator
                        │
                        ▼
              Final Report (verdict + confidence + signals)
```

**Tech stack:**
- **LangGraph** — agent orchestration graph
- **FastAPI** — async REST API
- **Claude claude-sonnet-4-6** — LLM for all agents
- **Tavily** — web search for fact-checking
- **Pillow + ExifRead** — image ELA analysis + metadata
- **Librosa** — audio spectral feature extraction
- **SQLite / PostgreSQL** — analysis storage
- **React + Vite** — frontend dashboard

---

## Quickstart

### 1. Clone and configure

```bash
cd verifai/backend
cp .env.example .env
```

Edit `.env`:
```
ANTHROPIC_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here     # optional but recommended for news fact-checking
DATABASE_URL=sqlite+aiosqlite:///./verifai.db
```

Get your keys:
- Anthropic: https://console.anthropic.com
- Tavily: https://tavily.com (free tier available)

### 2. Run the backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Backend runs at: http://localhost:8000
API docs at: http://localhost:8000/docs

### 3. Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at: http://localhost:5173

### 4. (Optional) Docker Compose

```bash
# From the root verifai/ directory
cp backend/.env.example backend/.env
# Edit backend/.env with your API keys
docker-compose up --build
```

---

## API Reference

### POST /api/analyse/text
Analyse news articles or reviews.

```bash
curl -X POST http://localhost:8000/api/analyse/text \
  -F "text=Scientists discover cure for all diseases" \
  -F "content_type=news"
```

### POST /api/analyse/image
Analyse an image for forgery.

```bash
curl -X POST http://localhost:8000/api/analyse/image \
  -F "file=@suspicious_photo.jpg"
```

### POST /api/analyse/audio
Analyse audio for deepfake.

```bash
curl -X POST http://localhost:8000/api/analyse/audio \
  -F "file=@voice_clip.mp3"
```

### GET /api/analyses
List recent analyses.

### GET /api/analyses/{id}
Get full analysis with agent details.

---

## Agent Details

### Orchestrator Agent
Uses Claude to route content to the right specialist agents. Handles mixed content (e.g. a URL could be news or contain an image).

### News Agent
1. Extracts the main factual claim using LLM
2. Searches the web via Tavily for corroborating/contradicting sources
3. Analyses source credibility, claim consistency, language patterns
4. Returns: `source_score`, `claim_score`, `language_score`

### Review Agent
1. Extracts heuristic features (superlative density, specificity, sentence length)
2. LLM analyses for bot patterns, template reuse, sentiment anomalies
3. Returns: `specificity_score`, `sentiment_score`, `ai_generated_probability`

### Image Agent
1. **ELA (Error Level Analysis)** — detects re-saved/edited regions
2. **EXIF metadata** — checks camera info, software, GPS, timestamps
3. **Claude Vision** — visual inspection for inconsistencies, AI generation signs
4. Returns: `visual_consistency_score`, `ai_generation_probability`, `ela_score`

### Audio Agent
1. **Librosa spectral analysis** — MFCCs, pitch, ZCR, RMS energy, silence ratio
2. LLM interprets features for deepfake patterns
3. Returns: `deepfake_probability`, `voice_consistency_score`, `spectral_anomaly_score`

### Verdict Aggregator
- Weighted scoring across all agents
- LLM synthesises conflicting signals
- Final output: `verdict`, `confidence`, `reasoning`, `signals`

**Verdict scale:**
- `fake` — strong evidence of fabrication
- `likely_fake` — multiple suspicious signals
- `uncertain` — inconclusive
- `likely_real` — mostly credible signals
- `real` — strong authenticity signals

---

## Project Structure

```
verifai/
├── backend/
│   ├── main.py                  # FastAPI app entry point
│   ├── config.py                # Settings
│   ├── pipeline.py              # LangGraph graph definition
│   ├── requirements.txt
│   ├── .env.example
│   ├── agents/
│   │   ├── orchestrator.py      # Routes to specialist agents
│   │   ├── news_agent.py        # Fake news detection
│   │   ├── review_agent.py      # Fake review detection
│   │   ├── image_agent.py       # Image forgery detection
│   │   ├── audio_agent.py       # Audio deepfake detection
│   │   └── verdict_aggregator.py
│   ├── api/
│   │   └── routes.py            # FastAPI endpoints
│   └── models/
│       ├── state.py             # LangGraph state schema
│       └── database.py          # SQLAlchemy models
└── frontend/
    ├── index.html
    ├── vite.config.js
    ├── package.json
    └── src/
        ├── main.jsx
        ├── App.jsx              # Main UI component
        ├── App.css
        └── api.js               # API client
```

---

## Extending VerifAI

### Add a new agent
1. Create `backend/agents/my_agent.py` with an async `my_agent_node(state)` function
2. Add the node to `pipeline.py` in `build_graph()`
3. Add the result key to `AnalysisState` in `models/state.py`
4. Include the result in `verdict_aggregator.py`

### Add video detection
- Use `ffmpeg` to extract frames → run through image agent
- Extract audio track → run through audio agent
- Combine both results in aggregator

### Improve news fact-checking
- Integrate ClaimBuster API for claim extraction
- Add GDELT for news source reputation scoring
- Add Google Fact Check Tools API

---

## Limitations

- Audio deepfake detection is heuristic-based — purpose-built models (e.g. Wav2Vec2 fine-tuned on deepfake datasets) would be significantly more accurate
- Image ELA works best on JPEG — PNG/WebP results are less reliable
- News fact-checking quality depends on Tavily search results and Claude's knowledge cutoff
- No video support yet (Phase 3 feature)

---

## License

MIT
