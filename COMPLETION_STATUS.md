# 📊 PaperGuard — Completion Status

> **⚠️ UPDATE THIS FILE BEFORE EVERY PUSH.**
> This is the single source of truth for where the project stands.

**Last Updated:** 2026-07-07
**Updated By:** merge — @vediumsameer (local model training) + @monishreddy (Phase 2 agents)

---

## 👥 Team Responsibilities

- **@vediumsameer**: Local AI Detection Model Training (Running on local RTX 3050).
- **@monishreddy**: Everything else (Agent Architecture, API Integrations, Streamlit UI, Qwen Cloud).

---

## Overall Progress

| Phase                                              |    Status    | Notes                                                    |
| :------------------------------------------------- | :----------: | :------------------------------------------------------- |
| Phase 1: Core Infrastructure + API Layer + Caching |  ✅ Complete | Finished core, models, and services                      |
| Phase 1.5: Local Model Training                    | 🟡 In Progress | **@vediumsameer** running multi-dataset aggregation locally |
| Phase 2: All Agents Working Independently          |  ✅ Complete | **@monishreddy** — all 4 agents built + verified         |
| Phase 3: Custom Orchestrator + Conflict Resolution | ⬜ Not Started | **@monishreddy**                                        |
| Phase 4: Streamlit UI                              | ⬜ Not Started | **@monishreddy**                                        |
| Phase 5: Polish & Deploy                           | ⬜ Not Started | **@monishreddy**                                        |

**Legend:** ✅ Complete | 🟡 In Progress | ⬜ Not Started | ❌ Blocked

---

## Phase 1: Core Infrastructure (Complete)

### Person 1 Tasks

| Task                                             | File                       | Status | Notes                                           |
| :----------------------------------------------- | :------------------------- | :----: | :---------------------------------------------- |
| PDF text extraction (pymupdf4llm)                | `core/pdf_parser.py`       |   ✅   | Completed via pymupdf4llm                       |
| Text chunker (sections → paragraphs → sentences) | `core/text_chunker.py`     |   ✅   | Sections, paragraphs, sentences splitting added |
| Reference parser (LLM-based)                     | `core/reference_parser.py` |   ✅   | Basic parsing using LLM or regex fallback       |
| AWS Textract fallback for scanned PDFs           | `core/pdf_parser.py`       |   ⬜   | Optional                                        |

### Person 2 Tasks

| Task                          | File                           | Status | Notes |
| :---------------------------- | :----------------------------- | :----: | :---- |
| Gemini 3.1 Flash Lite wrapper | `services/gemini.py`           |   ✅   | Done  |
| Semantic Scholar API wrapper  | `services/semantic_scholar.py` |   ✅   | Done  |
| CrossRef API wrapper          | `services/crossref.py`         |   ✅   | Done  |
| Serper web search wrapper     | `services/serper.py`           |   ✅   | Done  |
| File-based JSON cache         | `services/cache.py`            |   ✅   | Done  |

### Shared Tasks

| Task                    | File                   | Status | Notes                                                                                         |
| :---------------------- | :--------------------- | :----: | :-------------------------------------------------------------------------------------------- |
| Report data model       | `models/report.py`     |   ✅   | Done                                                                                          |
| Reference data model    | `models/reference.py`  |   ✅   | Done                                                                                          |
| Project structure setup | Various                |   ✅   | Folders and inits created                                                                     |
| requirements.txt        | `requirements.txt`     |   ✅   | Done                                                                                          |
| .env.example            | `.env.example`         |   ✅   | Done                                                                                          |
| Sample test papers      | `tests/sample_papers/` |   🟡   | Added `sample_paper.md` fixture (cites [1]-[4], incl. 1 fabricated DOI); still need real PDFs |

### Phase 1 Checkpoint

- [ ] `python -m core.pdf_parser sample.pdf` → clean text output
- [ ] `python -m services.crossref 10.1038/xxx` → metadata returned
- [ ] Cache works: second API call returns cached result instantly
- [ ] Two-column PDF parses correctly (pymupdf4llm)

---

## Phase 1.5: Local Model Training (Task: @vediumsameer)

| Task                            | File                     | Status | Notes                                            |
| :------------------------------ | :----------------------- | :----: | :----------------------------------------------- |
| Defactify & Ateeqq Aggregation  | `train_multi_dataset.py` |   🟡   | Running locally (125k samples)                   |
| DistilBERT Fine-tuning          | `train_multi_dataset.py` |   ⬜   | Pending dataset aggregation                      |
| Push to HF Hub                  | `train_multi_dataset.py` |   ⬜   | Target: `vediumsameer/paperguard-ai-detector`    |

---

## Phase 2: All Agents Working Independently (Tasks: @monishreddy)

### Priority Build Order

| Priority | Agent                          | File                           | Status | Notes                                                              |
| :------: | :----------------------------- | :----------------------------- | :----: | :----------------------------------------------------------------- |
|   1st    | Citation Verification Agent ⭐ | `agents/citation_agent.py`     |   ✅   | Existence (CrossRef+S2) + LLM claim check; 4-tier + pattern flag   |
|   2nd    | Writing Quality Agent          | `agents/quality_agent.py`      |   ✅   | Structure + readability (math) + per-section LLM prose review      |
|   3rd    | AI Detection Agent             | `agents/ai_detection_agent.py` |   ✅   | LLM classifier (0.7) + burstiness (0.3); per-section, weighted     |
|   4th    | Plagiarism Agent               | `agents/plagiarism_agent.py`   |   ✅   | Serper web + CrossRef→S2 scholarly + LLM similarity; honest limits |

> **Shared foundation:** `agents/base.py` (lazy PDF/LLM imports, `.pdf/.md/.txt`
> loading, section/reference helpers, `BaseAgent`, and a common `run_cli`
> harness) + `agents/__init__.py`. All agents return a `models.report.AgentResult`
> and degrade gracefully when API keys are absent.

### Phase 2 Checkpoint

- [x] `python -m agents.citation_agent sample` → JSON output (verified on `.md`; live CrossRef resolves real DOIs and flags a fabricated DOI)
- [x] `python -m agents.quality_agent sample` → JSON output (verified via CLI, rc=0)
- [x] `python -m agents.ai_detection_agent sample` → JSON output (verified via CLI, rc=0)
- [x] `python -m agents.plagiarism_agent sample` → JSON output (run()/JSON verified with stubbed services; scholarly path needs no key)

> **Verification notes**
>
> - Verified against `tests/sample_papers/sample_paper.md` in two modes:
>   degraded (no API keys → pure-math/structure + existence-only) and
>   LLM-enabled (stubbed Gemini) — all four emit valid `AgentResult` JSON.
> - CLIs also accept `.md`/`.txt` (not just `.pdf`) so agents are testable
>   without the PDF stack or API keys.
> - Fixed a blocking bug in `core/reference_parser.py` (malformed `f\"\"\"`
>   f-string caused a `SyntaxError` on import — reference extraction was broken).
> - Live API runs need keys in `.env` (`GEMINI_API_KEY`, `SERPER_API_KEY`,
>   optional `SEMANTIC_SCHOLAR_API_KEY`). CrossRef needs none.

---

## Phase 3: Custom Orchestrator + Conflict Resolution (Tasks: @monishreddy)

| Task                  | File                          | Status | Notes |
| :-------------------- | :---------------------------- | :----: | :---- |
| Orchestrator pipeline | `agents/orchestrator.py`      |   ⬜   |       |
| Conflict resolver     | `agents/conflict_resolver.py` |   ⬜   |       |
| Report generator      | `models/report_generator.py`  |   ⬜   |       |
| End-to-end CLI test   | `main.py`                     |   ⬜   |       |

### Phase 3 Checkpoint

- [ ] `python main.py paper.pdf` → full JSON report with conflicts resolved
- [ ] Plagiarism + Citation conflict correctly resolved
- [ ] AI + Quality conflict correctly resolved

---

## Phase 4: Streamlit UI (Tasks: @monishreddy)

| Task                        | File     | Status | Notes                                    |
| :-------------------------- | :------- | :----: | :--------------------------------------- |
| Upload flow + progress      | `app.py` |   ⬜   |                                          |
| Report dashboard            | `app.py` |   ⬜   |                                          |
| Citation table view         | `app.py` |   ⬜   |                                          |
| AI detection breakdown      | `app.py` |   ⬜   | Connects to @vediumsameer's HF model API |
| PDF export                  | `app.py` |   ⬜   |                                          |
| Disclaimers                 | `app.py` |   ⬜   |                                          |
| `@st.cache_data` protection | `app.py` |   ⬜   |                                          |

### Phase 4 Checkpoint

- [ ] Upload PDF → see report in browser
- [ ] Download report as PDF
- [ ] `@st.cache_data` prevents re-analysis on reload

---

## Phase 5: Polish & Deploy

| Task                               | Status | Notes |
| :--------------------------------- | :----: | :---- |
| Test with 10+ real papers          |   ⬜   |       |
| Edge cases (no refs, short papers) |   ⬜   |       |
| Rate limiting + error handling     |   ⬜   |       |
| Deploy to Streamlit Cloud          |   ⬜   |       |
| AWS S3 setup (if needed)           |   ⬜   |       |
| README + docs                      |   ✅   | Done  |
| Billing alerts on AWS              |   ⬜   |       |

---

## Known Issues / Blockers

| Issue                                                     | Severity |  Status  | Notes                                                                                     |
| :-------------------------------------------------------- | :------: | :------: | :---------------------------------------------------------------------------------------- |
| `core/reference_parser.py` had malformed `f\"\"\"` string |   High   | ✅ Fixed | Caused `SyntaxError` on import; reference extraction was fully broken. Now fixed.         |
| No API keys configured yet                                |  Medium  | ⬜ Open  | Agents run in degraded mode without keys; add `.env` for LLM claim/AI/quality/plagiarism. |
| Semantic Scholar keyless endpoint is slow/unreliable      |   Low    | ⬜ Open  | Agents wrap calls in try/except → `None`. A free S2 key improves abstract retrieval.      |

---

## API Keys Status

| API                |   Key Obtained?    | Free Tier Limits                          |
| :----------------- | :----------------: | :---------------------------------------- |
| Gemini (AI Studio) |         ⬜         | 15–30 RPM, 250K–1M TPM                     |
| Qwen (DashScope)   |         ⬜         | Need to sign up for Hackathon free tier   |
| Semantic Scholar   |         ⬜         | 1 req/sec (free key)                      |
| CrossRef           | ✅ (no key needed) | Unlimited (polite pool with email)        |
| Serper             |         ⬜         | 2,500 credits/month                       |
