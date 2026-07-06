# 📊 PaperGuard — Completion Status

> **⚠️ UPDATE THIS FILE BEFORE EVERY PUSH.**
> This is the single source of truth for where the project stands.

**Last Updated:** 2026-07-06
**Updated By:** Initial setup

---

## Overall Progress

| Phase | Status | Notes |
|:---|:---:|:---|
| Phase 1: Core Infrastructure + API Layer + Caching | ✅ Complete | Finished core, models, and services |
| Phase 2: All Agents Working Independently | ⬜ Not Started | |
| Phase 3: Custom Orchestrator + Conflict Resolution | ⬜ Not Started | |
| Phase 4: Streamlit UI | ⬜ Not Started | |
| Phase 5: Polish & Deploy | ⬜ Not Started | |

**Legend:** ✅ Complete | 🟡 In Progress | ⬜ Not Started | ❌ Blocked

---

## Phase 1: Core Infrastructure + API Layer + Caching

### Person 1 Tasks

| Task | File | Status | Notes |
|:---|:---|:---:|:---|
| PDF text extraction (pymupdf4llm) | `core/pdf_parser.py` | ✅ | Completed via pymupdf4llm |
| Text chunker (sections → paragraphs → sentences) | `core/text_chunker.py` | ✅ | Sections, paragraphs, sentences splitting added |
| Reference parser (LLM-based) | `core/reference_parser.py` | ✅ | Basic parsing using LLM or regex fallback |
| AWS Textract fallback for scanned PDFs | `core/pdf_parser.py` | ⬜ | Optional |

### Person 2 Tasks

| Task | File | Status | Notes |
|:---|:---|:---:|:---|
| Gemini 3.1 Flash Lite wrapper | `services/gemini.py` | ✅ | Done |
| Semantic Scholar API wrapper | `services/semantic_scholar.py` | ✅ | Done |
| CrossRef API wrapper | `services/crossref.py` | ✅ | Done |
| Serper web search wrapper | `services/serper.py` | ✅ | Done |
| File-based JSON cache | `services/cache.py` | ✅ | Done |

### Shared Tasks

| Task | File | Status | Notes |
|:---|:---|:---:|:---|
| Report data model | `models/report.py` | ✅ | Done |
| Reference data model | `models/reference.py` | ✅ | Done |
| Project structure setup | Various | ✅ | Folders and inits created |
| requirements.txt | `requirements.txt` | ✅ | Done |
| .env.example | `.env.example` | ✅ | Done |
| Sample test papers | `tests/sample_papers/` | ⬜ | Need real PDFs |

### Phase 1 Checkpoint
- [ ] `python -m core.pdf_parser sample.pdf` → clean text output
- [ ] `python -m services.crossref 10.1038/xxx` → metadata returned
- [ ] Cache works: second API call returns cached result instantly
- [ ] Two-column PDF parses correctly (pymupdf4llm)

---

## Phase 2: All Agents Working Independently

### Priority Build Order

| Priority | Agent | File | Status | Notes |
|:---:|:---|:---|:---:|:---|
| 1st | Citation Verification Agent ⭐ | `agents/citation_agent.py` | ⬜ | Killer feature |
| 2nd | Writing Quality Agent | `agents/quality_agent.py` | ⬜ | Easiest |
| 3rd | AI Detection Agent | `agents/ai_detection_agent.py` | ⬜ | LLM classifier + burstiness |
| 4th | Plagiarism Agent | `agents/plagiarism_agent.py` | ⬜ | Build last |

### Phase 2 Checkpoint
- [ ] `python -m agents.citation_agent sample.pdf` → JSON output
- [ ] `python -m agents.quality_agent sample.pdf` → JSON output
- [ ] `python -m agents.ai_detection_agent sample.pdf` → JSON output
- [ ] `python -m agents.plagiarism_agent sample.pdf` → JSON output

---

## Phase 3: Custom Orchestrator + Conflict Resolution

| Task | File | Status | Notes |
|:---|:---|:---:|:---|
| Orchestrator pipeline | `agents/orchestrator.py` | ⬜ | |
| Conflict resolver | `agents/conflict_resolver.py` | ⬜ | |
| Report generator | `models/report_generator.py` | ⬜ | |
| End-to-end CLI test | `main.py` | ⬜ | |

### Phase 3 Checkpoint
- [ ] `python main.py paper.pdf` → full JSON report with conflicts resolved
- [ ] Plagiarism + Citation conflict correctly resolved
- [ ] AI + Quality conflict correctly resolved

---

## Phase 4: Streamlit UI

| Task | File | Status | Notes |
|:---|:---|:---:|:---|
| Upload flow + progress | `app.py` | ⬜ | |
| Report dashboard | `app.py` | ⬜ | |
| Citation table view | `app.py` | ⬜ | |
| AI detection breakdown | `app.py` | ⬜ | |
| PDF export | `app.py` | ⬜ | |
| Disclaimers | `app.py` | ⬜ | |
| `@st.cache_data` protection | `app.py` | ⬜ | |

### Phase 4 Checkpoint
- [ ] Upload PDF → see report in browser
- [ ] Download report as PDF
- [ ] `@st.cache_data` prevents re-analysis on reload

---

## Phase 5: Polish & Deploy

| Task | Status | Notes |
|:---|:---:|:---|
| Test with 10+ real papers | ⬜ | |
| Edge cases (no refs, short papers) | ⬜ | |
| Rate limiting + error handling | ⬜ | |
| Deploy to Streamlit Cloud | ⬜ | |
| AWS S3 setup (if needed) | ⬜ | |
| README + docs | ✅ | Done |
| Billing alerts on AWS | ⬜ | |

---

## Known Issues / Blockers

| Issue | Severity | Status | Notes |
|:---|:---:|:---:|:---|
| (none yet) | | | |

---

## API Keys Status

| API | Key Obtained? | Free Tier Limits |
|:---|:---:|:---|
| Gemini (AI Studio) | ⬜ | 15–30 RPM, 250K–1M TPM |
| Semantic Scholar | ⬜ | 1 req/sec (free key) |
| CrossRef | ✅ (no key needed) | Unlimited (polite pool with email) |
| Serper | ⬜ | 2,500 credits/month |
