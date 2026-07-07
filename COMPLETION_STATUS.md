# PaperGuard — Completion Status

> **UPDATE THIS FILE BEFORE EVERY PUSH.** Single source of truth for project state.

**Last Updated:** 2026-07-07
**Updated By:** Safety Net architecture (Detector + Linguistic agents) + CrewAI decision + repo cleanup

---

## Scope

PaperGuard targets academic integrity verification for **both** student work
**and** research-paper submissions (IEEE, ACM, Elsevier, etc.). Language and
framing throughout is research-grade, not student-only.

## Team

- **@vediumsameer**: Local AI-detection model training (RTX 3050). v2.0 mega
  model trained and pushed to `vediumsameer/paperguard-ai-detector`.
- **@monishreddy**: Agents, API integrations, orchestration, Streamlit UI.

---

## Key Architectural Decisions (current)

1. **Orchestration = CrewAI** (agents-as-tools pattern). Supersedes the earlier
   "custom orchestrator, no framework" note in `IMPLEMENTATION_PLAN.md`.
   Deterministic work (API lookups, PyTorch inference, math) stays as tools;
   the LLM does reasoning/synthesis/conflict-resolution only.
2. **AI-Detection Safety Net.** PyTorch Detector (math) + LLM Linguistic agent
   (context) → Conflict Resolver overrides the model's mode-collapse mistakes.
3. **Detector model** = `vediumsameer/paperguard-ai-detector` (v2.0 mega
   weights), loaded locally via `transformers`. Label index 0 = AI, 1 = Human.
4. **Reasoning LLM** = Gemini (free tier), pluggable so Qwen can drop in later.

---

## Overall Progress

| Phase | Status | Notes |
| :--- | :---: | :--- |
| Phase 1: Core + services + caching | ✅ Complete | pdf/text/reference + CrossRef/S2/Serper/Gemini + cache |
| Phase 1.5: Local AI-detection model | ✅ Complete | v2.0 mega model trained + on HF Hub |
| Phase 2: Four base agents (standalone) | ✅ Complete | citation / quality / ai_detection / plagiarism |
| Phase 2.5: AI-Detection Safety Net | 🟡 In Progress | Detector + Linguistic built; Conflict Resolver drafted, wiring next |
| Phase 3: CrewAI orchestrator + conflict resolution | 🟡 In Progress | Design set (CrewAI); build pending |
| Phase 4: Streamlit UI | ⬜ Not Started | |
| Phase 5: Polish & deploy | ⬜ Not Started | |

**Legend:** ✅ Complete · 🟡 In Progress · ⬜ Not Started · ❌ Blocked

---

## Phase 1 — Core Infrastructure (Complete)

| Task | File | Status |
| :--- | :--- | :---: |
| PDF extraction (pymupdf4llm) | `core/pdf_parser.py` | ✅ |
| Text chunker (sections→paragraphs→sentences) | `core/text_chunker.py` | ✅ |
| Reference parser (LLM + heuristic) | `core/reference_parser.py` | ✅ |
| Gemini wrapper | `services/gemini.py` | ✅ |
| CrossRef / Semantic Scholar / Serper wrappers | `services/*.py` | ✅ |
| File-based JSON cache | `services/cache.py` | ✅ |
| Data models (report, reference) | `models/*.py` | ✅ |

---

## Phase 1.5 — Local AI-Detection Model (Complete, @vediumsameer)

| Task | File | Status | Notes |
| :--- | :--- | :---: | :--- |
| Multi/mega-dataset training pipeline | `train_mega_dataset.py` | ✅ | Opus distill + Ateeqq + AI-text-detection-pile |
| Fine-tuned DistilBERT (v2.0) | — | ✅ | eval_loss ~0.0003 |
| Push to HF Hub | — | ✅ | `vediumsameer/paperguard-ai-detector` |
| OOD validation gauntlet | `ood_stress_test.py` | ✅ | Exposed mode collapse → motivated the Safety Net |

> The OOD stress test showed the raw model collapses on ESL and style-masked
> text. This is the reason the Detector is paired with the Linguistic agent.

---

## Phase 2 — Base Agents (Complete, @monishreddy)

| Agent | File | Status |
| :--- | :--- | :---: |
| Citation Verification (existence + claim, 4-tier) | `agents/citation_agent.py` | ✅ |
| Writing Quality (structure + readability + prose) | `agents/quality_agent.py` | ✅ |
| AI Detection (LLM classifier + burstiness) | `agents/ai_detection_agent.py` | ✅ |
| Plagiarism (Serper + scholarly + LLM similarity) | `agents/plagiarism_agent.py` | ✅ |

Shared foundation: `agents/base.py` (lazy imports, `.pdf/.md/.txt` loading,
section/reference helpers, `BaseAgent`, `run_cli`). All agents degrade
gracefully without API keys.

---

## Phase 2.5 — AI-Detection Safety Net (In Progress)

| Component | File | Status | Notes |
| :--- | :--- | :---: | :--- |
| Detector Agent (PyTorch, per-paragraph) | `agents/detector_agent.py` | ✅ | Lazy model load, graceful degradation |
| Linguistic Agent (LLM, per-paragraph) | `agents/linguistic_agent.py` | ✅ | Uses shared `LINGUISTIC_AGENT_PROMPT`; pluggable LLM |
| Conflict Resolver (scenarios A/B/C) | `agents/conflict_resolver.py` | 🟡 | Core logic drafted; add per-paragraph batch + heatmap |
| Wire Detector+Linguistic+Resolver into a per-paragraph verdict | orchestrator | ⬜ | Next |

---

## Phase 3 — CrewAI Orchestrator + Conflict Resolution (In Progress)

| Task | File | Status |
| :--- | :--- | :---: |
| Verify CrewAI installs cleanly in the env | — | ⬜ |
| Wrap deterministic logic as CrewAI tools | `agents/tools/` | ⬜ |
| Build the crew (agents + tasks) | `agents/orchestrator.py` | ⬜ |
| Cross-agent conflict rules (plag↔citation, ai↔quality) | orchestrator | ⬜ |
| CLI entry point | `main.py` | ⬜ |

---

## Phase 4 — Streamlit UI (Not Started)

Upload → progress → dashboard (AI heatmap with override reasoning, citation
table, plagiarism/quality panels) → PDF export → disclaimers → `@st.cache_data`.

---

## Phase 5 — Polish & Deploy (Not Started)

Real-paper testing, edge cases, rate limiting/retries, Streamlit Cloud deploy,
docs, billing alerts.

---

## Recent Cleanup

- Removed duplicate legacy `core/citation_agent.py` (superseded by `agents/citation_agent.py`).
- Removed superseded trainers `train_ai_detector.py` (v1), `train_multi_dataset.py` (v1.5) and ephemeral `test_checkpoint.py` (all preserved in git history).
- Hardened `.gitignore` (`hf_cache/`, `*.arrow`, `*.safetensors`).
- Added `torch`, `transformers`, `accelerate` to `requirements.txt`.

---

## Known Issues / Notes

| Issue | Severity | Status | Notes |
| :--- | :---: | :---: | :--- |
| Env split: `training/venv` has torch+transformers+CUDA; system Python has agent deps but a broken transformers (hub version conflict) | Medium | ⬜ Open | Consolidate to one env before end-to-end runs |
| CrewAI dependency weight (pydantic v2, litellm) may conflict | Medium | ⬜ Open | Verify install before full refactor |
| No API keys configured | Medium | ⬜ Open | Agents run degraded without `.env` |

---

## API Keys Status

| API | Key needed? | Free tier |
| :--- | :---: | :--- |
| Gemini (AI Studio) | Yes | 15–30 RPM |
| Serper | Yes | 2,500 credits/month |
| Semantic Scholar | Optional | 1 req/sec (key) |
| CrossRef | No | Unlimited (polite pool) |
