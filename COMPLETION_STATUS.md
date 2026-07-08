# PaperGuard — Completion Status

> **UPDATE THIS FILE BEFORE EVERY PUSH.** Single source of truth for project state.

**Last Updated:** 2026-07-07
**Updated By:** Safety Net wired + validated; CrewAI orchestrator + tools + CLI built; env consolidated (.venv)

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
| Phase 2.5: AI-Detection Safety Net | ✅ Complete | Detector + Linguistic + Conflict Resolver wired; validated on OOD cases |
| Phase 3: CrewAI orchestrator + conflict resolution | ✅ Complete | Crew built + engine fallback; end-to-end Report validated (engine path) |
| Phase 4: Streamlit UI | 🟡 Built | `app.py` boots clean; heatmap/citations/panels render off validated report; PDF export + live browser smoke-test pending |
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
| Conflict Resolver (scenarios A/B/C) | `agents/conflict_resolver.py` | ✅ | None-safe; per-paragraph batch + heatmap + document rollup |
| Safety-net runner (per-paragraph verdict + heatmap) | `agents/safety_net.py` | ✅ | Ties detector+linguistic+resolver over a document |

> **Detector salvaged via logit-margin calibration.** The v2.0 model's *softmax*
> is saturated (reports ~0% AI even on real AI text), but the raw logit margin
> (human-ai) still separates classes: clean/academic AI ~6-8, human ~16-18. The
> Detector now scores off a logistic calibration of that margin instead of the
> softmax. Result: it correctly flags clean AI ~76% and academic AI ~71% while
> keeping human/ESL text ~10% (previously 0% for everything). The "math" agent
> now contributes real signal. Calibration constants are heuristic (env-overridable)
> and should ideally be fit on a labelled dev set for a research-grade release.
>
> **Remaining blind spot:** slang/style-masked AI can still read as human from
> the model alone — this is exactly what the Linguistic agent + Conflict Resolver
> override (`logit_saturation` scenario) catches, which was validated on the OOD cases.

---

## Phase 3 — CrewAI Orchestrator + Conflict Resolution (In Progress)

| Task | File | Status |
| :--- | :--- | :---: |
| Verify CrewAI installs cleanly in the env | `.venv` | ✅ (crewai 1.15.1, no conflicts) |
| Wrap deterministic logic as CrewAI tools | `agents/crew_tools.py` | ✅ |
| Build the crew (agents + tasks) | `agents/orchestrator.py` | ✅ (4 specialists + editor) |
| Cross-agent conflict rules (plag↔citation, ai↔quality) | `agents/orchestrator.py` | ✅ |
| CLI entry point | `main.py` | ✅ |
| Live crew run with a valid Gemini key | — | ⬜ (engine path validated; crew kickoff needs a valid `GEMINI_API_KEY`) |

> Environment consolidated into a single `.venv` (Python 3.10) with the full
> stack: torch 2.12.1 (CPU), transformers 5.13.0, crewai 1.15.1, google-generativeai,
> streamlit. `requirements.txt` now installs the whole app.
>
> **Deployment note:** the full stack (torch + crewai + chromadb + onnxruntime)
> is heavy for free Streamlit Community Cloud. Phase 5 may need to serve the
> detector via the HF Inference API (instead of local torch) or use a larger host.

---

## Next Session (tomorrow) — planned work

See `IMPLEMENTATION_PLAN.md` → "Pending Work / Next-Session Backlog" for detail.

| # | Item | File(s) | Priority |
| :--- | :--- | :--- | :---: |
| A | ~~Streamlit UI (heatmap, citation table, panels, disclaimers, caching)~~ **BUILT** — remaining: PDF export + live browser smoke-test | `app.py` | done |
| B1 | Embedding-based stylometric drift / "Frankenstein" patchwork detection | `agents/detector_agent.py` | 2 |
| B2 | Re-fit detector calibration (temperature/Platt) on a labelled dev slice | training script | 3 |
| C | Live CrewAI crew run (needs valid `GEMINI_API_KEY`) | `.env` | 2 |
| D | Real-paper testing + deployment (HF Inference API for detector on free hosts) | — | 4 |
| E | Tech debt: migrate `reference_parser` off EOL `google.generativeai`; unify model name | `core/reference_parser.py` | 4 |

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
| Detector v2.0 softmax saturation (0% AI on all inputs) | High | ✅ Resolved | Now scores off calibrated logit margin: clean AI ~76%, academic AI ~71%, human ~10%. Calibration constants heuristic; fit on a labelled dev set for production |
| Invalid `GEMINI_API_KEY` in `.env` | High | ⬜ Open | Linguistic agent + CrewAI crew need a valid key; without it AI detection is detector-only |
| Env consolidated into `.venv` | — | ✅ Resolved | Full stack installs from `requirements.txt` |
| Full stack heavy for free Streamlit Cloud | Medium | ⬜ Open | Consider HF Inference API for the detector at deploy time |

---

## API Keys Status

| API | Key needed? | Free tier |
| :--- | :---: | :--- |
| Gemini (AI Studio) | Yes | 15–30 RPM |
| Serper | Yes | 2,500 credits/month |
| Semantic Scholar | Optional | 1 req/sec (key) |
| CrossRef | No | Unlimited (polite pool) |
