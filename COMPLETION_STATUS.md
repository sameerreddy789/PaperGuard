# PaperGuard — Completion Status (done work)

> **Workflow:** this file tracks **completed** work. Planned / future work lives
> in `TASKS.md`. When a task in `TASKS.md` is finished, move it here.
> (The old `IMPLEMENTATION_PLAN.md` has been removed.)

**Last Updated:** 2026-07-08

---

## Scope & goal

PaperGuard is an academic-integrity checker for student work **and** research
papers (IEEE/ACM/Elsevier). It mimics Turnitin's two headline outputs — an
**AI-writing %** and a **Similarity/plagiarism %** — and adds **citation claim
verification** as a differentiator.

## Architecture (current)

- **CrewAI** orchestrates a crew of specialist agents (agents-as-tools). All
  fact-lookups/computation are deterministic tools; the LLM only reasons/
  synthesises. Crew LLM is provider-configurable (Gemini today; Qwen/DashScope
  ready for Alibaba).
- **AI detection = model-only** — the fine-tuned PaperGuard detector (calibrated
  logit margin) + embedding-based stylometric patchwork detection. No LLM used
  for detection.
- **Other layers** use the LLM for reasoning only: citation claim checks, quality
  prose review, plagiarism similarity, reference parsing.

---

## Completed

### Phase 1 — Core + services (✅)
| Item | File |
| :--- | :--- |
| PDF extraction (pymupdf4llm) | `core/pdf_parser.py` |
| Text chunker (sections→paragraphs→sentences) | `core/text_chunker.py` |
| Reference parser (LLM + **improved heuristic**: APA/IEEE/numeric title+author+DOI) | `core/reference_parser.py` |
| Service wrappers (Gemini/CrossRef/Semantic Scholar/Serper) + JSON cache | `services/*.py` |
| Data models (report, reference) | `models/*.py` |

### Phase 1.5 — Detector model (✅)
- v2.0 "mega" DistilBERT trained and **deployed** to HF
  `vediumsameer/paperguard-ai-detector` (`eval_loss=0.0003`). Training pipeline:
  `train_mega_dataset.py`. OOD validation: `ood_stress_test.py`.
- **Note:** the near-zero eval loss reflects overfitting to easy data → softmax
  saturation. Two retraining attempts (v2.1, v2.2) were made — see Phase 1.6.

### Phase 1.6 — Detector retraining attempts + honest benchmarking (✅ done; v2.0 retained)

A **frozen external benchmark** was built and both retrains were scored on it.
The headline finding: **v2.0 is still the best detector; do not overwrite it.**

| Component | File | Notes |
| :--- | :--- | :--- |
| Frozen benchmark (240 samples) | `benchmark_samples.json` | 40 AI (Gemini/Claude S5/GPT-5.5/Grok, default+disguised) + 200 human across 5 registers (arXiv/news/Yelp/Gutenberg/student). Not in any training set. |
| Benchmark harness | `benchmark_detector.py` | AUC, threshold sweep, dev/test split, per-register FPR, arXiv canary. |
| Results log | `benchmark_results.md` | v2.0 vs v2.1 vs v2.2, full per-register/per-model tables. |
| v2.1 trainer | (superseded by `train_v2_2.py`) | RAID adversarial + Ateeqq + pile, 2 epochs. |
| v2.2 trainer (fixed) | `train_v2_2.py` | Adds a **balanced, two-class, multi-register held-out eval** + a **nan-AUC abort alarm** + push safety gate. |

**Benchmark results (frozen set — directly comparable):**

| | AUC | human FPR | disguised recall | verdict |
| :--- | :---: | :---: | :---: | :--- |
| **v2.0 (deployed)** | **0.911** | **0.5%** | 0% | **shipped** |
| v2.1 | 0.391 | 70% | 30% | HARD BLOCK |
| v2.2 | 0.458 | 56% | 35% | HARD BLOCK |

**What was fixed vs what remains:**
- ✅ **Training methodology is fixed.** v2.1 failed because its held-out eval was
  AI-only → `eval_auc=nan` every epoch → FPR was never measured → the model
  over-flagged human text invisibly. `train_v2_2.py` fixes this: the held-out
  eval is now two-class and register-diverse (v2.2 held-out: AUC 0.986, FPR 1.4%,
  real numbers), and a callback aborts on `nan` AUC.
- ❌ **The model still doesn't beat v2.0.** Even with an honest eval, v2.2 scores
  AUC 0.458 (below random) on the frozen benchmark. Root cause is now clearly a
  **data problem**: RAID adversarial text does not transfer to 2025-model
  disguised AI, and continued training on it **degrades** v2.0's discrimination
  (catastrophic forgetting). v2.1/v2.2 artifacts retained for forensics only.
- **Decision:** v2.0 remains the production detector; HF is **not** overwritten.
  The path to v2.3 is training data that actually represents 2025-model disguised
  text, not another RAID retrain (see TASKS.md).

### Phase 2 — Base agents (✅)
| Agent | File |
| :--- | :--- |
| Citation verification (existence + 4-tier claim support) | `agents/citation_agent.py` |
| Writing quality (structure + readability + prose) | `agents/quality_agent.py` |
| Plagiarism (Serper + scholarly + LLM similarity) | `agents/plagiarism_agent.py` |

Shared foundation `agents/base.py` (lazy imports, `.pdf/.md/.txt` loading, CLI).

### Phase 2.5 — AI detection (model-only) (✅)
| Component | File | Notes |
| :--- | :--- | :--- |
| Detector (calibrated logit margin) | `agents/detector_agent.py` | Softmax is saturated; we score off the margin. Clean AI ~76%, academic ~71%, human/ESL ~10% (was 0% for all). `embed_text` for stylometry. |
| AI-detection engine (heatmap + patchwork) | `agents/ai_detection.py` | Model-only per-paragraph heatmap + document verdict. |
| Stylometric patchwork detection | `agents/ai_detection.py` | Robust median/MAD embedding-outlier flag for mixed authorship ("Frankenstein"). Validated. |
| Calibration re-fit tool | `fit_calibration.py` | Logistic fit of MIDPOINT/SCALE from labelled margins (`--self-test` validated). |

> **Decision (2026-07-08): removed the LLM "safety net"** from AI detection.
> Deleted `linguistic_agent.py`, `conflict_resolver.py`, `ai_detection_agent.py`,
> `safety_net.py`. The calibrated model + patchwork are the detection signal;
> Gemini/LLM is reserved for orchestration and the other tasks.

### Phase 3 — CrewAI orchestrator (✅)
- `agents/orchestrator.py`: crew of 4 specialists + editor, with a deterministic
  engine fallback and cross-agent conflict notes; `agents/crew_tools.py` wraps
  logic as tools; `main.py` CLI. Crew LLM configurable for Qwen/DashScope.
- Validated end-to-end (engine path) on the sample paper.

### Phase 4 — Streamlit UI (✅ built)
- `app.py`: per-paragraph AI heatmap + patchwork flags, 4-tier citation table,
  plagiarism/quality panels, executive summary, **PDF + JSON export**, sidebar
  (keys, detector model, crew toggle), `@st.cache_data`. Boots clean.
- Remaining: live browser smoke-test with a real upload.

### Cross-cutting done
- Migrated off the EOL `google.generativeai` SDK to **`google-genai`**;
  `services/gemini.py` is the single integration point; model unified via
  `PAPERGUARD_GEMINI_MODEL`.
- Improved heuristic reference parser (real titles → citation lookups work
  without an LLM key; validated: 3 real refs resolved, 1 fabricated flagged).
- Environment consolidated into `.venv` (torch, transformers, crewai, streamlit).
- Cleanup: removed duplicate `core/citation_agent.py`, superseded trainers
  (v1/v1.5), `test_checkpoint.py`, stale `test_model.py`, and the removed
  AI-detection LLM modules.
- **LLM determinism:** verdict + synthesis calls now default to **temperature
  0.0** (`services/gemini.py`, `agents/orchestrator.py`), overridable via
  `PAPERGUARD_LLM_TEMPERATURE`, so a paper yields reproducible reports run-to-run.
  JSON output was already enforced (`response_mime_type` + fence stripping).

---

## What's next

Detector retraining is **done and closed** (v2.0 is final; v2.1/v2.2 failed the
benchmark — Phase 1.6). See **`TASKS.md`** for the remaining plan: agent-centric
report improvements, plagiarism upgrade (fingerprint + semantic), the Integrity
Dashboard, and **Alibaba Cloud deployment with Qwen**.

## API keys

| API | Needed? | Notes |
| :--- | :---: | :--- |
| Gemini (or Qwen/DashScope) | for LLM tasks + crew | Currently no valid key in `.env` |
| Serper | for web plagiarism search | optional |
| Semantic Scholar | optional | improves abstract retrieval |
| CrossRef | no key | unlimited |
