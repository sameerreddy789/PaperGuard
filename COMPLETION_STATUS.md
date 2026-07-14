# PaperGuard — Completion Status (done work)

> **Workflow:** this file tracks **completed** work. Planned / future work lives
> in `TASKS.md`. When a task in `TASKS.md` is finished, move it here.
> (The old `IMPLEMENTATION_PLAN.md` has been removed.)

**Last Updated:** 2026-07-14

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

### Phase 1.7 — Model swap to desklib/ai-text-detector-v1.01 (✅ done, 2026-07-14)

With in-house retraining closed (Phase 1.6), external HF detector models were
benchmarked head-to-head on the SAME frozen 240-sample set to check whether a
better-trained model exists externally. Two candidates were tested and one
commercial-SaaS category was referenced (not directly testable, no API key):

| Model | AUC | Human FPR @50 | Disguised recall | Verdict |
| :--- | :---: | :---: | :---: | :--- |
| v2.0 (previous, in-house) | 0.911 | 0.5% | 0% | retired |
| **desklib/ai-text-detector-v1.01** | **0.968** | 7.0% | **75%** | **adopted** |
| mdrakibali/deberta-ai-detector-v3 | 0.767 | 14.5% | 35% | ruled out |
| CopyLeaks / Winston AI (SaaS, reference only) | n/a (independent studies) | 19-35% (Copyleaks) | n/a | not integrated |

- mdrakibali's HF config exposed only generic `LABEL_0`/`LABEL_1` (no
  descriptive label names) and its own model card admitted the label direction
  was an unconfirmed guess. Rather than trust that guess, the direction was
  confirmed empirically (obvious human vs. obvious AI-boilerplate text; index 1
  = AI, P≈1.000 vs P≈0.001) before running the full benchmark — it still lost
  decisively on every axis.
- **Adopted `desklib/ai-text-detector-v1.01`** (deberta-v3-large): the only
  candidate that both improves meaningfully on v2.0's exact blind spot
  (disguised-AI recall) and keeps FPR workable at a higher operating threshold.

**Code changes** (`agents/detector_agent.py`, `agents/ai_detection.py`,
`agents/orchestrator.py`, `app.py`, `requirements.txt`, `README.md`,
`.env.example`, `Dockerfile`, `DEPLOYMENT.md`):
- New `_DesklibAIDetectionModel` class (mean-pooling + single-logit classifier
  head, matching the model card's own reference implementation exactly, incl. a
  `all_tied_weights_keys` compatibility shim for the current `transformers`
  version) replaces the old `AutoModelForSequenceClassification` load path —
  this model is not a standard classification head.
- `score_text` now reads `sigmoid(logit)` directly as `ai_probability`; the old
  logit-margin recalibration (`_CALIB_MIDPOINT`/`_CALIB_SCALE`, needed because
  v2.0's softmax was saturated) is removed — this model's sigmoid output was
  not found to be saturated on the benchmark.
- `_LIKELY_AI` raised from 65 to **90** (env-overridable via
  `PAPERGUARD_DETECTOR_AI_THRESHOLD`) to match the benchmark's measured FPR/
  recall operating point for this model (~0.5% FPR / ~85-87.5% recall at ~90-95,
  vs. 7.0% FPR at the model's naive 50% default).
- `embed_text` (used by stylometric patchwork detection) now pools from the
  wrapper's underlying DeBERTa encoder (`bundle.model.model`) instead of
  `output_hidden_states`, since the custom architecture's `forward` doesn't
  expose that kwarg.
- Added `sentencepiece` to `requirements.txt` (needed for the DeBERTa-v2/v3
  tokenizer; the old DistilBERT tokenizer didn't require it).

**Verified, not assumed:** the custom model wrapper's output was checked
byte-for-byte against the model card's own reference `predict_single_text`
function on its own example texts (both matched exactly: AI text P=0.9974,
human text P=0.4245) before trusting any downstream score. `embed_text` and the
full `run_ai_detection`/orchestrator pipeline were run end-to-end on the sample
paper after the swap (engine mode, no LLM key) and produced a coherent report.
Old v2.0 fallback code paths referencing the retired `training/mega_dataset_model_v2`
local directory were removed from `app.py`'s sidebar default (that directory
used an incompatible model architecture/class for the new model).

### Phase 2 — Base agents (✅)
| Agent | File |
| :--- | :--- |
| Citation verification (existence + 4-tier claim support) | `agents/citation_agent.py` |
| Writing quality (structure + readability + prose) | `agents/quality_agent.py` |
| Plagiarism (Serper + scholarly + LLM similarity) | `agents/plagiarism_agent.py` |

Shared foundation `agents/base.py` (lazy imports, `.pdf/.md/.txt` loading, CLI).

### Phase 2.5 — AI detection (model-only) (✅; detector swapped in Phase 1.7)
| Component | File | Notes |
| :--- | :--- | :--- |
| Detector (desklib deberta-v3-large, sigmoid output) | `agents/detector_agent.py` | Direct `sigmoid(logit)` AI probability; "Likely AI" threshold raised to ~90 (benchmark-derived operating point) rather than recalibrating, since this model's output was not found to be saturated. `embed_text` for stylometry. |
| AI-detection engine (heatmap + patchwork) | `agents/ai_detection.py` | Model-only per-paragraph heatmap + document verdict. |
| Stylometric patchwork detection | `agents/ai_detection.py` | Robust median/MAD embedding-outlier flag for mixed authorship ("Frankenstein"). Validated. |
| Calibration re-fit tool (legacy, v2.0-era) | `fit_calibration.py` | Logistic fit of MIDPOINT/SCALE from labelled margins — was used for the retired v2.0 DistilBERT's saturated-softmax problem; not needed for the current detector, kept for reference/future models that exhibit the same saturation issue. |

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
- Superseded/extended by Phase 5 below (Integrity Dashboard, combined overlay,
  annotated PDF).

### Phase 5 — Agent-centric hardening + Turnitin-parity UI + deployment (✅)

Closed out the full remaining `TASKS.md` backlog in one pass: structured hard
facts, a genuinely multi-signal plagiarism check, citation retraction/DOI
checks, a real Integrity Dashboard, an annotated-PDF export, a second LLM
backend, and a verified container.

| Item | File(s) | Notes |
| :--- | :--- | :--- |
| Structured hard facts | `models/report.py`, `agents/orchestrator.py` | `Report.conflict_notes` (fabricated/retracted citation counts, AI-vs-structure conflicts) and `Report.headline_metrics` (deterministic AI%/Similarity%/Citation-Health/Quality + bands) are now **always** populated by `_build_report`, independent of whether the LLM crew ran — previously these facts only reached the user if the LLM chose to mention them in its prose summary. |
| Plagiarism: 3-signal scoring + dedupe | `agents/plagiarism_agent.py` | Each candidate match is now scored by (a) deterministic word-shingle **Jaccard n-gram overlap** (no LLM/key needed, catches verbatim copying), (b) **semantic cosine similarity** reusing `DetectorAgent.embed_text` (no new model/dependency), (c) the existing LLM judgment — the best available score wins, so plagiarism detection degrades gracefully instead of going fully dark without a Gemini key. Added `_looks_quoted_and_cited`: a paragraph that is both quoted and matches an already-extracted `Reference` is **downgraded** (not counted as plagiarism) rather than flagged — cross-agent dedupe with the citation agent. |
| Citation: retraction + DOI-consistency | `services/crossref.py`, `agents/citation_agent.py` | Added `get_retraction_notices` (reads CrossRef's `message['updated-by']`, confirmed against a live API call on a real retracted paper — Wakefield's 1998 Lancet MMR paper). Added `_check_retraction` and `_check_doi_consistency` (title/year/first-author vs. the resolved CrossRef record) to every reference; both feed into `retracted_count`/`doi_mismatch_count`, new findings, a citation-health penalty, and `status="failed"` on any retraction. |
| Integrity Dashboard + combined overlay | `app.py` | `render_dashboard` gives the two Turnitin-style headline numbers (AI% / Similarity%) visual priority with band coloring, plus a "Hard facts" panel rendering `conflict_notes`. New `render_overlay` tab joins the AI-detection heatmap with plagiarism matches (by normalized paragraph **text**, not index — the two agents select different paragraph subsets) and stylometric outliers into one per-paragraph view with inline badges. `build_pdf` now reads the same `headline_metrics`/`conflict_notes` fields so the PDF and UI never disagree. |
| Annotated PDF export | `core/pdf_parser.py`, `app.py` | New `highlight_pdf` opens the **original** uploaded PDF bytes with PyMuPDF (`fitz`) and highlights flagged spans in place via `page.search_for` + `add_highlight_annot` — deliberately avoids rebuilding a full text-to-bbox pipeline (pymupdf4llm's Markdown extraction discards that). `spans_from_report` builds the highlight list from likely-AI paragraphs, flagged plagiarism matches, and patchwork outliers. Degrades gracefully (reports a `not_found` count for spans that don't re-locate, e.g. due to PDF text reflow). |
| Qwen/DashScope LLM backend | `services/dashscope_llm.py`, `services/llm.py`, `agents/base.py`, `core/reference_parser.py` | New OpenAI-compatible Qwen client (`services/dashscope_llm.py`) with the identical `call_llm`/`call_llm_json` contract as `services/gemini.py`. New `services/llm.py` selects between them via `PAPERGUARD_LLM_PROVIDER` (`auto`/`gemini`/`dashscope`; auto prefers DashScope only if it's the *only* key set). `agents.base.get_llm()` now returns this selector, so every existing sub-agent call site (citation/plagiarism/quality/reference parsing) works unchanged on either backend. (The crew-level LLM, `agents/orchestrator.py`, was already provider-agnostic via CrewAI/LiteLLM's `PAPERGUARD_CREW_MODEL`.) |
| Containerization | `Dockerfile`, `.dockerignore`, `DEPLOYMENT.md` | CPU-only PyTorch (explicit `--index-url .../whl/cpu` layer, overridable for GPU builds), non-root user, `$PORT`-driven, `HEALTHCHECK` against Streamlit's `/_stcore/health`. **Actually built and run in this environment**: `docker build` succeeded (827MB image), `docker run` started the container, and both `GET /` and `GET /_stcore/health` returned HTTP 200 while Docker's own `HEALTHCHECK` independently reported the container as `(healthy)`. `DEPLOYMENT.md` documents the ACR push + Function Compute 3.0 / PAI-EAS / ECS deployment steps and a live smoke-test checklist. |

**Verification methodology used throughout Phase 5** (not just written and
assumed correct): live CrossRef API calls against a real retracted DOI;
offline unit checks of the Jaccard/cosine/dedupe math with assertions; a
synthetic PDF built with `fitz` to prove `highlight_pdf` actually finds and
annotates real text; 7 explicit provider-selection scenarios for the LLM
router; and a real `docker build && docker run` with HTTP health checks against
the running container. All temporary verification scripts were deleted after
use; none were committed.

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

Detector retraining is **done and closed** (Phase 1.6) and the full
agent/product backlog is **done** (Phase 5). The only thing left is **executing
the live Alibaba Cloud deployment** — the container is built and locally
verified, but the actual ACR push / Function Compute or PAI-EAS setup / live
smoke-test require Alibaba account credentials not available in this
environment. See **`TASKS.md`** ("Remaining: execute the live Alibaba Cloud
deployment") and **`DEPLOYMENT.md`** for the exact steps.

## API keys

| API | Needed? | Notes |
| :--- | :---: | :--- |
| Gemini **or** DashScope (Qwen) | for LLM tasks + crew | Set either `GEMINI_API_KEY` or `DASHSCOPE_API_KEY`; `services/llm.py` auto-selects. Currently no valid key in `.env`. See `.env.example`. |
| Serper | for web plagiarism search | optional (n-gram + semantic similarity still work without it) |
| Semantic Scholar | optional | improves abstract retrieval |
| CrossRef | no key | unlimited; also powers retraction + DOI-consistency checks |
