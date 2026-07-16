# PaperGuard — Tasks & Strategy (future work)

> **Workflow:** this file holds *planned / in-progress* work. When a task is
> finished, move it to `COMPLETION_STATUS.md`. New tasks are added here first.
> (This file replaces the old `IMPLEMENTATION_PLAN.md`.)

**Core goal:** mimic Turnitin's two headline outputs — an **AI-writing %** and
a **Similarity/plagiarism %** — at comparable accuracy, and add our
differentiator: **citation claim verification**.

**Owners:** model training is **closed** (@sameerreddy — v2.0 final). The
agent/product/deployment track (@technical-monish) is **done** except the
actual live Alibaba deployment (see below) — everything needed for it has
been built and locally verified.

---

## Snapshot (where we are)

- **Detector = `desklib/ai-text-detector-v1.01`** (deberta-v3-large) — swapped
  in 2026-07-14 after beating both our in-house v2.0 (retired) and a second
  external candidate on the same frozen benchmark (AUC 0.968 vs v2.0's 0.911;
  disguised-AI recall 75% vs v2.0's 0%). In-house retraining (v2.1/v2.2) is
  closed, see `COMPLETION_STATUS.md` Phase 1.6 for that history and Phase 1.7
  for the swap details.
- **AI detection = model-only** (direct sigmoid classifier output + embedding
  patchwork). No LLM.
- **Agent society is complete**: orchestrator surfaces deterministic hard facts
  + headline metrics as structured `Report` fields (not just LLM prose);
  plagiarism combines n-gram overlap + semantic embeddings + LLM judgment with
  cross-agent quote/citation dedupe; citations add retraction + DOI-consistency
  checks; the UI has a Turnitin-style Integrity Dashboard, a combined
  per-paragraph overlay, and an annotated-PDF export. See
  `COMPLETION_STATUS.md` Phase 5 for the full list.
- **LLM backend is dual**: Gemini or Qwen/DashScope for both the crew-level
  synthesis LLM and the sub-agent calls (citation/plagiarism/quality/reference
  parsing), selectable via `PAPERGUARD_LLM_PROVIDER` (auto-detects from
  whichever API key is set). Both default to `PAPERGUARD_LLM_TEMPERATURE=0.0`
  for reproducible reports.
- **Containerized and locally verified**: `Dockerfile` + `.dockerignore` at the
  repo root; a real `docker build` + `docker run` was executed in this
  environment and the container reported `(healthy)` with both `/` and
  `/_stcore/health` returning HTTP 200. See `DEPLOYMENT.md`.

---

## Remaining: execute the live Alibaba Cloud deployment

Everything needed to deploy is built, documented, and locally verified in
`DEPLOYMENT.md` — but the actual cloud steps (ACR push, Function Compute/
PAI-EAS setup, live smoke-test on a public URL) require an Alibaba account and
credentials that were not available in the environment this was built in.

- [ ] Create/confirm an Alibaba Container Registry (ACR) namespace and push
  the image (`DEPLOYMENT.md` §1).
- [ ] Set the production environment variables (DashScope key, etc. —
  `DEPLOYMENT.md` §2) on the chosen compute target. **Do not commit secrets.**
- [ ] Deploy via Function Compute 3.0 (recommended) or PAI-EAS or a plain
  ECS+Docker fallback (`DEPLOYMENT.md` §3a/3b/3c).
- [ ] Run the live smoke-test checklist against the deployed URL
  (`DEPLOYMENT.md` §4): health check, small-text upload, real-PDF upload +
  annotated-PDF export, crew on/off toggle, live CrossRef/Semantic Scholar
  reachability.
- [ ] If demo timing matters, address cold-start (keep ≥1 warm instance or use
  PAI-EAS/ECS instead of pure scale-to-zero FC — see `DEPLOYMENT.md` "Notes").

---

## Nice-to-haves (not blocking, pick up if there's runway left)

- Pin exact dependency versions in `requirements.txt` (currently unpinned —
  fine for a hackathon, but reproducibility risk longer-term).
- Real-paper testing pass: a genuine two-column IEEE/ACM PDF end-to-end,
  checking the annotated-PDF highlight hit-rate on dense two-column layouts
  (PDF text reflow/hyphenation can make `highlight_pdf`'s snippet search miss
  spans — see the "not_found" stat it reports).
- Broaden plagiarism retrieval further (more open-access scholarly sources
  beyond CrossRef/Semantic Scholar; more Serper query variants).
- Live browser smoke-test of `app.py` with a real file upload through the
  actual Streamlit UI (this session verified the container serves the app and
  responds correctly over HTTP, but did not click through the UI in a browser).

---

## Tracked risks / blind spots

- **Reduced, not eliminated:** fully style-masked / disguised 2025-model AI now
  evades the detector 25% of the time (down from 100% with the old v2.0 after
  the swap to desklib/ai-text-detector-v1.01 — see Snapshot above). In-house
  RAID retraining did **not** fix v2.0's version of this (v2.1/v2.2 failed the
  benchmark — that training chapter is closed, no v2.3 planned; the fix instead
  came from adopting an external model trained on more representative data).
  **Reproduced concretely on 2026-07-16** on a synthetic mixed-authorship test
  paper (`tests/sample_papers/green_spaces_test.md`): the detector scored the
  whole paper 94.82% "Likely AI", including 4 deliberately human-voiced
  sections. Root cause is the model's own judgment on informal prose wrapped in
  academic scaffolding, not our integration — no code fix changes that. What
  *was* fixed (signal-combination layer, see `PROJECT_REPORT.md` Section 10):
  cross-referencing each AI-flagged paragraph against `QualityAgent`'s
  independent tone verdict and surfacing an explicit "LOW-CONFIDENCE AI
  VERDICT" conflict note when they disagree, plus a softer `near_outliers`
  patchwork tier for style shifts too moderate to cross the strict threshold.
  Mitigation: patchwork detection (now two-tiered) + the "Uncertain" band for
  human review + the new tone-conflict note, framing AI-detection as one
  signal, not a verdict.
- **New since the swap:** desklib v1.01 is a deberta-v3-large (~400M params,
  1.74GB), ~6.7x the size of the retired DistilBERT (~66M params, 260MB) —
  heavier CPU inference latency and cold-start time; not yet stress-tested at
  the same real-paper scale as v2.0 was. Calibration (the `_LIKELY_AI=90`
  threshold) was chosen from the frozen benchmark's FPR/recall sweep, not
  independently re-validated on a live paper corpus yet.
- Plagiarism can't match Turnitin's private student-paper DB — scoped to open
  web + open-access scholarly, and the UI/PDF export both say so honestly.
- **Fixed 2026-07-16:** CrossRef's `search_works_by_title()` matches on the
  *title* field, so a body-paragraph key phrase (never a paper's own title)
  could return a completely unrelated top hit with no relevance check —
  confirmed missing an uncredited verbatim WHO-constitution quote on the
  synthetic test paper above. Added a keyword-overlap relevance guard, a
  Semantic Scholar search fallback, and a small deterministic known-text
  fingerprint list (WHO health definition, UDHR Article 1) for the handful of
  extremely famous passages title-search will never retrieve. See
  `PROJECT_REPORT.md` Section 10 for the before/after numbers. This remains a
  narrow supplement — general plagiarism retrieval coverage is still bounded
  by the "Nice-to-haves" item above.
- Free-tier compute limits: torch + crewai + chromadb is heavy (~830MB image);
  size the Alibaba instance accordingly, or serve the detector via HF Inference
  if a tighter memory budget is needed.
- No valid `GEMINI_API_KEY` in `.env` currently — set either that or
  `DASHSCOPE_API_KEY` before a live deployment (see `.env.example`).
- The annotated-PDF export is best-effort: `highlight_pdf`'s text search can
  miss a span if the PDF's text layer reflows/hyphenates it differently than
  the extracted paragraph text; it reports a `not_found` count rather than
  silently claiming full coverage.
