# Detector Benchmark — fresh multi-model text (2025 models)

Test set: `benchmark_samples.json` — 40 AI passages (5 default + 5 disguised each
from Gemini, Claude Sonnet 5, GPT-5.5, Grok). Scored with `benchmark_detector.py`
(threshold 50 on the calibrated AI probability).

> **Caveat:** all 40 samples are AI, so this measures **AI recall only** — NOT
> accuracy or false-positive rate. A "call everything AI" detector would score
> 100% here. Real human samples are still needed to measure FPR before we can
> claim v2.1 is genuinely better (and not just trigger-happy).

**Corpus composition (state this next to every FPR/AUC number):** 40 AI
(20 default + 20 disguised, from Gemini/Claude/GPT-5.5/Grok) + 120 human across
3 registers, all pre-LLM and in NEITHER v2.0 nor v2.1 training:
- `human_arxiv`  — 40 formal STEM abstracts (arxiv-abstracts-2021)
- `human_news`   — 40 formal non-STEM articles (cnn_dailymail)
- `human_informal` — 40 informal first-person (Yelp reviews)

> Caveats: (1) informal human = Yelp reviews (register matches "disguised" AI,
> topic does not). (2) AUC/FPR depend on the human mix — arXiv abstracts sit
> closest to the AI cluster, so an arXiv-only negative set understates AUC.
> (3) N is small (20/group AI, 40/group human) — treat as indicative.

## v2.0 (deployed baseline) — calibration MIDPOINT=12, SCALE=2.5

| Metric | Value |
|---|---|
| Human FPR — arXiv (formal STEM) | 2.5% (1/40) — the riskiest human case |
| Human FPR — news | 0% |
| Human FPR — informal (Yelp) | 0% |
| AI recall — default | 50% (10/20) |
| AI recall — disguised | **0% (0/20)** |
| AI recall — overall (cutoff 50) | 25% (10/40) |
| **AUC (diverse human)** | **0.859**  (arXiv-only human: 0.701) |

Threshold sweep (cutoff -> recall / FPR / Youden J):
`50: 25% / 0.8% / 0.24` · `40: 37.5% / 0.8%` · `30: 42.5% / 1.7%` · `20: 55% / 3.3% / 0.52`
Dev/test split (threshold on dev, reported on test): cutoff 15 -> TEST recall 63.6% / FPR 6.9%.

Per-model recall (lower = evades better): Claude 10% · Gemini 20% · GPT-5.5 30% · Grok 40%.

**Findings:**
1. **No formality confound** — v2.0 gives 0% FPR on informal (Yelp) human, so its
   disguised-AI miss is a genuine signal gap, not register-keying.
2. **Formal abstracts are the hardest human** — the only human false-positive is
   an arXiv abstract (matches the "abstracts structurally resemble AI" concern).
3. **Recalibration is a real cheap partial win** — moving cutoff 50->~20 doubles
   recall (25%->55%) at ~3% FPR, no retrain. But disguised recall stays weak.
4. **AUC 0.86 is a ceiling** — no threshold yields both high recall and <1% FPR.
   The disguised blind spot needs the model to *learn* it → v2.1.

_This file (benchmark_samples.json) is FROZEN as the v2.1 holdout. v2.1 trains on
RAID (separate), so no leakage. Any extra disguised training data for v2.2 must be
a fresh batch, never these samples._

## v2.1 (after adversarial/multi-model training) — TBD

_Re-run `benchmark_detector.py` with `PAPERGUARD_DETECTOR_MODEL=training/paperguard_v2_1`
(after re-fitting calibration) and record here for the before/after comparison._

| Group | AI recall |
|---|---|
| default | _tbd_ |
| disguised | _tbd_ |
| Overall | _tbd_ |
