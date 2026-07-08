# Detector Benchmark — fresh multi-model text (2025 models)

Test set: `benchmark_samples.json` — 40 AI passages (5 default + 5 disguised each
from Gemini, Claude Sonnet 5, GPT-5.5, Grok). Scored with `benchmark_detector.py`
(threshold 50 on the calibrated AI probability).

> **Caveat:** all 40 samples are AI, so this measures **AI recall only** — NOT
> accuracy or false-positive rate. A "call everything AI" detector would score
> 100% here. Real human samples are still needed to measure FPR before we can
> claim v2.1 is genuinely better (and not just trigger-happy).

**Corpus composition (state this next to every FPR/AUC number):** 40 AI
(20 default + 20 disguised, from Gemini/Claude/GPT-5.5/Grok) + 160 human across
4 registers, all pre-LLM and in NEITHER v2.0 nor v2.1 training:
- `human_arxiv`      — 40 formal STEM abstracts (arxiv-abstracts-2021)
- `human_news`       — 40 formal non-STEM articles (cnn_dailymail)
- `human_informal`   — 40 informal first-person (Yelp reviews)
- `human_humanities` — 40 humanities/literary prose (Project Gutenberg)

> Caveats: (1) informal=Yelp and humanities=Gutenberg match register/topic axes
> but not "modern student essay" exactly. (2) AUC/FPR depend on the human mix —
> arXiv abstracts sit closest to the AI cluster, so an arXiv-only negative set
> understates AUC. (3) N is small (20/group AI, 40/group human) — wide CIs;
> report rates with N.

## v2.0 (deployed baseline) — calibration MIDPOINT=12, SCALE=2.5

| Metric | Value |
|---|---|
| Human FPR — arXiv (formal STEM) | 2.5% (1/40) — the ONLY human group with any FPR |
| Human FPR — news / informal / humanities | 0% / 0% / 0% |
| AI recall — default | 50% (10/20) |
| AI recall — disguised | **0% (0/20)** |
| **AUC (4-register human)** | **0.891**  (3-register 0.859; arXiv-only 0.701) |

**Honest generalization number (dev/test split, threshold picked on DEV only):**
cutoff 15 → **TEST recall 63.6% / FPR 5.0%** (n~102). This is the number to quote.

Full-set threshold sweep (DESCRIPTIVE only — do NOT pick a cutoff from this and
report it on the same data): `50: 25%/0.6%` · `30: 42.5%/1.2%` · `20: 55%/2.5%`.

Per-model recall (lower = evades better): Claude 10% · Gemini 20% · GPT-5.5 30% · Grok 40%.

**Findings:**
1. **No formality OR topic confound** — 0% FPR on news, informal, AND humanities
   human. v2.0's disguised-AI miss is a genuine signal gap, not register/topic keying.
2. **Formal STEM abstracts are the hardest human** — the sole human false-positive
   is a dense physics abstract (matches "abstracts structurally resemble AI").
3. **Recalibration alone is not a clean fix** — the honest dev/test result is
   63.6% recall at **5.0% FPR** (1 in 20 humans flagged — too high to ship). You
   cannot get both high recall and low FPR at AUC ~0.89.
4. **The disguised blind spot needs the model to LEARN it → v2.1.** AUC is an
   information ceiling; the threshold sweep is not the fix.

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
