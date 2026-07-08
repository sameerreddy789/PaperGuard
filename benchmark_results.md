# Detector Benchmark — fresh multi-model text (2025 models)

Test set: `benchmark_samples.json` — 40 AI passages (5 default + 5 disguised each
from Gemini, Claude Sonnet 5, GPT-5.5, Grok). Scored with `benchmark_detector.py`
(threshold 50 on the calibrated AI probability).

> **Caveat:** all 40 samples are AI, so this measures **AI recall only** — NOT
> accuracy or false-positive rate. A "call everything AI" detector would score
> 100% here. Real human samples are still needed to measure FPR before we can
> claim v2.1 is genuinely better (and not just trigger-happy).

Human set: 120 pre-2022 arXiv abstracts (real, human, in NEITHER v2.0 nor v2.1
training) — style caveat: formal abstracts match "default" AI style; the
"disguised" first-person AI has no informal-human counterpart yet.

## v2.0 (deployed baseline) — calibration MIDPOINT=12, SCALE=2.5

| Metric | Value |
|---|---|
| Human FPR (real arXiv) | **0.8% (1/120)** |
| AI recall — default | 50% (10/20) |
| AI recall — disguised | **0% (0/20)** |
| AI recall — overall | 25% (10/40) |
| **AUC** | **0.701** |

Threshold sweep (cutoff -> recall / FPR / Youden J):
`50: 25% / 0.8% / 0.24` · `40: 37.5% / 1.7% / 0.36` · `30: 42.5% / 5.0% / 0.38` · `20: 55% / 13.3% / 0.42`

Per-model recall (lower = evades better): Claude 10% · Gemini 20% · GPT-5.5 30% · Grok 40%.

**Takeaway:** v2.0 is *trustworthy but nearly blind* — FPR is a great 0.8%, but
disguised recall is 0% and AUC is only 0.70. Recalibrating the threshold (50->~35)
is a cheap partial win (recall ~40% at ~2-3% FPR) but can't fix the blind spot;
AUC caps it. The real fix must come from the model — that's v2.1.

## v2.1 (after adversarial/multi-model training) — TBD

_Re-run `benchmark_detector.py` with `PAPERGUARD_DETECTOR_MODEL=training/paperguard_v2_1`
(after re-fitting calibration) and record here for the before/after comparison._

| Group | AI recall |
|---|---|
| default | _tbd_ |
| disguised | _tbd_ |
| Overall | _tbd_ |
