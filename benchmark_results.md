# Detector Benchmark — fresh multi-model text (2025 models)

Test set: `benchmark_samples.json` — 40 AI passages (5 default + 5 disguised each
from Gemini, Claude Sonnet 5, GPT-5.5, Grok). Scored with `benchmark_detector.py`
(threshold 50 on the calibrated AI probability).

> **Caveat:** all 40 samples are AI, so this measures **AI recall only** — NOT
> accuracy or false-positive rate. A "call everything AI" detector would score
> 100% here. Real human samples are still needed to measure FPR before we can
> claim v2.1 is genuinely better (and not just trigger-happy).

## v2.0 (deployed baseline) — calibration MIDPOINT=12, SCALE=2.5

| Group | AI recall |
|---|---|
| default (normal AI) | 50% (10/20) |
| disguised (human-styled AI) | **0% (0/20)** |
| **Overall** | **25% (10/40)** |

Per-model recall (lower = evades better): Claude 10% · Gemini 20% · GPT-5.5 30% · Grok 40%.

**Takeaway:** v2.0 has a total blind spot on disguised AI (0/20) and misses half
of even default AI from 2025 models (its RAID training data is from 2024).

## v2.1 (after adversarial/multi-model training) — TBD

_Re-run `benchmark_detector.py` with `PAPERGUARD_DETECTOR_MODEL=training/paperguard_v2_1`
(after re-fitting calibration) and record here for the before/after comparison._

| Group | AI recall |
|---|---|
| default | _tbd_ |
| disguised | _tbd_ |
| Overall | _tbd_ |
