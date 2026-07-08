# Detector Benchmark — fresh multi-model text vs. clean human corpora

Frozen test set: `benchmark_samples.json`, scored by `benchmark_detector.py`.
Both are FROZEN — v2.0 and v2.1 must be scored with the identical file + script +
env pattern (`PAPERGUARD_DETECTOR_MODEL=...`) so the before/after is clean.

## Corpus composition (always quote this next to any FPR/AUC number)

**240 samples** — 40 AI + 200 human across 5 registers, all pre-LLM and in
NEITHER v2.0 nor v2.1 training:

| Group | N | Source | Register |
|---|---|---|---|
| `default`          | 20 | Gemini/Claude S5/GPT-5.5/Grok | modern AI, normal style |
| `disguised`        | 20 | same models, "write like a human" prompt | modern AI, evasive |
| `human_arxiv`      | 40 | arxiv-abstracts-2021 | formal STEM |
| `human_news`       | 40 | cnn_dailymail | formal non-STEM journalism |
| `human_informal`   | 40 | Yelp reviews | informal first-person |
| `human_humanities` | 40 | Project Gutenberg | literary/humanities |
| `human_student`    | 40 | essays-big5 | real student writing (deployment register) |

> Caveats: (1) student/informal/humanities match register+topic axes but aren't
> "modern argumentative student essay" exactly. (2) AUC/FPR depend on the human
> mix — arXiv abstracts sit closest to the AI cluster (arXiv-only human gives
> AUC 0.70 vs 0.91 on the full mix), so always report the composition. (3) N is
> small (20/group AI, 40/group human) — wide CIs; report rates with N.

## v2.0 (deployed baseline) — calibration MIDPOINT=12, SCALE=2.5

| Metric | Value |
|---|---|
| Human FPR — arXiv | 2.5% (1/40) — the ONLY human group with any FPR |
| Human FPR — news / informal / humanities / student | 0% / 0% / 0% / 0% |
| Overall human FPR (cutoff 50) | 0.5% (1/200) |
| AI recall — default | 50% (10/20) |
| AI recall — disguised | **0% (0/20)** |
| **AUC (5-register human)** | **0.911** |
| Deployment op point (dev threshold s.t. FPR≤1% → test) | cutoff 35: **recall 36.4% / FPR 0.0%** |
| Max-Youden (unshippable, for contrast) | cutoff 10: recall 72.7% / FPR 15.3% |

Per-model recall (lower = evades better): Claude 10% · Gemini 20% · GPT-5.5 30% · Grok 40%.

**Findings:**
1. **No formality OR topic confound** — 0% FPR on news, informal, humanities,
   AND real student writing. The disguised-AI miss is a genuine signal gap.
2. **Formal STEM abstracts are the hardest human** — the sole false-positive is a
   dense physics abstract (the "abstracts structurally resemble AI" case).
3. **Recalibration is not the fix** — at a shippable ≤1% FPR, v2.0 recall is only
   36.4%, and disguised recall stays ~0%. AUC 0.91 is buoyed by *default* AI;
   disguised AI scores (~8–16%) overlap the human band (~8%), so no threshold
   separates them. Only the model learning the distribution can.

## How to read v2.1 (PRE-REGISTERED before seeing the number)

Run on the SAME frozen set + script. v2.1 is a **genuine win** iff:
- **Primary — disguised recall** climbs well off 0% at the ≤1% FPR operating
  point (target ≥50%). This is the whole point of the retrain.
- **AUC** rises meaningfully above 0.911 (target ≥0.95) AND the gain shows up in
  the *disguised* region, not only default AI.
- **Deployment recall (at FPR≤1%)** rises well above 36.4%.
- **No FPR regression on ANY register** — check per-register, not just aggregate.
  **arXiv is the canary:** if disguised recall rises but arXiv FPR rises
  disproportionately vs the other four, v2.1 learned to flag *AI-like formality*,
  not *AI* → partial failure, not success.

Failure/caution signals: AUC up but disguised still ~0 (didn't learn the hard
case); any register (esp. student/informal) FPR jumping 0%→>5% (over-aggressive).

## v2.1 (after adversarial/multi-model training) — TBD

_Re-run with `PAPERGUARD_DETECTOR_MODEL=training/paperguard_v2_1` (after re-fitting
calibration) and fill in the same rows as v2.0 above, plus per-register FPR._
