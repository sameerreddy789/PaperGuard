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

## Decision boundary (PRE-REGISTERED — decide the action before the number)

Judged on the frozen external benchmark at the **≤1% FPR operating point**, with
per-register FPR checked (arXiv canary):

- **SHIP as a clear win** — disguised recall **≥50%** AND no FPR regression on any
  register (arXiv ≤2.5%, others ~0%).
- **SHIP as an interim improvement** — disguised recall **25–50%** (a large jump
  from v2.0's 0%) AND no FPR regression on any register. Rationale: strictly
  better than v2.0 with no downside → deploy, queue v2.2 for the rest.
- **DO NOT ship → iterate to v2.2** — disguised recall **<25%** (marginal), OR the
  AUC gain is only from default AI (disguised still ~0).
- **HARD BLOCK regardless of recall** — any of: arXiv FPR rises disproportionately
  vs the other four registers (= learned to flag *formality*, not AI);
  student/informal/humanities FPR jumps 0%→>2%; overall deploy-FPR worse than
  v2.0. (Falsely accusing real human writing is the cancel-the-product failure.)

**Overfitting check (epoch-1 vs epoch-2):** if epoch-2 held-out RAID recall comes
back **below** epoch-1's 92.1% even as training loss keeps dropping, the second
pass is overfitting training-specific patterns — prefer the epoch-1 checkpoint.

## v2.1 (after adversarial/multi-model training) — FAILED (HARD BLOCK)

Trained on 240,988 balanced samples (RAID adversarial + Ateeqq + pile), 2 epochs,
RTX 3050. Held-out RAID AI-recall: ep1 92.1%, ep2 93.9%. Scored on the SAME frozen
set + script + env pattern as v2.0. **Result: catastrophic FPR regression on both
epoch checkpoints. Trips the pre-registered HARD BLOCK. NOT shippable.**

Both epoch checkpoints scored on the SAME frozen set + script + env pattern as
v2.0, each with its correct tokenizer (see the tokenizer caveat below).

| Metric | v2.0 (deployed) | v2.1 epoch-1 (ckpt-30124) | v2.1 epoch-2 = saved `paperguard_v2_1` |
|---|---|---|---|
| **AUC (5-register)** | **0.911** | **0.398** | **0.391** |
| Overall human FPR (cutoff 50) | 0.5% (1/200) | **100% (200/200)** | **70% (140/200)** |
| AI recall — disguised | 0% | (all flagged) | 30% |
| Deploy op point (FPR≤1% dev→test) | recall 36.4% / FPR 0.0% | recall 18.2% / FPR 32.7% | recall 13.6% / FPR 17.3% |

Per-register FPR (saved `paperguard_v2_1`, cutoff 50) — *these are real; see the
"accuracy vs FPR" correction below*:

| Register | v2.0 FPR | v2.1 saved FPR |
|---|---|---|
| arXiv (formal STEM) | 2.5% | **100%** |
| news | 0% | **77.5%** |
| informal (Yelp) | 0% | **80%** |
| student | 0% | **70%** |
| humanities (Gutenberg) | 0% | **22.5%** |

**What v2.1 actually is:** a model that ranks external human text *above* external
AI text (AUC < 0.5 on both epochs). It over-flags nearly every modern human
register (arXiv/news/Yelp/student all 70–100% FPR; only old literary Gutenberg is
spared-ish at 22.5%) while *passing* the 2025-model disguised AI (scores 35–45%,
below threshold). It over-fit to RAID-AI surface features that also fire on modern
human prose but miss casual human-styled AI. **Both epochs are consistently bad
(AUC 0.40 vs 0.39) — there is no drift, no collapse, no salvageable checkpoint.**

**Root cause — the held-out training eval had NO human class.**
`metric_for_best_model="auc"` returned `nan` every epoch because the held-out set
was AI-only (`Only one class present in y_true`). So the 92–94% "held-out recall"
NEVER measured false positives — the external-distribution FPR blowup was
invisible during the entire run. `best_model_checkpoint` stayed `null`, so the
saved `paperguard_v2_1` is just the *last* (epoch-2) weights (confirmed:
byte-identical MD5 to `checkpoint-60248`).

**Ruled out (not the cause):**
- *Label/index inversion* — `id2label` is identical in v2.0 and v2.1
  (`0=ai, 1=human`); detector reads the same slot. And held-out RAID AI-recall is
  93.9%, so labels are correct on RAID — this is a domain-shift/over-fit failure,
  not a flipped label.
- *Miscalibration* — AUC is a monotonic (calibration-invariant) ranking metric;
  both epochs are < 0.5, so no MIDPOINT/SCALE re-fit can recover them.

**Two corrections to the first-pass reading of these results (both were my error,
not model behavior):**
1. *"Epoch-2 collapsed to 100% FPR / constant-AI"* — WRONG. That run pointed at
   `checkpoint-60248`, which has **no tokenizer files**; `AutoTokenizer` fell back
   to a mismatched tokenizer → garbage token IDs → saturated "~100% AI on
   everything." An artifact of the missing tokenizer, not the weights. With the
   correct tokenizer, epoch-2 = the saved model (AUC 0.391, FPR 70%).
2. *"The per-register FPR line is a script bug printing 0.0%"* — WRONG, no bug.
   The `=== By group ===` line reports **accuracy (correct-rate) per group**, not
   FPR. For a human group, `0.0%` = 0% correct = **100% flagged = 100% FPR**. The
   numbers were self-consistent all along (correct humans 0+31+8+9+12 = 60/200 →
   70% FPR, matching the overall line). The arXiv canary works fine.

> **Operational note for v2.2 benchmarking:** raw `checkpoint-*` dirs from the
> Trainer do **not** contain tokenizer files. Copy `tokenizer.json` +
> `tokenizer_config.json` into any checkpoint before pointing the benchmark at it,
> or you'll measure a mismatched-tokenizer artifact (silently — it won't error).

**Decision (per pre-registered criteria):** HARD BLOCK — overall deploy-FPR is
catastrophically worse than v2.0 on both checkpoints (AUC below random). **v2.1 is
discarded. v2.0 (`mega_dataset_model_v2`) remains the production model; the HF
deployment is NOT overwritten.**

**For v2.2 (the actual fix):**
- The training eval set MUST contain **both classes** (balanced human + AI from
  held-out registers) so AUC/FPR are real numbers and early-stopping can select on
  them. If AUC comes back `nan`, that must raise a loud training-time alarm — not
  be silently tolerated (this would have caught the failure at epoch 1, not after
  5+ GPU-hours + external audit).
- Add diverse human negatives (news, informal, student, humanities — not just
  pile) to the **training** set, not only the eval, so the model can't satisfy the
  objective by flagging modern human prose.
- Re-check the arXiv canary (and all five registers) every epoch.
