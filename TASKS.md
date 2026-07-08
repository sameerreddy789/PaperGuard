# PaperGuard — Tasks & Strategy (future work)

> **Workflow:** this file holds *planned / in-progress* work. When a task is
> finished, move it to `COMPLETION_STATUS.md`. New tasks are added here first.
> (This file replaces the old `IMPLEMENTATION_PLAN.md`.)

**Runway:** ~8 days to submission. **Core goal:** mimic Turnitin's two headline
outputs — an **AI-writing %** and a **Similarity/plagiarism %** — at comparable
accuracy, and add our differentiator: **citation claim verification**.

**Owners:** model training → **@sameerreddy** · everything else (agents, UI,
plagiarism, deployment, docs) → **@technical-monish**.

---

## Snapshot (where we are)

- **Detector = v2.0 "mega"** — DistilBERT (cased, ~66M params, 6 layers),
  deployed at [`vediumsameer/paperguard-ai-detector`](https://huggingface.co/vediumsameer/paperguard-ai-detector).
  Trained (from v1.5) on Claude-Opus-4.8-distill (5k) + Ateeqq academic (6k) +
  artem9k `ai-text-detection-pile` (250k), 2 epochs, `eval_loss=0.0003`.
- **⚠️ The 0.0003 eval loss is a red flag, not a trophy.** It means the model
  overfit an *easy, separable* distribution (the pile). That is exactly why its
  softmax saturates (~0% AI on everything) and why it fails on out-of-
  distribution text. We already work around saturation by scoring off the
  **calibrated logit margin**, and the model now separates clean/academic AI
  (~70–90%) from human (~10%).
- **AI detection = model-only** (calibrated margin + embedding patchwork). No LLM.
- **Other layers:** citation (4-tier + claim verify), plagiarism (Serper +
  scholarly + LLM similarity), quality. CrewAI orchestrates; LLM (Gemini today,
  Qwen-ready) does reasoning/synthesis only.

---

## Model Training — v2.1  (Owner: @sameerreddy)

### Verdict on "train with 10M docs?"
**No — not worth it.** DistilBERT (66M params) converges early; the extra ~9.75M
docs are mostly redundant *easy* samples we already saturate on, so accuracy gains
are marginal. And 11+ days at 100% on an RTX 3050 is a real thermal/stability risk.
For detection, **diversity + difficulty beat raw volume** (RAID/M4GT literature).

### How many MORE quality docs to add
Sweet spot: add **~250k–500k NEW curated hard/diverse docs** on top of the current
~261k → a **~500k–750k total pool**. Beyond ~750k a 66M DistilBERT plateaus; more
data won't move the needle. (For a bigger jump, swap the base to
`microsoft/deberta-v3-base` or `roberta-base` — stronger, but more compute.)

### Curated ~500k pool (balanced ~50/50 AI/human, all HF-loadable)

| Slice | Size | Why |
|:---|:---:|:---|
| `liamdugan/raid` (multi-model + **12 adversarial attacks**) | ~150k | Robustness to masked/paraphrased AI (the current blind spot) |
| M4GT / SemEval-2024 Task 8 (incl. mixed human-machine) | ~100k | Multi-domain/model + patchwork signal |
| DeepSeek-R1 / reasoning CoT sets (+ matched human) | ~50k | Detect o1 / R1 / Claude-thinking style |
| `Ateeqq` + arXiv human academic | ~100k | Keep ESL / false-positive rate low |
| Retained from current pile | ~100k | Stability / guard against catastrophic forgetting |

### Training schedule (RTX 3050 — "1 epoch ≈ 150k")
1. Cap each epoch to **~150k** (subsample the 500k pool per epoch) → ~8h/epoch, safe thermals.
2. **2–3 epochs** (rotate subsamples), lr 1–2e-5, fp16, class-balanced, weight_decay 0.01.
3. Continue-train from v2.0 first (faster); fall back to `distilbert-base-cased` if it won't budge.
4. **Add real metrics** — accuracy / F1 / AUC / **FPR on a HELD-OUT set** (unseen
   models + adversarial). This is the success gauge, NOT `eval_loss`.
5. Re-fit calibration with `fit_calibration.py` on the new held-out set; push
   **v2.1** to HF and **update the (stale v1.5) model card**.

### Expected outcome (set expectations)
- Easy-text accuracy stays ~99% (already saturated — won't visibly improve).
- Real wins: **much lower false positives on human/ESL** and **much better recall
  on adversarial / masked / reasoning AI** — i.e. we fix the blind spot, not a
  vanity metric. Frame the demo around robustness, not raw accuracy.

### Tasks (@sameerreddy)
- [ ] Build the curation script (merge + balance + dedup the 5 slices → ~500k pool).
- [ ] Add held-out eval set (unseen models + RAID adversarial) + metrics to the trainer.
- [ ] Run 2–3 epochs @150k; log accuracy/F1/AUC/FPR per epoch.
- [ ] Re-fit calibration; push v2.1 to HF; update model card.

---

## Other layers (detector is our strength; lift the rest toward Turnitin)  (Owner: @technical-monish)

### Plagiarism — highest priority for Turnitin parity
- [ ] Deterministic **verbatim overlap**: n-gram / MinHash / Rabin-Karp
  fingerprinting of paragraphs vs retrieved sources (catches copy-paste, no LLM).
- [ ] **Semantic similarity** via sentence embeddings (reuse the detector encoder
  or add sentence-transformers) against retrieved candidates.
- [ ] Broaden retrieval: exact-phrase Serper queries for suspicious sentences;
  more open-access scholarly sources.
- [ ] **Cross-agent dedupe**: if a matched span is properly quoted + cited
  (citation agent), downgrade it from "plagiarism".
- [ ] Emit a **Similarity %** + per-source breakdown + highlighted spans.

### Citation — already strong; small adds
- [ ] DOI metadata consistency check (year/author/title mismatch = tampered).
- [ ] Retracted-paper check (Crossref / Retraction Watch).

### Quality — keep as is (minor).

---

## Output optimization / leverage  (Owner: @technical-monish)

- [ ] **Integrity Dashboard**: two Turnitin-style headline numbers (AI % +
  Similarity %) + Citation Health + patchwork flags, up top.
- [ ] Single **per-paragraph overlay** combining AI heat + plagiarism highlight
  + patchwork in one view.
- [ ] Turnitin-style **annotated PDF** (highlighted spans, per-source list).
- [ ] Confidence bands + honest disclaimers everywhere.

---

## Alibaba Cloud deployment (MANDATORY)  (Owner: @technical-monish)

- **Reasoning LLM → Qwen via DashScope** (OpenAI-compatible, LiteLLM-supported).
  Crew LLM is already configurable: set `PAPERGUARD_CREW_MODEL=dashscope/qwen-plus`
  (+ `DASHSCOPE_API_KEY`, optional `PAPERGUARD_CREW_API_BASE=https://dashscope-intl.aliyuncs.com/compatible-mode/v1`).
- [ ] Add a Qwen/OpenAI-compatible backend to `services/gemini.py` (or a new
  `services/llm.py`) so the **sub-agent tasks** (citation/quality/plagiarism/
  reference) also run on Qwen → fully Alibaba-native.
- [ ] **Containerize** (Dockerfile) the Streamlit app + detector.
- [ ] Deploy on Alibaba: **Function Compute 3.0** (serverless GPU, containers
  ≤30 GB, scale-to-zero, 24 h tasks) *or* **PAI-EAS** (one-click model serving) —
  ECS GPU VM as the simplest fallback.
- [ ] Live smoke-test end-to-end on the deployed URL.

---

## Prioritized 8-day plan

Two tracks run in parallel: **@sameerreddy** owns training; **@technical-monish**
owns everything else.

| Day | @sameerreddy (training) | @technical-monish (everything else) |
|:---|:---|:---|
| 1–2 | Curate ~500k hard/diverse pool + build held-out eval set | Plagiarism upgrade (fingerprint + semantic + cross-agent dedupe); Alibaba account + Qwen wiring |
| 2–4 | Run 2–3 epochs @150k; log accuracy/F1/AUC/FPR | Integrity Dashboard (two headline numbers + combined overlay) |
| 4–5 | Eval v2.1 on adversarial/held-out; re-fit calibration; push to HF + update card | Turnitin-style annotated PDF; citation adds (DOI consistency, retraction check) |
| 5–6 | Support integration of v2.1; sanity-check scores in the app | Containerize; add Qwen backend for sub-agents |
| 6–7 | (buffer / optional 2nd run if metrics lag) | Deploy on Alibaba (FC/PAI); Qwen end-to-end; live smoke-test |
| 7–8 | Final model card + eval writeup for the demo | Real-paper testing (two-column IEEE), edge cases, docs, demo polish |

---

## Tracked risks / blind spots

- Fully style-masked whole-document AI can score low from the model alone
  (RAID-adversarial training + patchwork detection mitigate, not eliminate).
- Plagiarism can't match Turnitin's private student-paper DB — scope to open web
  + open-access scholarly and say so honestly.
- Free-tier compute limits: torch + crewai + chromadb is heavy; size the Alibaba
  instance accordingly (or serve the detector via HF Inference).
- No valid `GEMINI_API_KEY` in `.env` currently (or use Qwen/DashScope).
