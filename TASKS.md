# PaperGuard — Tasks & Strategy (future work)

> **Workflow:** this file holds *planned / in-progress* work. When a task is
> finished, move it to `COMPLETION_STATUS.md`. New tasks are added here first.
> (This file replaces the old `IMPLEMENTATION_PLAN.md`.)

**Runway:** ~8 days to submission. **Core goal:** mimic Turnitin's two headline
outputs — an **AI-writing %** and a **Similarity/plagiarism %** — at comparable
accuracy, and add our differentiator: **citation claim verification**.

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

## Should we train more? YES — one focused run (maybe two)

Not "more easy data" — the model already maxes easy data. The goal is
**generalization** and closing the **slang/style-masked AI blind spot**.
Success metric shifts from "low loss on the pile" to **accuracy + low false-
positive-rate on held-out models and adversarial text**.

### Recommended datasets (all HuggingFace-loadable)

| Dataset | Why it helps |
|:---|:---|
| `liamdugan/raid` (10M+ docs, 11 LLMs, 11 genres, **12 adversarial attacks**) | Paraphrase/synonym/homoglyph/whitespace attacks — the direct fix for masked-AI evasion. Use a balanced adversarial subset. |
| M4GT / SemEval-2024 Task 8 (M4) | Multi-model, multi-domain; **subtask C = mixed human-machine** → trains the patchwork signal. |
| DeepSeek-R1 / reasoning CoT sets (R1-distill, OpenThoughts) | Reasoning-model outputs, so we detect o1 / R1 / Claude-thinking style text. |
| `Ateeqq/AI-and-Human-Generated-Text` (keep) | Academic prose → keeps ESL/false-positive rate down. |
| Comprehensive Human-vs-AI (arXiv 2510.22874) | Hard set (≈58% baseline) → good generalization test set. |

### Training plan (RTX 3050 ≈ 13h per 250k×2ep, or Alibaba PAI GPU)

1. Curate a balanced **hard** mix (~200–300k): RAID-adversarial subset + M4GT
   mixed + CoT/reasoning + Ateeqq academic + balanced human.
2. Continue-train from v2.0 (or restart from `distilbert-base-cased`), 2–3
   epochs, lr 1–2e-5, fp16, class-balanced.
3. Evaluate on **held-out RAID adversarial + unseen models**; report Accuracy +
   FPR. Add proper metrics (accuracy/F1/AUC) to the training script.
4. Re-fit calibration with `fit_calibration.py` on the new held-out set; push
   **v2.1** to HF and update the model card.

Feasible in the runway: 1 run comfortably, 2 if curation is quick.

---

## Other layers (detector is our strength; lift the rest toward Turnitin)

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

## Output optimization / leverage

- [ ] **Integrity Dashboard**: two Turnitin-style headline numbers (AI % +
  Similarity %) + Citation Health + patchwork flags, up top.
- [ ] Single **per-paragraph overlay** combining AI heat + plagiarism highlight
  + patchwork in one view.
- [ ] Turnitin-style **annotated PDF** (highlighted spans, per-source list).
- [ ] Confidence bands + honest disclaimers everywhere.

---

## Alibaba Cloud deployment (MANDATORY)

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

| Day | Focus |
|:---|:---|
| 1–2 | Plagiarism upgrade (fingerprint + semantic + cross-agent dedupe); Alibaba account + Qwen wiring |
| 2–4 | Curate hard dataset; launch training run (overnight) |
| 4–5 | Eval v2.1 on adversarial/held-out; re-fit calibration; push to HF + update card |
| 5–6 | Integrity Dashboard (combined overlay, two headline numbers) + annotated PDF |
| 6–7 | Containerize + deploy on Alibaba (FC/PAI); Qwen end-to-end; live smoke-test |
| 7–8 | Real-paper testing (incl. two-column IEEE), edge cases, docs, demo polish |

---

## Tracked risks / blind spots

- Fully style-masked whole-document AI can score low from the model alone
  (RAID-adversarial training + patchwork detection mitigate, not eliminate).
- Plagiarism can't match Turnitin's private student-paper DB — scope to open web
  + open-access scholarly and say so honestly.
- Free-tier compute limits: torch + crewai + chromadb is heavy; size the Alibaba
  instance accordingly (or serve the detector via HF Inference).
- No valid `GEMINI_API_KEY` in `.env` currently (or use Qwen/DashScope).
