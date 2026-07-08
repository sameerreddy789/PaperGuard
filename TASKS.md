# PaperGuard — Tasks & Strategy (future work)

> **Workflow:** this file holds *planned / in-progress* work. When a task is
> finished, move it to `COMPLETION_STATUS.md`. New tasks are added here first.
> (This file replaces the old `IMPLEMENTATION_PLAN.md`.)

**Runway:** ~8 days to submission. **Core goal:** mimic Turnitin's two headline
outputs — an **AI-writing %** and a **Similarity/plagiarism %** — at comparable
accuracy, and add our differentiator: **citation claim verification**.

**Owners:** model training is **closed** (@sameerreddy — v2.0 final). All remaining
work (agents, UI, plagiarism, deployment, docs) → **@technical-monish**.

---

## Snapshot (where we are)

- **Detector = v2.0 "mega"** — DistilBERT (cased, ~66M params, 6 layers),
  deployed at [`vediumsameer/paperguard-ai-detector`](https://huggingface.co/vediumsameer/paperguard-ai-detector).
  **v2.0 is confirmed the best detector we have** (frozen benchmark AUC 0.911,
  human FPR 0.5%). Two retrains (v2.1, v2.2) were attempted and **both failed** —
  see below and `COMPLETION_STATUS.md` Phase 1.6.
- **Retraining is DONE (and did not beat v2.0).** v2.1 failed the frozen
  benchmark (AUC 0.391, FPR 70%); v2.2 fixed the training methodology (honest
  two-class eval) but still failed (AUC 0.458, FPR 56%). Continued RAID training
  **degrades** v2.0 rather than improving it. **v2.0 stays deployed; HF is not
  overwritten.**
- **AI detection = model-only** (calibrated margin + embedding patchwork). No LLM.
- **Other layers:** citation (4-tier + claim verify), plagiarism (Serper +
  scholarly + LLM similarity), quality. CrewAI orchestrates; LLM (Gemini today,
  Qwen-ready) does reasoning/synthesis only, now at **temperature 0.0** for
  reproducible reports (`PAPERGUARD_LLM_TEMPERATURE`).

---

## Model Training — DONE (v2.1 + v2.2 both failed; v2.0 retained)  (Owner: @sameerreddy)

Two retrains were built, run, and benchmarked. **Neither beat v2.0.** Full write-up
in `benchmark_results.md`; summary in `COMPLETION_STATUS.md` Phase 1.6.

### What was done ✅
- [x] Built a **frozen external benchmark** (`benchmark_samples.json`,
  `benchmark_detector.py`): 40 AI (Gemini/Claude S5/GPT-5.5/Grok, default+disguised)
  + 200 human across 5 registers; AUC, per-register FPR, dev/test split, arXiv canary.
- [x] Trained **v2.1** (RAID adversarial + Ateeqq + pile, 2 epochs). → **FAILED**:
  benchmark AUC 0.391, FPR 70%. Root cause: **AI-only held-out eval** → `eval_auc=nan`
  every epoch → FPR never measured → invisible over-flagging.
- [x] Built **`train_v2_2.py`** (fixed pipeline): balanced two-class multi-register
  held-out eval + **nan-AUC abort callback** + pre-train two-class assertion +
  benchmark-leak guard + push safety gate. Verified end-to-end.
- [x] Trained **v2.2**. Held-out eval is now honest (AUC 0.986, FPR 1.4%) — the
  methodology bug is fixed — but the frozen benchmark still **FAILED**: AUC 0.458,
  FPR 56%.

### The key learning 🔑
- The eval methodology is fixed, but the model is a **data problem**: RAID
  adversarial text **does not transfer** to 2025-model disguised AI, and continued
  training on it **degrades** v2.0 (catastrophic forgetting). v2.0 (0.911) >> v2.1
  (0.391) ≈ v2.2 (0.458) on the benchmark.
- **Held-out ≠ benchmark.** A great held-out number (v2.2: 0.986) told us nothing
  about real 2025-model text. The frozen benchmark is the only decision gauge.

### Decision: training chapter is CLOSED 🚩
No further retraining (no v2.3). Two honest attempts showed continued training
degrades v2.0 on the target task, and v2.0 is already our best detector. **v2.0 is
final for this project.** All remaining effort goes to the agents/product and
deployment, where the real, defensible value is. AI-detection is framed as one
calibrated signal (with a "needs human review" band), not a verdict.

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

**Training is complete and closed** (v2.0 retained). All remaining work is
agent/product + deployment — now a single track.

| Day | Product + deployment |
|:---|:---|
| 1–2 | Orchestrator fix (always surface hard facts); agent-centric report reframe (3-band + "needs human review"); Plagiarism upgrade (fingerprint + semantic + cross-agent dedupe) |
| 2–4 | Integrity Dashboard (two headline numbers + combined per-paragraph overlay); Alibaba account + Qwen wiring |
| 4–5 | Turnitin-style annotated PDF; citation adds (DOI consistency, retraction check) |
| 5–6 | Containerize; add Qwen backend for the sub-agent tasks |
| 6–7 | Deploy on Alibaba (FC/PAI); Qwen end-to-end; live smoke-test |
| 7–8 | Real-paper testing (two-column IEEE), edge cases, docs, demo polish |

---

## Tracked risks / blind spots

- **Confirmed, unresolved:** fully style-masked / disguised 2025-model AI evades
  the detector (v2.0 disguised recall 0%), and RAID retraining did **not** fix it
  (v2.1/v2.2 failed the benchmark). Mitigation for now: patchwork detection +
  surfacing the "Uncertain" band for human review, and framing AI-detection as one
  signal, not a verdict. A real fix needs v2.3 trained on 2025-model disguised data.
- Plagiarism can't match Turnitin's private student-paper DB — scope to open web
  + open-access scholarly and say so honestly.
- Free-tier compute limits: torch + crewai + chromadb is heavy; size the Alibaba
  instance accordingly (or serve the detector via HF Inference).
- No valid `GEMINI_API_KEY` in `.env` currently (or use Qwen/DashScope).
