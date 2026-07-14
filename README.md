# PaperGuard

> **Multi-Agent Academic Integrity Verification**
> An open, agent-based alternative to Turnitin: it reports an **AI-writing %** and
> a **plagiarism/similarity %**, and — uniquely — verifies that each citation's
> source actually *supports* the claim it's attached to. Built for student work
> and research-paper submissions (IEEE, ACM, Elsevier).

---

## Why PaperGuard

Turnitin gives two headline numbers (AI % and Similarity %) but is closed and
checks neither whether a citation *exists* nor whether it *supports* the claim.
PaperGuard reproduces the two headline numbers with open tooling and adds
citation claim verification.

| Capability | Turnitin | GPTZero | **PaperGuard** |
|:---|:---:|:---:|:---:|
| Plagiarism / similarity % | Yes (private DB) | No | Yes (open web + scholarly) |
| AI-writing % | Yes | Yes | Yes (own trained model + stylometry) |
| Citation existence check | No | No | Yes |
| **Citation claim verification** | No | No | **Yes (differentiator)** |
| Writing-quality analysis | No | No | Yes |
| Open source | No | No | **Yes** |

---

## How AI detection works (model-only)

AI detection is done by
[`desklib/ai-text-detector-v1.01`](https://huggingface.co/desklib/ai-text-detector-v1.01)
(a fine-tuned `microsoft/deberta-v3-large` that leads the RAID benchmark).
**No LLM is used for detection** — LLMs are reserved for agent orchestration
and the other tasks.

We previously trained and deployed our own DistilBERT (v2.0). It was replaced
after a head-to-head benchmark against desklib v1.01 and a second external
candidate (mdrakibali/deberta-ai-detector-v3) on the same frozen 240-sample
benchmark (`benchmark_samples.json`; see `benchmark_results.md` and
`PROJECT_REPORT.md` Section 1):

| Metric | v2.0 (previous, in-house) | desklib v1.01 (current) |
|---|---:|---:|
| AUC | 0.911 | **0.968** |
| Disguised-AI recall | 0% | **75%** |
| Human FPR @ deployment threshold | 0.5% | ~0.5% (at cutoff ~90) |

1. **Direct sigmoid classifier output.** Unlike v2.0's DistilBERT (whose raw
   softmax was saturated and needed a logit-margin recalibration), desklib's
   single-logit sigmoid output was not found to be saturated on our benchmark.
   Instead of recalibrating, the "Likely AI" decision threshold is raised from
   the model's naive 50% default to ~90 — the FPR/recall sweet spot measured on
   the frozen benchmark (~0.5% human FPR at ~85–87.5% AI recall, vs. 7.0% FPR
   at 50%).
2. **Stylometric patchwork detection.** Paragraph embeddings from the same
   model are compared to the document's overall style; strong outliers (robust
   median/MAD) are flagged as possible mixed authorship — AI pasted into human
   writing.

Output: a per-paragraph AI heatmap + patchwork flags. Known blind spot
(reduced, not eliminated): style-masked/disguised AI still evades the detector
25% of the time (down from 100% with v2.0) — mitigated, not eliminated, by
patchwork detection.

---

## The agent society (CrewAI)

Orchestrated as a **CrewAI** crew (agents-as-tools). Deterministic work (API
lookups, model inference, math) stays in tools; the LLM only reasons and
synthesises, so fact-lookups are trustworthy.

| Agent | Role | Tools |
|:---|:---|:---|
| Citation Verifier | Existence + 4-tier claim support | CrossRef, Semantic Scholar, LLM claim check |
| AI-Detection Analyst | Per-paragraph AI % + patchwork | Fine-tuned detector, embeddings (no LLM) |
| Plagiarism Checker | Overlap with open web + scholarly | Serper, CrossRef, Semantic Scholar, LLM similarity |
| Writing-Quality Reviewer | Structure, readability, prose | Readability math, LLM prose review |
| Orchestrator / Editor | Coordinates, resolves cross-agent conflicts, builds report | LLM synthesis |

---

## Tech stack

| Component | Choice |
|:---|:---|
| Language | Python 3.10+ |
| Agent framework | CrewAI (agents-as-tools) |
| Reasoning LLM | Gemini (default); **Qwen via Alibaba DashScope** for deployment (LiteLLM) |
| AI-detection model | `desklib/ai-text-detector-v1.01` (deberta-v3-large, local via `transformers`) |
| PDF parsing | pymupdf4llm (layout-aware, two-column safe) |
| Reference APIs | CrossRef + Semantic Scholar |
| Web search | Serper |
| UI | Streamlit |
| Deployment target | **Alibaba Cloud** (Function Compute 3.0 / PAI-EAS) |

---

## Project structure

```
paperguard/
├── main.py                    # CLI: full analysis of a paper
├── app.py                     # Streamlit UI (heatmap, panels, PDF/JSON export)
├── requirements.txt
├── .env.example
│
├── core/                      # PDF / text / reference processing
│   ├── pdf_parser.py  text_chunker.py  reference_parser.py
│
├── services/                  # External APIs + cache
│   ├── gemini.py (google-genai)  crossref.py  semantic_scholar.py  serper.py  cache.py
│
├── agents/
│   ├── base.py                # shared BaseAgent + CLI harness
│   ├── citation_agent.py      # citation existence + claim verification
│   ├── detector_agent.py      # AI detector (desklib deberta-v3-large + embeddings)
│   ├── ai_detection.py        # model-only AI-detection engine (heatmap + patchwork)
│   ├── plagiarism_agent.py    # open-source overlap detection
│   ├── quality_agent.py       # writing-quality assessment
│   ├── crew_tools.py          # deterministic logic wrapped as CrewAI tools
│   └── orchestrator.py        # CrewAI crew (specialists + editor) + engine fallback
│
├── models/                    # Pydantic data models
├── tests/                     # Sample papers
├── fit_calibration.py         # Legacy margin-calibration tool (see file header)
├── benchmark_detector.py      # Frozen-benchmark harness for detector model comparisons
│
├── TASKS.md                   # Future work + strategy (start here for what's next)
├── COMPLETION_STATUS.md       # Completed work
└── README.md
```

---

## Quick start

```bash
git clone https://github.com/sameerreddy789/PaperGuard.git
cd PaperGuard
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env            # add keys (Gemini or Qwen/DashScope, Serper)

python main.py path/to/paper.pdf  # CLI
streamlit run app.py              # UI
```

Runs without any LLM key in a degraded mode (AI detector + CrossRef
citations still work). Individual agents are CLI-runnable, e.g.
`python -m agents.citation_agent tests/sample_papers/sample_paper.md`.

---

## Deployment (Alibaba Cloud)

- **LLM = Qwen via DashScope** (OpenAI-compatible, LiteLLM). Set:
  ```
  PAPERGUARD_CREW_MODEL=dashscope/qwen-plus
  DASHSCOPE_API_KEY=...
  PAPERGUARD_CREW_API_BASE=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
  ```
- **Hosting:** containerize and deploy on **Function Compute 3.0** (serverless
  GPU, scale-to-zero) or **PAI-EAS** (one-click model serving). See `TASKS.md`.

---

## Honest limitations

- Not a Turnitin replacement for plagiarism: coverage is open web + open-access
  scholarly sources, not Turnitin's private student-paper database.
- AI detection is a probabilistic indicator; fully style-masked AI can evade the
  model. Results are shown with confidence and, where relevant, reasoning.
- Best used as a **pre-submission self-check**.

## Docs

- **`TASKS.md`** — future work, strategy, and the 8-day plan.
- **`COMPLETION_STATUS.md`** — what's already done.

## License

MIT.
