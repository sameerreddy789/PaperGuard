# PaperGuard

> **Multi-Agent Academic Integrity Verification**
> An open, agent-based system that verifies citations, detects AI-generated
> text, checks for plagiarism, and assesses writing quality — built for authors
> and reviewers of academic work, from student assignments to research papers
> submitted to venues like IEEE, ACM, and Elsevier.

---

## Why PaperGuard

Existing tools each solve one slice of the problem and none verify whether a
citation actually supports the claim it is attached to. PaperGuard combines four
verification concerns into one coordinated multi-agent pipeline, and adds a
self-correcting "safety net" for AI detection so a single brittle model never
gets the last word.

| Capability | Turnitin | GPTZero | **PaperGuard** |
|:---|:---:|:---:|:---:|
| Plagiarism detection | Yes (proprietary DB) | No | Yes (open sources) |
| AI-content detection | Yes | Yes | Yes (model + LLM safety net) |
| Citation existence check | No | No | Yes |
| **Citation claim verification** | No | No | **Yes (headline feature)** |
| Writing-quality analysis | No | No | Yes |
| Open source | No | No | **Yes** |

**Headline feature:** we don't just check that a reference exists — we verify
that the cited work's abstract actually *supports* the claim being made. No
mainstream tool does this.

---

## The AI-Detection Safety Net

Raw statistical detectors are brittle. Our fine-tuned classifier
([`vediumsameer/paperguard-ai-detector`](https://huggingface.co/vediumsameer/paperguard-ai-detector),
v2.0) can suffer *mode collapse* — over-flagging rigid non-native (ESL) writing
as AI, or being fooled into "100% human" by style-masked AI text. Instead of
trusting it blindly, we wrap it in a cognitive safety net:

```
                 per paragraph
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
  Detector Agent              Linguistic Agent
  (PyTorch model,             (LLM reads tone,
   pure statistics)            structure, intent)
        │                           │
        └─────────────┬─────────────┘
                      ▼
              Conflict Resolver
   • Agree           → keep score, highlight paragraph
   • Model ~100% Human + LLM high AI  → override (style-masked AI caught)
   • Model ~100% AI  + LLM low AI     → override (ESL false-positive cleared)
   • Otherwise disagree               → weighted 40/60 consensus (favour context)
```

The result is a per-paragraph AI heatmap with a transparent, defensible verdict
— and reasoning attached wherever the LLM overrides the model.

---

## The Agent Society (CrewAI)

PaperGuard is orchestrated as a **CrewAI** multi-agent crew. Each agent owns a
concern and is equipped with deterministic **tools** (API lookups, PyTorch
inference, math). The LLM handles reasoning, synthesis, and conflict resolution
— never fact-lookups, which stay deterministic so results are trustworthy.

| Agent | Role | Tools it uses |
|:---|:---|:---|
| Citation Verifier | Existence + claim support (4-tier) | CrossRef, Semantic Scholar, LLM claim check |
| AI-Detection (Detector + Linguistic) | Per-paragraph AI likelihood + safety net | PyTorch model, LLM, burstiness math |
| Plagiarism Checker | Overlap with open web + scholarly sources | Serper, CrossRef, Semantic Scholar, LLM similarity |
| Writing-Quality Reviewer | Structure, readability, prose | Readability math, LLM prose review |
| Orchestrator / Conflict Resolver | Coordinates agents, resolves cross-agent conflicts, builds the report | — |

---

## Tech Stack

| Component | Choice |
|:---|:---|
| Language | Python 3.10+ |
| Agent framework | CrewAI (agents-as-tools pattern) |
| Reasoning LLM | Gemini (free tier); pluggable (Qwen-ready) |
| AI-detection model | `vediumsameer/paperguard-ai-detector` (DistilBERT, v2.0, local via `transformers`) |
| PDF parsing | pymupdf4llm (layout-aware, handles two-column papers) |
| Reference APIs | CrossRef (unlimited) + Semantic Scholar (abstracts) |
| Web search | Serper |
| UI | Streamlit |
| Caching | File-based JSON cache |

### AI-detection training data
The detector was fine-tuned on an aggregation of open datasets (125k+ samples)
spanning many frontier LLMs (GPT-4o, LLaMA-3, Claude, Gemini, Mistral, Qwen),
including the Defactify text dataset, Ateeqq AI-vs-Human text, a Claude Opus
distillation set, and a slice of the AI-text-detection pile. Raw datasets and
checkpoints are gitignored (see `train_mega_dataset.py` for the pipeline).

---

## Project Structure

```
paperguard/
├── main.py                    # CLI entry: full analysis of a paper
├── app.py                     # Streamlit UI (planned)
├── requirements.txt
├── .env.example               # API keys template
│
├── core/                      # PDF/text/reference processing
│   ├── pdf_parser.py          # pymupdf4llm extraction
│   ├── text_chunker.py        # sections → paragraphs → sentences
│   └── reference_parser.py    # LLM + heuristic reference extraction
│
├── services/                  # External API wrappers (+ cache)
│   ├── gemini.py  crossref.py  semantic_scholar.py  serper.py  cache.py
│
├── agents/                    # Verification agents + orchestration
│   ├── base.py                # shared BaseAgent + CLI harness
│   ├── citation_agent.py      # citation existence + claim verification
│   ├── detector_agent.py      # PyTorch AI classifier (the "math")
│   ├── linguistic_agent.py    # LLM contextual AI analyst (the "brain")
│   ├── conflict_resolver.py   # AI-detection safety net logic
│   ├── ai_detection_agent.py  # burstiness signal (feeds the resolver)
│   ├── plagiarism_agent.py    # open-source overlap detection
│   ├── quality_agent.py       # writing-quality assessment
│   └── orchestrator.py        # CrewAI crew (planned)
│
├── models/                    # Pydantic data models (report, reference)
├── tests/                     # Sample papers + tests
├── train_mega_dataset.py      # Detector training pipeline (reference)
├── ood_stress_test.py         # Out-of-distribution validation gauntlet
│
├── IMPLEMENTATION_PLAN.md      # Technical plan (see Architecture Update at top)
├── COMPLETION_STATUS.md        # Progress tracker (update before every push)
└── README.md
```

---

## Quick Start

Prerequisites: Python 3.10+, and free API keys for Gemini and Serper (CrossRef
needs none; Semantic Scholar is optional but recommended).

```bash
git clone https://github.com/sameerreddy789/PaperGuard.git
cd PaperGuard

python -m venv venv
venv\Scripts\activate            # Windows;  source venv/bin/activate on macOS/Linux

pip install -r requirements.txt

copy .env.example .env           # then edit .env with your keys

# CLI (engine)
python main.py path/to/paper.pdf

# UI (once available)
streamlit run app.py
```

Individual agents are runnable standalone for testing (they accept `.pdf`,
`.md`, or `.txt`):

```bash
python -m agents.citation_agent   tests/sample_papers/sample_paper.md
python -m agents.detector_agent   tests/sample_papers/sample_paper.md
python -m agents.linguistic_agent tests/sample_papers/sample_paper.md
```

---

## Honest Limitations

- **Not a Turnitin replacement.** Plagiarism checks cover open-access papers and
  the open web, not Turnitin's proprietary student-paper database.
- **AI detection is an indicator, not a verdict.** The safety net makes it far
  more robust than a bare classifier, but results are probabilistic and always
  shown with confidence and reasoning.
- **Best used as a pre-submission integrity check** — for authors to catch
  issues (fabricated citations, unsupported claims, AI-pattern sections) before
  reviewers do.

---

## Contributing

1. Update `COMPLETION_STATUS.md` before every push.
2. Follow the architecture in `IMPLEMENTATION_PLAN.md` (see the Architecture
   Update section at the top for current decisions).
3. Keep fact-lookups deterministic (tools), reasoning in the LLM (agents).
4. Be honest about limitations in the UI.

## License

MIT.
