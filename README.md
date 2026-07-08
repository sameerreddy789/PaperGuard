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
verification concerns into one coordinated multi-agent pipeline, powered by a
purpose-trained AI-detection model plus embedding-based stylometry.

| Capability | Turnitin | GPTZero | **PaperGuard** |
|:---|:---:|:---:|:---:|
| Plagiarism detection | Yes (proprietary DB) | No | Yes (open sources) |
| AI-content detection | Yes | Yes | Yes (own trained model + stylometry) |
| Citation existence check | No | No | Yes |
| **Citation claim verification** | No | No | **Yes (headline feature)** |
| Writing-quality analysis | No | No | Yes |
| Open source | No | No | **Yes** |

**Headline feature:** we don't just check that a reference exists — we verify
that the cited work's abstract actually *supports* the claim being made. No
mainstream tool does this.

---

## AI Detection

AI detection is done entirely by our own fine-tuned model
([`vediumsameer/paperguard-ai-detector`](https://huggingface.co/vediumsameer/paperguard-ai-detector),
DistilBERT, v2.0) — no LLM is used for detection. Two techniques make it robust:

1. **Calibrated logit margin.** The model's raw softmax is *saturated* (it
   reports ~0% AI even on real AI text). The discriminative signal actually
   lives in the logit margin (human − ai), which cleanly separates clean/
   academic AI (~6–8) from human text (~16–18). We score off a logistic
   calibration of that margin, so the detector correctly flags clean/academic AI
   (~70–90%) while keeping human text low (~10%).
2. **Stylometric patchwork detection.** Paragraph embeddings from the same model
   are compared against the document's overall style; paragraphs that deviate
   strongly (robust median/MAD outliers) are flagged as possible mixed
   authorship — "Frankenstein" AI text pasted into human writing.

The result is a per-paragraph AI heatmap plus patchwork flags. Known blind spot:
slang/style-masked AI can still read as human to the model; the patchwork check
partially mitigates this when such text is mixed into human writing.

> Gemini/LLMs are **not** used for AI detection — only for agent orchestration
> and the other tasks (citation claim checks, quality review, plagiarism
> similarity, reference parsing).

---

## The Agent Society (CrewAI)

PaperGuard is orchestrated as a **CrewAI** multi-agent crew. Each agent owns a
concern and is equipped with deterministic **tools** (API lookups, PyTorch
inference, math). The LLM handles reasoning, synthesis, and conflict resolution
— never fact-lookups, which stay deterministic so results are trustworthy.

| Agent | Role | Tools it uses |
|:---|:---|:---|
| Citation Verifier | Existence + claim support (4-tier) | CrossRef, Semantic Scholar, LLM claim check |
| AI-Detection Analyst | Per-paragraph AI likelihood + patchwork | Fine-tuned detector model, embeddings (no LLM) |
| Plagiarism Checker | Overlap with open web + scholarly sources | Serper, CrossRef, Semantic Scholar, LLM similarity |
| Writing-Quality Reviewer | Structure, readability, prose | Readability math, LLM prose review |
| Orchestrator / Editor | Coordinates agents, resolves cross-agent conflicts, builds the report | LLM synthesis |

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
│   ├── detector_agent.py      # fine-tuned AI detector (calibrated margin + embeddings)
│   ├── ai_detection.py        # model-only AI-detection engine (heatmap + patchwork)
│   ├── plagiarism_agent.py    # open-source overlap detection
│   ├── quality_agent.py       # writing-quality assessment
│   ├── crew_tools.py          # deterministic logic wrapped as CrewAI tools
│   └── orchestrator.py        # CrewAI crew (specialists + editor) + engine fallback
│
├── models/                    # Pydantic data models (report, reference)
├── tests/                     # Sample papers + tests
├── train_mega_dataset.py      # Detector training pipeline (reference)
├── fit_calibration.py         # Re-fit the detector's margin calibration
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
