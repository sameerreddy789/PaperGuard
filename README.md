# 🛡️ PaperGuard

> **Multi-Agent Academic Paper Verification System**
> A free, open-source tool that checks plagiarism, AI-generated content, citation validity, and writing quality — using a society of collaborating AI agents.

---

## ⚠️ IMPORTANT: Before Every Push

> **You MUST update [`COMPLETION_STATUS.md`](./COMPLETION_STATUS.md) before every git push.**
> Mark what's done, what's in progress, and what's next. This keeps the team synchronized.

```bash
# Before every push:
# 1. Update COMPLETION_STATUS.md
# 2. Then commit and push
git add .
git commit -m "your message"
git push origin main
```

---

## 🎯 What Makes PaperGuard Different?

| Feature | Turnitin | GPTZero | **PaperGuard** |
|:---|:---:|:---:|:---:|
| Plagiarism Detection | ✅ (proprietary DB) | ❌ | ✅ (open sources) |
| AI Content Detection | ✅ | ✅ (trained model) | ✅ (LLM classifier + burstiness) |
| Citation Existence Check | ❌ | ❌ | ✅ |
| **Citation Claim Verification** | ❌ | ❌ | **✅ ⭐ Killer Feature** |
| Writing Quality Analysis | ❌ | ❌ | ✅ |
| Free & Open Source | ❌ ($$$) | ❌ (freemium) | **✅ 100% Free** |

**Our killer feature:** We don't just check if a reference exists — we verify that the cited paper's abstract actually *supports* the claim being made. No mainstream tool does this.

---

## 🏗️ Tech Stack

| Component | Choice |
|:---|:---|
| **Language** | Python |
| **LLM** | Gemini 3.1 Flash Lite (free tier) |
| **PDF Parsing** | pymupdf4llm (layout-aware, handles two-column papers) |
| **Reference APIs** | CrossRef (primary, unlimited) + Semantic Scholar (abstracts) |
| **Web Search** | Serper API (2,500 free credits/month) |
| **Web Framework** | Streamlit |
| **Deployment** | Streamlit Community Cloud (v1) |
| **Orchestration** | Custom Python orchestrator (no framework) |

### 📚 Training Datasets (AI Detection Model)
Our AI detection model is locally fine-tuned on a massive aggregation of the following open-source datasets (totaling 125,000+ samples) to ensure robust detection across all frontier LLMs (GPT-4o, LLaMA-3, Claude, Gemini, etc.):
- [Rajarshi-Roy-research/Defactify_Text_Dataset](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Text_Dataset)
- [Ateeqq/AI-and-Human-Generated-Text](https://huggingface.co/datasets/Ateeqq/AI-and-Human-Generated-Text)
*(Note: To save space and bandwidth, the raw dataset files and cached checkpoints are explicitly `.gitignore`'d and are not hosted in this GitHub repository).*

---

## 📁 Project Structure

```
paperguard/
├── app.py                    # Streamlit web app entry point
├── requirements.txt          # Python dependencies
├── .env.example              # API keys template
│
├── core/
│   ├── __init__.py
│   ├── pdf_parser.py         # pymupdf4llm text extraction
│   ├── text_chunker.py       # Split into sections → paragraphs → sentences
│   └── reference_parser.py   # LLM-based reference extraction
│
├── services/
│   ├── __init__.py
│   ├── gemini.py             # Gemini 3.1 Flash Lite wrapper
│   ├── semantic_scholar.py   # Semantic Scholar API
│   ├── crossref.py           # CrossRef API
│   ├── serper.py             # Serper web search
│   └── cache.py              # File-based JSON cache
│
├── agents/
│   ├── __init__.py
│   ├── orchestrator.py       # Custom orchestrator
│   ├── citation_agent.py     # Citation verification (⭐ killer feature)
│   ├── ai_detection_agent.py # AI detection (LLM classifier + burstiness)
│   ├── plagiarism_agent.py   # Plagiarism checking
│   └── quality_agent.py      # Writing quality assessment
│
├── models/
│   ├── __init__.py
│   ├── report.py             # Report data model
│   └── reference.py          # Reference data model
│
├── tests/
│   ├── __init__.py
│   ├── test_pdf_parser.py
│   ├── test_services.py
│   ├── test_citation_agent.py
│   └── sample_papers/
│       └── (test PDFs go here)
│
├── cache/                    # Cached API responses (gitignored)
│
├── IMPLEMENTATION_PLAN.md    # Full technical plan
├── COMPLETION_STATUS.md      # Current progress tracker
└── README.md                 # This file
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- API Keys (all free):
  - [Gemini API Key](https://aistudio.google.com/) (Google AI Studio)
  - [Semantic Scholar API Key](https://www.semanticscholar.org/product/api) (free, optional but recommended)
  - [Serper API Key](https://serper.dev/) (2,500 free credits/month)

### Setup

```bash
# Clone the repo
git clone https://github.com/sameerreddy789/PaperGuard.git
cd PaperGuard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the app
streamlit run app.py
```

### CLI Mode (for testing without UI)
```bash
# Test PDF parsing
python -m core.pdf_parser path/to/paper.pdf

# Test citation verification
python -m agents.citation_agent path/to/paper.pdf

# Run full analysis
python main.py path/to/paper.pdf
```

---

## 📊 How It Works

```
Upload PDF
    │
    ▼
┌─────────────────────┐
│  pymupdf4llm        │  Layout-aware extraction
│  (handles 2-column) │  (no gibberish text)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Reference Parser   │  LLM extracts structured
│  (Gemini Flash Lite)│  references from text
└─────────┬───────────┘
          │
    ┌─────┼─────┬──────────┐
    ▼     ▼     ▼          ▼
  Citation  AI    Plagiarism  Writing
  Agent   Detection  Agent    Quality
  (⭐)    Agent              Agent
    │     │     │          │
    └─────┼─────┴──────────┘
          ▼
┌─────────────────────┐
│  Conflict Resolver  │  Agents disagree?
│  (Custom Pipeline)  │  Orchestrator decides.
└─────────┬───────────┘
          │
          ▼
    📋 Final Report
    📥 Download as PDF
```

---

## ⚠️ Honest Limitations

- **Not a Turnitin replacement.** We check open-access papers and web sources only. We don't have access to Turnitin's proprietary database of 1B+ student papers.
- **AI detection is an indicator, not a verdict.** We use LLM classification + burstiness math. It's less accurate than GPTZero's trained model.
- **Best used as a pre-submission self-check.** "Catch issues before your professor does."

---

## 📝 License

MIT License — free to use, modify, and distribute.

---

## 🤝 Contributing

1. Update `COMPLETION_STATUS.md` before every push
2. Follow the phase structure in `IMPLEMENTATION_PLAN.md`
3. Test with real academic papers before marking features as complete
4. Be honest about limitations in the UI
