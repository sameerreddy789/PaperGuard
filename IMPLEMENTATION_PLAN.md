# Implementation Plan: PaperGuard

> **Multi-Agent Academic Integrity Verification System**
> An open, agent-based tool that verifies citations, detects AI-generated
> content, checks plagiarism, and assesses writing quality — for both student
> work and research-paper submissions (IEEE, ACM, Elsevier, etc.).

---

## Architecture Update (v2) — authoritative, supersedes conflicting details below

> The sections further down capture the original plan and reasoning. Where they
> conflict with this update, **this section wins.**

### 1. Scope
PaperGuard is for academic integrity generally — not just students. Framing,
prompts, and UI copy are research-grade (e.g. "research-integrity check",
"pre-submission verification"), suitable for authors targeting IEEE/ACM/Elsevier
as well as coursework.

### 2. Orchestration = CrewAI (agents-as-tools)
We now use **CrewAI** as the multi-agent orchestration layer (this supersedes
the earlier "custom orchestrator, no framework" decision). Design rules:
- Each concern is a CrewAI `Agent` with a role/goal and a set of **tools**.
- All fact-lookups and computations stay **deterministic tools** (CrossRef,
  Semantic Scholar, Serper, the PyTorch detector, burstiness math, the conflict
  resolver). The LLM never performs lookups — only reasoning, synthesis, and
  conflict resolution. This keeps results trustworthy and reproducible.
- A coordinating agent/task assembles the final `models.report.Report`.

### 3. AI-Detection Safety Net (replaces the single AI-Detection agent design)
AI detection is a two-agent society with a resolver, run per paragraph:
- **Detector Agent** (`agents/detector_agent.py`) — the fine-tuned PyTorch
  model `vediumsameer/paperguard-ai-detector` (v2.0). Pure statistics. Fast but
  prone to mode collapse (ESL false-positives; "100% human" on style-masked AI).
- **Linguistic Agent** (`agents/linguistic_agent.py`) — an LLM reads tone,
  structure, and intent to catch the Detector's blind spots.
- **Conflict Resolver** (`agents/conflict_resolver.py`) — when the two diverge
  by >30, it overrides: adopt the LLM verdict on logit-saturation (model≈human,
  LLM=AI) or ESL false-positive (model=AI, LLM=human); otherwise a 40/60
  weighted consensus favouring context. Burstiness (`ai_detection_agent.py`)
  feeds the resolver as a tie-breaker.

Output: a per-paragraph AI heatmap with reasoning attached on every override.

### 4. Detector model details
`transformers` loads the model locally (auto-download + cache). Label index
`0 = AI`, `1 = human` (per model config). Graceful degradation: if
torch/transformers/model are unavailable, the Detector disables itself and the
Linguistic agent carries AI detection alone.

### 5. Reasoning LLM
Gemini (free tier) by default; the Linguistic agent takes a pluggable LLM
callable so Qwen (or another backend) can be swapped in without code changes.

---

## How Existing Platforms Work (What We're Learning From)

Before building, let's understand exactly what the big players do under the hood — so we can replicate the *approach* without their proprietary databases.

### Turnitin / iThenticate — Plagiarism Detection

```
How Turnitin Works (Simplified):

1. INGEST: Break submitted document into n-grams (overlapping word chunks)
         Example: "the quick brown fox" → ["the quick brown", "quick brown fox"]

2. HASH:   Convert each n-gram into a digital fingerprint (hash)
         "the quick brown" → 0x7F3A2B

3. COMPARE: Compare fingerprints against a massive index of:
         ├── Student paper repository (PROPRIETARY — we can't replicate this)
         ├── Internet crawl (billions of web pages)
         ├── Scholarly articles (CrossRef partnership)
         └── Licensed journals (PROPRIETARY)

4. SCORE:  Calculate similarity % = (matching words / total words) × 100

5. REPORT: Highlight matching sections, link to source documents
```

> [!IMPORTANT]
> **What we CAN replicate:** Web crawl matching (via search APIs) + scholarly matching (via Semantic Scholar + CrossRef — both free).
>
> **What we CAN'T replicate:** Their proprietary database of 1B+ student papers. This is Turnitin's real moat. We acknowledge this limitation honestly.

---

### GPTZero — AI Content Detection

```
How GPTZero Works (Simplified):

1. TOKENIZE: Break text into tokens (word fragments)

2. PERPLEXITY: For each sentence, measure "how surprised would a language
   model be by these word choices?"
   ├── LOW perplexity = very predictable = likely AI-written
   └── HIGH perplexity = unpredictable = likely human-written

3. BURSTINESS: Measure variation in sentence complexity across the document
   ├── LOW burstiness = uniform rhythm = likely AI (AI writes like a metronome)
   └── HIGH burstiness = varied rhythm = likely human (humans write messily)

4. CLASSIFY: Feed perplexity + burstiness scores into a trained classifier
   → Output: probability score (0-100%) that text is AI-generated

5. SENTENCE-LEVEL: Modern versions also score individual sentences
```

> [!IMPORTANT]
> **What we CAN replicate:** Perplexity and burstiness calculations don't need a trained model — they're statistical measures we can compute using any LLM's token probabilities. Gemini API returns log-probabilities which we can use directly.
>
> **What we CAN'T replicate (easily):** Their fine-tuned classifier trained on millions of labeled examples. Our version will be a simpler statistical approach — less accurate but still useful and honest about its limitations.

---

### CrossRef / Semantic Scholar — Citation Verification

```
How Citation Verification Works:

1. EXTRACT: Parse the References section from the paper
   → Get: author names, paper title, journal name, year, DOI (if present)

2. LOOKUP: For each reference:
   ├── If DOI present → CrossRef API: verify DOI exists & metadata matches
   ├── If no DOI → Semantic Scholar API: search by title + author
   └── If neither finds it → FLAG as potentially fabricated

3. VERIFY CLAIMS: (THE HARD PART — nobody does this well)
   → Check if the cited paper actually supports the claim made
   → This requires reading the cited paper's abstract and comparing
   → This is where our LLM agent adds unique value
```

> [!IMPORTANT]
> **Citation verification is our killer differentiator.** Turnitin doesn't do it. GPTZero doesn't do it. No mainstream tool does it well. With LLMs, we can read the abstract of a cited paper and check whether it actually supports the claim the author is making.

---

## System Architecture

### The Agent Society

```
                          ┌────────────────────┐
                          │                    │
                 ┌────────│  ORCHESTRATOR      │────────┐
                 │        │  AGENT             │        │
                 │        │                    │        │
                 │        │  • Receives paper  │        │
                 │        │  • Assigns tasks   │        │
                 │        │  • Collects results│        │
                 │        │  • Resolves conflicts       │
                 │        │  • Generates final │        │
                 │        │    report          │        │
                 │        └────────────────────┘        │
                 │                  │                    │
        ┌────────┴──┐    ┌────────┴──┐    ┌────────┴──┐
        ▼           ▼    ▼           ▼    ▼           ▼
┌──────────────┐┌──────────────┐┌──────────────┐┌──────────────┐
│ PLAGIARISM   ││ AI DETECTION ││ CITATION     ││ WRITING      │
│ AGENT        ││ AGENT        ││ VERIFICATION ││ QUALITY      │
│              ││              ││ AGENT        ││ AGENT        │
│ Checks text  ││ Analyzes     ││              ││              │
│ similarity   ││ perplexity & ││ Verifies     ││ Checks       │
│ against open ││ burstiness   ││ references   ││ grammar,     │
│ access papers││ patterns     ││ actually     ││ structure,   │
│ and web      ││              ││ exist &      ││ academic     │
│ sources      ││              ││ support      ││ tone         │
│              ││              ││ claims       ││              │
└──────┬───────┘└──────┬───────┘└──────┬───────┘└──────┬───────┘
       │               │               │               │
       └───────┬───────┴───────┬───────┘               │
               ▼               ▼                        │
       ┌──────────────────────────────┐                │
       │    CONFLICT RESOLVER         │◀───────────────┘
       │                              │
       │  When agents disagree:       │
       │  • Plag says "copied"        │
       │  • Citation says "properly   │
       │    quoted with citation"     │
       │  → Resolver decides: FALSE   │
       │    ALARM (properly cited)    │
       └──────────────────────────────┘
```

---

## Agent Specifications

### Agent 1: Plagiarism Check Agent

**Role:** Find text that matches existing published sources.

**How It Works (our free approach):**
```
Step 1: Break paper into paragraphs

Step 2: For each paragraph:
  a) Extract 2-3 key phrases
  b) Search using Serper API (web search, 2,500 free/month)
     → Query: exact phrase (in quotes)
     → Get: matching web pages
  c) Check CrossRef (unlimited, primary) for scholarly matches
     → Query: key phrases
     → Get: matching paper titles, DOIs
  d) ONLY if CrossRef finds a match:
     → Hit Semantic Scholar for the abstract (to compare content)
  e) If match found:
     → Use LLM to compare: "Is paragraph X saying the same thing
        as source Y? Rate similarity 0-100%"
  f) Cache ALL API responses (file-based JSON cache)
     → Same DOI/query never hits the API twice

Step 3: Compile matches with similarity scores and source links
```

**Free APIs Used:**
| API | What It Gives Us | Free Tier |
|:---|:---|:---|
| Semantic Scholar API | Paper search, abstracts, metadata | 1 req/sec (free key) |
| CrossRef API | DOI lookup, metadata verification | Unlimited (polite pool) |
| Serper API | Web text matching via Google results | 2,500 credits/month (free) |

**Limitations (be honest in the UI):**
- Cannot match against Turnitin's private student paper database
- Limited to open-access papers and web content
- Not a replacement for institutional Turnitin — a complementary pre-check

---

### Agent 2: AI Detection Agent

**Role:** Estimate whether sections were likely written by an AI model.

**How It Works (per-section batching):**
```
Step 1: Identify paper sections
  → Abstract, Introduction, Methodology, Results, Discussion, Conclusion
  → Each section is analyzed in ONE LLM call (5-7 total calls per paper)

Step 2: For each section, run TWO parallel analyses:

  a) LLM Classifier (primary signal):
     → Send entire section to Gemini 3.1 Flash Lite
     → Prompt: "Analyze this academic text section for patterns
        consistent with AI-generated writing. Consider: structural
        uniformity, vocabulary diversity, hedging language, and
        stylistic variation. Score AI probability 0-100% for each
        paragraph. Return structured JSON."
     → One LLM call per section = 5-7 calls total (not 50+)

  b) Burstiness Math (secondary signal — pure math, no API):
     → Compute sentence length variance per section
     → Compute vocabulary diversity (unique words / total words)
     → Low variance + low diversity → supports AI hypothesis
     → High variance + high diversity → supports human hypothesis

Step 3: Combine signals per section:
  → LLM confidence × 0.7 + burstiness signal × 0.3 = final score
  → Classify: "Likely Human" / "Likely AI" / "Uncertain"

Step 4: Overall document score:
  → Weighted average of section-level scores
  → Methodology sections get lower weight (naturally low-variance)
```

**Key Design Decisions:**
- **No logprobs.** Gemini's logprobs support is inconsistent across model versions. Instead of risking a brittle dependency, we use the LLM itself as a classifier.
- **Per-section batching.** Avoids the "Lost in the Middle" problem — LLMs struggle to structure massive output JSONs for entire documents. Per-section keeps each call focused and reliable.
- **Burstiness is pure math.** Zero API calls. Runs instantly. Gives a second opinion independent of the LLM.
- **No custom training.** Costs $0. Transparent methodology.

**Limitations (be honest in UI):**
- Less accurate than GPTZero or Turnitin AI detection
- May produce false positives on technical/formulaic writing
- Methodology sections naturally score higher — the Writing Quality Agent counterbalances this
- Should be treated as an indicator with confidence scores, never a binary verdict

---

### Agent 3: Citation Verification Agent ⭐ (Our Killer Feature)

**Role:** Verify that every reference in the paper (a) actually exists and (b) actually supports the claim being made.

**How It Works:**
```
Step 1: EXTRACT references from the paper
  → Use LLM to parse the References/Bibliography section
  → Extract: author, title, year, journal, DOI (if present)

Step 2: VERIFY EXISTENCE for each reference:
  a) If DOI present:
     → CrossRef API: GET /works/{doi}
     → Check: Does it return valid metadata?
     → If 404 → FABRICATED REFERENCE 🚨

  b) If no DOI:
     → Semantic Scholar API: search by title
     → Fuzzy match title + author names
     → If no close match found → POTENTIALLY FABRICATED 🚨

Step 3: VERIFY CLAIMS (the unique part — with abstract fallback chain):
  → For each in-text citation like "[12]" or "(Smith, 2023)":
     a) Extract the FULL PARAGRAPH containing the citation
        (not just the sentence — academic writing builds
        context over multiple sentences)
     b) Attempt to get abstract via this fallback chain:
        i)   Semantic Scholar abstract (covers ~95% of papers)
        ii)  If abstract is too short/vague → check Semantic Scholar
             openAccessPdf field for full-text link
        iii) If not OA → check Unpaywall API (free, 100K req/day)
             → finds legal OA versions on preprint servers,
             institutional repos, publisher green OA
        iv)  If still nothing → mark as PARTIALLY_VERIFIED
     c) If abstract/content obtained, ask LLM:
        "Read this paragraph from the submitted paper.
        Focus on the sentence citing [Smith, 2023]. Does the
        abstract of [Smith, 2023] support, contradict, or
        remain unrelated to how it is being used in this
        context? Respond: SUPPORTS / CONTRADICTS / UNRELATED /
        CANNOT DETERMINE. Explain your reasoning."

  NOTE: CrossRef handles existence checks (Step 2).
  Semantic Scholar + Unpaywall handle content retrieval (Step 3).

Step 4: Generate 4-tier verification report:
  → ✅ VERIFIED — paper exists AND abstract/content supports the claim
  → ⚠️ PARTIALLY VERIFIED — paper exists, but abstract is
     insufficient to confirm the claim (paywalled, vague abstract)
  → ❓ EXISTENCE ONLY — DOI resolves, but no content accessible
     to verify the specific claim being made
  → ❌ NOT FOUND — DOI returns 404, title search finds nothing
     → potentially fabricated reference 🚨

  PATTERN SIGNAL: If >50% of a paper's citations are
  PARTIALLY_VERIFIED or EXISTENCE_ONLY, flag this pattern —
  heavy reliance on unverifiable sources can itself be a
  red flag for fabrication.
```

**Why This Is Special:**
- **Turnitin:** Does NOT verify citations
- **GPTZero:** Does NOT verify citations
- **iThenticate:** Does NOT verify citations
- **Us:** We verify both existence AND claim accuracy

This is our primary differentiator and the feature that makes this project genuinely useful.

---

### Agent 4: Writing Quality Agent

**Role:** Assess academic writing quality — grammar, structure, tone.

**How It Works:**
```
Step 1: Analyze document structure
  → Check: Does it have Abstract, Introduction, Methodology,
    Results, Discussion, Conclusion, References?
  → Flag missing sections

Step 2: Assess writing quality per section
  → Use LLM to evaluate:
    • Grammar and spelling errors
    • Academic tone (vs. casual/conversational)
    • Sentence clarity and conciseness
    • Proper use of hedging language ("suggests" vs "proves")
    • Consistency of tense usage

Step 3: Check formatting
  → Citation style consistency (APA, IEEE, MLA, etc.)
  → Figure/table referencing
  → Heading hierarchy

Step 4: Generate improvement suggestions
  → Specific, actionable feedback per paragraph
```

---

### Agent 5: Orchestrator Agent

**Role:** Coordinate all agents, resolve conflicts, generate final report.

**Conflict Resolution Scenarios:**

| Scenario | Agent A Says | Agent B Says | Resolution |
|:---|:---|:---|:---|
| Properly cited quote | Plagiarism: "80% match!" | Citation: "Reference exists & is cited" | → **Not plagiarism** (properly attributed) |
| AI-generated with real citations | AI: "Likely AI-written" | Citation: "All refs verified" | → **Flag as AI-written** (real citations don't mean human-written) |
| Fabricated reference with original text | Plagiarism: "No matches" | Citation: "Ref #7 doesn't exist!" | → **Fabricated citation** (original text but fake sources) |
| Technical writing falsely flagged | AI: "90% AI confidence" | Quality: "Highly technical, consistent with domain conventions" | → **Lower AI confidence** (technical writing mimics AI patterns) |

---

## Tech Stack

| Component | Choice | Why |
|:---|:---|:---|
| **Language** | Python | Best ecosystem for NLP, APIs, LLM tools |
| **Agent Framework** | CrewAI (agents-as-tools) | Top-class multi-agent orchestration; deterministic tools keep fact-lookups trustworthy (see Architecture Update v2) |
| **LLM** | Gemini 3.1 Flash Lite (free tier) | 15–30 RPM / 250K–1M TPM free. Cheapest Gemini model, optimized for high-volume tasks |
| **PDF Parsing** | pymupdf4llm | Layout-aware extraction (handles two-column academic papers correctly) |
| **Web Framework** | Streamlit | Fastest path to a working web app. Free. Deployed on Streamlit Community Cloud for v1. |
| **Reference APIs** | CrossRef (primary) + Semantic Scholar (abstracts) + Unpaywall (OA fallback) | CrossRef is unlimited. Semantic Scholar for abstracts. Unpaywall finds legal OA versions of paywalled papers. |
| **Web Search** | Serper API | 2,500 free credits/month |
| **Deployment** | Streamlit Community Cloud (v1) | Free, zero-config. AWS is backup for v2 if we need more power. |

---

## Project Structure

```
paperguard/
├── app.py                    # Streamlit web app entry point
├── requirements.txt          # Python dependencies
├── .env                      # API keys (Gemini, Semantic Scholar)
│
├── agents/
│   ├── __init__.py
│   ├── orchestrator.py       # Orchestrator Agent
│   ├── plagiarism.py         # Plagiarism Check Agent
│   ├── ai_detection.py       # AI Detection Agent
│   ├── citation_verify.py    # Citation Verification Agent
│   └── writing_quality.py    # Writing Quality Agent
│
├── core/
│   ├── __init__.py
│   ├── pdf_parser.py         # PDF text extraction
│   ├── reference_parser.py   # Extract references from paper text
│   └── text_chunker.py       # Split text into paragraphs/sentences
│
├── services/
│   ├── __init__.py
│   ├── semantic_scholar.py   # Semantic Scholar API wrapper
│   ├── crossref.py           # CrossRef API wrapper
│   ├── serper.py             # Serper web search wrapper
│   └── gemini.py             # Gemini API wrapper
│
├── models/
│   ├── __init__.py
│   ├── report.py             # Report data models
│   └── reference.py          # Reference data models
│
├── ui/
│   ├── components.py         # Streamlit UI components
│   ├── report_view.py        # Report rendering
│   └── agent_log.py          # Live agent activity display
│
└── tests/
    ├── test_plagiarism.py
    ├── test_citation.py
    ├── test_ai_detection.py
    └── sample_papers/        # Test papers for development
        ├── clean_paper.pdf
        ├── plagiarized_paper.pdf
        └── ai_generated_paper.pdf
```

---

## Development Phases (Core First → UI Last)

> [!IMPORTANT]
> **Build the engine before the dashboard.** All agents must work as standalone Python modules with CLI/test outputs before any Streamlit code is written.

### Phase 1: Core Infrastructure + API Layer + Caching
> **Goal:** Every API wrapper works. PDFs parse correctly. References extract cleanly. Cache stores results. All testable via CLI.

| Task | Person 1 | Person 2 |
|:---|:---|:---|
| **PDF → Text** | `pdf_parser.py` — pymupdf4llm (layout-aware, handles two-column papers) | |
| | `text_chunker.py` — split into sections → paragraphs → sentences | |
| | Fallback: AWS Textract integration for scanned PDFs | |
| **Reference Extraction** | `reference_parser.py` — LLM extracts structured refs | |
| **API Wrappers** | | `gemini.py` — Gemini 3.1 Flash Lite wrapper |
| | | `semantic_scholar.py` — paper search API |
| | | `crossref.py` — DOI verification API |
| | | `serper.py` — web search API |
| | | `unpaywall.py` — open-access PDF finder |
| **Caching** | | `cache.py` — file-based JSON cache (key = DOI/title hash, value = API response) |
| **Shared Models** | `report.py` + `reference.py` — data models | |
| **Tests** | Sample papers + test harness | API response mocking |

**Checkpoint:** `python -m core.pdf_parser sample.pdf` → clean text. `python -m services.crossref 10.1038/xxx` → metadata (cached on second run).

---

### Phase 2: All Agents Working Independently
> **Goal:** Each agent runs as a standalone Python script. Input: text. Output: structured JSON. No UI.

**Build order (by priority):**

| Priority | Agent | Person | Why This Order |
|:---:|:---|:---|:---|
| **1st** | **Citation Verification Agent** | Person 1 | Killer feature. Non-negotiable. |
| **2nd** | **Writing Quality Agent** | Person 2 | Easiest to build (single LLM prompt per section). Ship fast. |
| **3rd** | **AI Detection Agent** | Person 2 | LLM classifier + burstiness math. No logprobs dependency. |
| **4th** | **Plagiarism Agent** | Person 1 | Weakest output, highest API cost. Build last. |

**Citation Agent tasks:**
- DOI/metadata lookup via CrossRef (primary — unlimited)
- Abstract retrieval via Semantic Scholar (only when needed for claim verification)
- Unpaywall fallback: find legal OA versions when abstract is insufficient
- Full-paragraph claim verification (compare paragraph context vs abstract via LLM)
- 4-tier classification: VERIFIED / PARTIALLY_VERIFIED / EXISTENCE_ONLY / NOT_FOUND
- Pattern detection: flag papers with >50% unverifiable citations

**AI Detection Agent tasks (per-section batching):**
- LLM classifier: one call per academic section (5-7 total) — "Score AI probability 0-100%"
- Burstiness math: sentence length variance + vocabulary diversity (pure math, no API)
- Section-level classification with combined scoring

**Writing Quality Agent tasks:**
- Structure analysis (Abstract, Intro, Methods, Results, Discussion, Conclusion)
- Tone & grammar assessment
- Improvement suggestions

**Plagiarism Agent tasks:**
- Key phrase extraction from paragraphs
- Search via Serper + Semantic Scholar (with cache)
- LLM-based similarity scoring

**Checkpoint:** `python -m agents.citation_verify sample.pdf` → JSON output from terminal.

---

### Phase 3: Custom Orchestrator + Conflict Resolution
> **Goal:** One command runs all agents, resolves conflicts, produces unified report. Still CLI-only.

```python
# orchestrator.py — this is literally all we need
def analyze_paper(pdf_path: str) -> dict:
    text = pdf_parser.extract(pdf_path)
    references = reference_parser.extract(text)
    
    # Run agents (order doesn't matter — results are independent)
    citation_results = citation_agent.run(text, references)
    quality_results = quality_agent.run(text)
    ai_results = ai_detection_agent.run(text)
    plagiarism_results = plagiarism_agent.run(text)
    
    # Resolve conflicts between agents
    resolved = conflict_resolver.resolve(
        citation_results, quality_results,
        ai_results, plagiarism_results
    )
    
    # Generate unified report
    return report_generator.create(resolved)
```

**Conflict resolution logic:**
- Plagiarism flags + Citation confirms proper quote → Remove flag
- AI detection flags + Quality confirms technical writing → Lower confidence
- Citation finds fabricated ref + Plagiarism finds no match → Fabricated citation alert

**Checkpoint:** `python main.py paper.pdf` → full JSON report with conflicts resolved.

---

### Phase 4: Streamlit UI (Deployed on Streamlit Community Cloud)
> **Goal:** Wrap the working backend in a clean web interface. Deployed free on Streamlit Cloud.

| Task | Person 1 | Person 2 |
|:---|:---|:---|
| **Upload Flow** | File upload → PDF parser → per-section progress ("Abstract analyzed... Introduction analyzed...") | |
| **Report Dashboard** | Overall scores (citation health, AI %, plagiarism %, quality score) | |
| **Detailed Sections** | Citation verification table (each ref → ✅/❌/⚠️) | AI detection per-section breakdown |
| | Plagiarism flagged sections with sources | Writing quality suggestions |
| **Export** | Download full report as PDF | |
| **Disclaimers** | Clear limitations notice (not a Turnitin replacement) | |
| **Caching** | `@st.cache_data` on orchestrator call — prevents re-analysis on UI reloads | |

**Why UI comes last:** Backend already works from CLI. UI is just a wrapper.

**Streamlit protection:** Wrap the orchestrator in `@st.cache_data` so if Streamlit re-runs the script (it does this on every widget interaction), the cached report is returned instantly instead of re-running the 2-minute analysis.

---

### Phase 5: Polish & Deploy
> **Goal:** Production-ready, deployed, tested with real papers.

| Task | Both |
|:---|:---|
| Test with 10+ real academic papers across domains |
| Handle edge cases (no references, very short papers, scanned PDFs → Textract fallback) |
| Rate limiting, retries, and error handling for all APIs |
| Deploy to Streamlit Community Cloud (free) |
| AWS account setup (S3 for PDF storage, Textract for scanned PDFs) |
| Write README + user docs |
| Set up billing alerts on AWS ($5 threshold) |



---

## User Flow

```
┌─────────────────────────────────────────────────────┐
│                   PAPERGUARD                         │
│                                                      │
│  ┌──────────────────────────────────────┐            │
│  │  📄 Upload your paper (PDF)          │            │
│  │  [Browse files...]                   │            │
│  └──────────────────────────────────────┘            │
│                                                      │
│  [🚀 Analyze Paper]                                  │
│                                                      │
│  ═══════════════════════════════════════════          │
│                                                      │
│  ⏳ Processing... Citation Agent (4/34 refs done)    │
│                                                      │
│  ═══════════════════════════════════════════          │
│                                                      │
│  📋 FINAL REPORT                                     │
│  ┌──────────────────────────────────────┐            │
│  │ Citation Health:      85% ✅          │            │
│  │ AI Content Score:     12% ⚠️          │            │
│  │ Plagiarism Score:     8%  ✅          │            │
│  │ Writing Quality:      7.2/10 ✅       │            │
│  │                                       │            │
│  │ 🚨 Issues Found: 3                   │            │
│  │ ├── Ref #7: DOI not found (fabricated?)│           │
│  │ ├── Ref #12: Doesn't support the claim│           │
│  │ └── Para 8: High AI probability (78%) │            │
│  └──────────────────────────────────────┘            │
│                                                      │
│  ⚠️ Disclaimer: This is a pre-submission self-check. │
│  It does NOT replace your institution's Turnitin.    │
│                                                      │
│  [📥 Download Full Report as PDF]                    │
└─────────────────────────────────────────────────────┘
```

---

## AWS Services That Actually Help Us

| AWS Service | Free Tier | How We Use It | Priority |
|:---|:---|:---|:---|
| **S3** | 5 GB (12 months) | Store uploaded PDFs + generated reports | ✅ Use from start |
| **Textract** | 1,000 pages/month (3 months) | OCR fallback for scanned PDFs that PyMuPDF can't read | ⚠️ Add when needed |
| **Comprehend** | 5M characters/month (12 months) | Entity extraction + keyphrase extraction from paper text (helps Citation Agent find claims) | ⚠️ Optional enhancement |
| **Lambda** | 1M requests/month (always free) | Run agents as serverless functions (future scaling) | ❌ v2 only |
| **DynamoDB** | 25 GB + 200M requests/month (always free) | Cache layer for API responses (replace file-based cache) | ❌ v2 only |
| **EC2** | 750 hrs/month t3.micro (12 months) | Host Streamlit if Community Cloud is too slow | 🔄 Backup plan |

> [!TIP]
> **For v1, we only need S3.** Textract and Comprehend are nice-to-haves that use the $100-200 promotional credits. DynamoDB and Lambda are v2 scaling optimizations.

---

## Free API Budget

| API | Free Tier | What We Use It For | Sufficient? |
|:---|:---|:---|:---|
| **Gemini 3.1 Flash Lite** | 15–30 RPM, 250K–1M TPM | All LLM tasks (detection, analysis, classification) | ✅ With batching |
| **Semantic Scholar** | 1 req/sec (with free key) | Paper search, abstract retrieval | ✅ With caching |
| **CrossRef** | Unlimited (polite pool) | DOI verification, metadata | ✅ Yes |
| **Unpaywall** | 100,000 req/day (free, just needs email) | Find legal OA versions of paywalled papers | ✅ More than enough |
| **Serper** | 2,500 credits/month | Web text matching for plagiarism | ✅ ~40 papers/month |
| **AWS S3** | 5 GB (12 months) | PDF + report storage | ✅ Yes |
| **AWS Textract** | 1,000 pages/month (3 months) | Scanned PDF OCR | ✅ If needed |
| **AWS Comprehend** | 5M chars/month (12 months) | Entity/keyphrase extraction | ✅ If needed |

> [!TIP]
> **Total cost: $0.** Every component uses free tiers. Caching reduces API calls by 60-80% on repeated lookups.

---

## AWS Free Tier — What You Get

If you create a **new AWS account**, you get:

| Resource | What You Get Free | What We'd Use It For |
|:---|:---|:---|
| **Promotional Credits** | $100 immediately + up to $100 more (by completing console activities) | General compute and services |
| **EC2** | ~750 hrs/month of t2.micro or t3.micro (for 12 months) | Hosting the Streamlit app |
| **S3** | 5 GB storage | Storing uploaded PDFs and generated reports |
| **Lambda** | 1 million requests/month (always free) | Running agent tasks serverlessly (future optimization) |
| **API Gateway** | 1 million API calls/month (12 months) | Exposing our agents as API endpoints |
| **Credit Expiry** | 12 months from account creation | Plenty of time |

> [!WARNING]
> **Set billing alerts immediately** when creating an AWS account. Even with free tier, unexpected usage can generate charges. Set a $5 budget alert on day 1.

### Hosting Decision

| Option | Pros | Cons | Recommendation |
|:---|:---|:---|:---|
| **Streamlit Community Cloud** | Zero setup, free forever, auto-deploys from GitHub | Limited resources, can be slow, no background jobs | ✅ Start here for v1 |
| **AWS EC2 (free tier)** | Full control, faster, can run background tasks | Requires setup, credits expire in 12 months | 🔄 Migrate when app grows |

**Plan:** Start with Streamlit Community Cloud (easiest). Move to AWS EC2 later if the app needs more power.

---

## Verification Plan

### Automated Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Test citation verification against known papers
python -m pytest tests/test_citation.py -v

# Test AI detection with known AI vs human samples
python -m pytest tests/test_ai_detection.py -v
```

### Manual Verification
1. **Test with a clean, original paper** → expect low plagiarism, low AI, all refs valid
2. **Test with a paper containing copy-pasted paragraphs** → expect plagiarism flags
3. **Test with a ChatGPT-generated essay** → expect high AI score
4. **Test with fabricated references** → expect citation agent to catch them
5. **Test with properly cited quotes** → expect conflict resolution to clear them

---

## 🔴 Reality Check — Honest Risks & Fallbacks

This section exists because I'd rather you know the hard truths NOW than discover them in Week 4.

### Risk 1: Logprobs May Not Work on Flash Lite

| The Problem | Impact | Fallback |
|:---|:---|:---|
| Gemini API logprobs support is inconsistent across model versions. Flash Lite may return `400: Logprobs not supported for this model`. | Our AI Detection Agent's perplexity calculation breaks completely. | **Fallback A:** Use a different Gemini model just for logprobs (e.g., Gemini 2.0 Flash or Gemini Pro). Free tier allows mixing models. |
| | | **Fallback B:** Skip perplexity math entirely. Instead, use the LLM itself as a classifier: "Analyze this text. Rate the probability it was written by AI (0-100%). Explain your reasoning." Less scientific, but still useful and simpler to build. |
| | | **Fallback C:** Use burstiness-only analysis (sentence length variance, vocabulary diversity) — these are pure math, no logprobs needed. |

> [!WARNING]
> **Test this on Day 1.** Before writing any AI detection code, run a quick test: send a request with `response_logprobs=True` to Flash Lite. If it works, great. If not, switch to Fallback B immediately. Don't waste a week building something that can't work.

---

### Risk 2: Semantic Scholar Rate Limits (1 req/sec)

| The Problem | Impact | Fallback |
|:---|:---|:---|
| 1 request per second = 60 requests per minute. A paper with 30 references needs 30+ lookups just for citation verification, plus more for plagiarism checking. | A single paper analysis could take **2-5 minutes** just waiting for API rate limits. | **Mitigation:** Batch and parallelize smartly. Citation Agent and Plagiarism Agent share results — if Citation already looked up a paper, Plagiarism reuses the cached result. |
| | | **Mitigation:** Request a free API key increase from Semantic Scholar (they grant higher limits for academic/research projects). |
| | | **Fallback:** Use CrossRef as primary (unlimited), Semantic Scholar as secondary. |

---

### Risk 3: Plagiarism Detection Will Never Match Turnitin

| The Problem | Impact | Fallback |
|:---|:---|:---|
| Turnitin has a proprietary database of **1 billion+ student papers.** We will never have access to this. Our plagiarism check only covers open-access papers + web content. | Users may submit a paper that passes our check but fails Turnitin. If we oversell our capabilities, we lose trust. | **Mitigation: Be brutally honest in the UI.** Show a clear disclaimer: "This checks against open-access papers and web sources. It does NOT replace your institution's Turnitin check." |
| | | **Positioning:** Frame our tool as a **pre-submission self-check**, not a replacement. "Catch issues before your professor does." |

---

### Risk 4: AI Detection Accuracy

| The Problem | Impact | Fallback |
|:---|:---|:---|
| Even GPTZero (with millions in training data) has false positive rates of 5-10%. Our simpler approach will be worse. | Technical papers, formal writing, and non-native English speakers may get falsely flagged as AI-written. | **Mitigation:** Always show confidence scores, never binary yes/no. Use language like "This section shows patterns consistent with AI-generated text (67% confidence)" instead of "This is AI-written." |
| | | **Mitigation:** The Writing Quality Agent acts as a counterbalance — if it confirms the text is domain-appropriate, the Orchestrator lowers AI confidence. This is our conflict resolution in action. |

---

### Risk 5: Gemini Free Tier Limits Under Load

| The Problem | Impact | Fallback |
|:---|:---|:---|
| 15–30 RPM on free tier. A single paper analysis requires: ~1 call for reference extraction, ~15 calls for citation claim verification (one per reference with abstract), 5-7 calls for AI detection (per section), 5-7 calls for quality analysis, ~10 calls for plagiarism comparison. Total: ~40 calls per paper. | At 15 RPM, a paper takes ~3 minutes just in rate-limit waiting. Workable for personal use but feels slow. | **Mitigation (already built in):** Per-section batching cuts AI detection + quality from 50+ calls to 10-14 calls. |
| | | **Mitigation:** `@st.cache_data` ensures re-runs don't re-analyze. File-based cache prevents redundant API calls. |
| | | **Fallback:** Add Groq as secondary free LLM. Load-balance between Gemini and Groq. |

---

### Risk 6: PDF Two-Column Layout (Critical)

| The Problem | Impact | Fallback |
|:---|:---|:---|
| Most published academic papers (IEEE, ACM, Nature, Elsevier) use two-column layouts. Standard PDF text extraction reads left-to-right across columns, stitching text from both columns into one gibberish line. | **Entire pipeline breaks.** Broken text = broken n-grams for plagiarism, broken sentences for AI detection, and mangled citations. Every agent produces garbage output. | **Fix (already applied):** Use `pymupdf4llm` instead of base PyMuPDF. It extracts text in correct reading order and outputs clean Markdown. |
| | | **Fallback:** `pdfplumber` as a secondary parser (slower but highly accurate at column detection). |

---

### The Honest Timeline

| What the Plan Says | What Will Probably Happen | Buffer |
|:---|:---|:---|
| Phase 1: Core + APIs + Cache | Straightforward — API wrappers, pymupdf4llm, cache | Lowest risk |
| Phase 2: Agents | Citation is most complex (claim verification). AI detection is simpler now (no logprobs). | Citation is the one to watch |
| Phase 3: Orchestrator | Simple Python pipeline — no framework overhead | Low risk |
| Phase 4: Streamlit UI | Streamlit is genuinely fast for dashboards | Low risk |
| Phase 5: Polish + Deploy | Edge cases, real paper testing, Streamlit Cloud deploy | Standard |

> [!IMPORTANT]
> **The MVP that must ship no matter what:**
> 1. ✅ Citation Verification Agent (killer feature — non-negotiable)
> 2. ✅ Writing Quality Agent (trivially easy — just LLM prompts)
> 3. ✅ Custom Orchestrator with conflict resolution
> 4. ✅ Streamlit UI with report + PDF export
> 5. ✅ File-based caching for all API calls
> 6. ⚠️ AI Detection — LLM classifier + burstiness (no logprobs dependency)
> 7. ⚠️ Plagiarism — weakest output, build last, be honest about limitations
> 8. ⚠️ AWS Textract for scanned PDFs — add if time permits
