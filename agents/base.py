"""
Shared conventions and helpers for PaperGuard agents.

Design goals:
- Agents accept plain text (already extracted from a PDF) plus optional
  pre-parsed references, and return a `models.report.AgentResult`.
- Heavy / optional dependencies (pymupdf4llm for PDFs, the Gemini SDK) are
  imported lazily so that an agent can still run in a "degraded" mode for
  local testing and CLI checkpoints when those libs or API keys are missing.
- A single `run_cli` harness gives every agent an identical command-line
  interface and supports `.pdf`, `.md`, and `.txt` inputs (the text formats
  make it trivial to test agents without a real PDF or PDF libraries).
"""

from __future__ import annotations

import os
import re
import sys
import json
import argparse
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

# Ensure the project root is importable when an agent is executed as a script
# (e.g. `python -m agents.citation_agent`) as well as when imported normally.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.report import AgentResult  # noqa: E402
from models.reference import Reference  # noqa: E402


# --------------------------------------------------------------------------- #
# Optional dependency accessors (lazy, never raise at import time)
# --------------------------------------------------------------------------- #
def get_llm():
    """
    Lazily return the `services.gemini` module, or ``None`` if the Gemini SDK
    is not installed. Note that even when this returns a module, individual
    calls return ``None`` when ``GEMINI_API_KEY`` is not configured, so callers
    must always handle a ``None`` response.
    """
    try:
        from services import gemini  # noqa: WPS433 (intentional lazy import)
        return gemini
    except Exception:  # pragma: no cover - environment dependent
        return None


def llm_available() -> bool:
    """True only when the Gemini SDK is importable AND an API key is present."""
    if get_llm() is None:
        return False
    return bool(os.getenv("GEMINI_API_KEY"))


def load_env() -> None:
    """Best-effort load of a local .env file (no-op if python-dotenv missing)."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Input loading
# --------------------------------------------------------------------------- #
def load_text(path: str | Path) -> str:
    """
    Load paper text from a file.

    - ``.pdf``  -> extracted via ``core.pdf_parser`` (requires pymupdf4llm).
    - ``.md`` / ``.txt`` / other -> read directly as UTF-8 text.

    The text formats let us exercise every agent from the CLI without a real
    PDF or the PDF parsing dependency installed.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path_obj.suffix.lower() == ".pdf":
        # Lazy import so that non-PDF inputs work without pymupdf4llm installed.
        from core.pdf_parser import PDFParser

        return PDFParser().parse(path_obj)

    return path_obj.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Text structuring helpers shared by multiple agents
# --------------------------------------------------------------------------- #
_REFERENCES_HEADER = re.compile(
    r"^\s*(?:#{1,6}\s+)?(?:\*\*?)?(?:\d+\.?\s*)?"
    r"(References|Bibliography|Works Cited|Literature Cited)"
    r"(?:\*\*?)?\s*:?\s*$",
    re.IGNORECASE,
)


def split_body_and_references(text: str) -> Tuple[str, str]:
    """
    Split a paper into (body_text, references_text).

    Finds the last line that looks like a References/Bibliography header and
    treats everything after it as the reference list. Returns the full text as
    the body and an empty references string when no such header is found.
    """
    lines = text.splitlines()
    header_idx: Optional[int] = None
    for idx, line in enumerate(lines):
        if _REFERENCES_HEADER.match(line):
            header_idx = idx  # keep the last match (avoids in-text mentions)

    if header_idx is None:
        return text, ""

    body = "\n".join(lines[:header_idx]).strip()
    references = "\n".join(lines[header_idx + 1:]).strip()
    return body, references


def chunk_text(text: str) -> Dict[str, List[List[str]]]:
    """Chunk text into {section: [[sentence, ...paragraph], ...]} via TextChunker."""
    from core.text_chunker import TextChunker

    return TextChunker().chunk(text)


def extract_references(text: str) -> List[Reference]:
    """
    Extract structured references from a paper's full text.

    Isolates the References section first, then delegates to
    ``core.reference_parser.ReferenceParser`` (LLM when available, heuristic
    fallback otherwise).
    """
    _body, references_text = split_body_and_references(text)
    if not references_text:
        return []

    from core.reference_parser import ReferenceParser

    return ReferenceParser().parse_references(references_text)


def iter_paragraphs(text: str) -> List[Tuple[str, str]]:
    """
    Return a flat list of ``(section_name, paragraph_text)`` tuples.

    Paragraph text is reconstructed by joining the chunker's sentence lists so
    downstream agents get readable paragraph strings.
    """
    structured = chunk_text(text)
    paragraphs: List[Tuple[str, str]] = []
    for section, paras in structured.items():
        for sentences in paras:
            para_text = " ".join(sentences).strip()
            if para_text:
                paragraphs.append((section, para_text))
    return paragraphs


# --------------------------------------------------------------------------- #
# Base agent
# --------------------------------------------------------------------------- #
class BaseAgent(ABC):
    """Common interface for all verification agents."""

    #: Human-readable agent name, surfaced in the report.
    name: str = "BaseAgent"

    #: Whether ``run`` benefits from pre-parsed references being passed in.
    needs_references: bool = False

    @abstractmethod
    def run(
        self,
        text: str,
        references: Optional[List[Reference]] = None,
    ) -> AgentResult:
        """Analyze ``text`` and return a structured :class:`AgentResult`."""
        raise NotImplementedError

    # -- helpers for subclasses --------------------------------------------- #
    def _result(
        self,
        status: str,
        findings: List[str],
        metadata: Dict[str, Any],
    ) -> AgentResult:
        return AgentResult(
            agent_name=self.name,
            status=status,
            findings=findings,
            metadata=metadata,
        )


# --------------------------------------------------------------------------- #
# CLI harness
# --------------------------------------------------------------------------- #
def run_cli(agent: BaseAgent, description: str) -> None:
    """
    Standard command-line entry point for an agent.

    Usage: ``python -m agents.<module> <paper.pdf|.md|.txt> [-o out.json]``
    Prints the agent's result as JSON to stdout.
    """
    load_env()

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("input", help="Path to the paper (.pdf, .md, or .txt)")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional path to write the JSON result to.",
    )
    args = parser.parse_args()

    try:
        text = load_text(args.input)
    except Exception as exc:  # noqa: BLE001
        print(f"Error loading input: {exc}", file=sys.stderr)
        sys.exit(1)

    references: Optional[List[Reference]] = None
    if agent.needs_references:
        try:
            references = extract_references(text)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: reference extraction failed: {exc}", file=sys.stderr)
            references = []

    result = agent.run(text, references=references)
    payload = json.dumps(result.model_dump(), indent=2, ensure_ascii=False)

    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
        print(f"Result written to {args.output}")
    else:
        print(payload)
