"""
CrewAI tools -- deterministic capabilities the agents call.

Design principle (agents-as-tools): every fact-lookup and computation is a
deterministic tool. The LLM agents decide *when* to call them and *how to
interpret* the results, but never perform the lookups themselves. This keeps
citations, similarity scores, and model outputs trustworthy and reproducible.

Because academic papers are long, we do NOT pass the paper text through the LLM
as a tool argument. Instead the orchestrator sets the current paper via
``set_paper_context`` and the (argument-free) tools read from that context. Each
tool also caches its structured ``AgentResult`` into the context so the
orchestrator can assemble the final report deterministically, regardless of what
the LLM writes in its synthesis.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from models.reference import Reference

# --------------------------------------------------------------------------- #
# Shared paper context
# --------------------------------------------------------------------------- #
_CONTEXT: Dict[str, Any] = {
    "text": None,
    "references": None,
    "results": {},   # agent_name -> AgentResult.model_dump()
}


def set_paper_context(text: str, references: Optional[List[Reference]] = None) -> None:
    """Register the paper to analyze and reset cached results."""
    _CONTEXT["text"] = text or ""
    _CONTEXT["references"] = references
    _CONTEXT["results"] = {}


def get_cached_results() -> Dict[str, Any]:
    """Return the structured results accumulated by tool calls."""
    return _CONTEXT["results"]


def _ensure_references() -> List[Reference]:
    if _CONTEXT["references"] is None:
        from agents.base import extract_references
        try:
            _CONTEXT["references"] = extract_references(_CONTEXT["text"] or "")
        except Exception:
            _CONTEXT["references"] = []
    return _CONTEXT["references"]


def _store(name: str, payload: Dict[str, Any]) -> None:
    _CONTEXT["results"][name] = payload


# --------------------------------------------------------------------------- #
# Core implementations (usable with or without CrewAI installed)
# --------------------------------------------------------------------------- #
def _run_ai_detection() -> Dict[str, Any]:
    from agents.safety_net import run_safety_net
    result = run_safety_net(_CONTEXT["text"] or "")
    _store("AIDetectionSafetyNet", result)
    # Return a compact summary for the LLM (omit the full heatmap to save tokens).
    return {
        "overall_ai_score": result.get("overall_ai_score"),
        "classification": result.get("classification"),
        "paragraphs_analyzed": result.get("paragraphs_analyzed"),
        "conflicts_detected": result.get("conflicts_detected"),
        "overrides_applied": result.get("overrides_applied"),
        "safety_net_active": result.get("safety_net_active"),
        "flagged_paragraphs": result.get("flagged_paragraphs"),
        "components": result.get("components"),
    }


def _run_citation() -> Dict[str, Any]:
    from agents.citation_agent import CitationAgent
    result = CitationAgent().run(_CONTEXT["text"] or "", references=_ensure_references())
    payload = result.model_dump()
    _store(result.agent_name, payload)
    meta = payload.get("metadata", {})
    return {
        "status": payload.get("status"),
        "reference_count": meta.get("reference_count"),
        "citation_health_score": meta.get("citation_health_score"),
        "tier_counts": meta.get("tier_counts"),
        "not_found_count": meta.get("not_found_count"),
        "findings": payload.get("findings", [])[:8],
    }


def _run_plagiarism() -> Dict[str, Any]:
    from agents.plagiarism_agent import PlagiarismAgent
    result = PlagiarismAgent().run(_CONTEXT["text"] or "")
    payload = result.model_dump()
    _store(result.agent_name, payload)
    meta = payload.get("metadata", {})
    return {
        "status": payload.get("status"),
        "plagiarism_score": meta.get("plagiarism_score"),
        "flagged_paragraph_count": meta.get("flagged_paragraph_count"),
        "findings": payload.get("findings", [])[:8],
    }


def _run_quality() -> Dict[str, Any]:
    from agents.quality_agent import QualityAgent
    result = QualityAgent().run(_CONTEXT["text"] or "")
    payload = result.model_dump()
    _store(result.agent_name, payload)
    meta = payload.get("metadata", {})
    return {
        "status": payload.get("status"),
        "overall_quality_score": meta.get("overall_quality_score"),
        "structure": meta.get("structure"),
        "findings": payload.get("findings", [])[:8],
    }


# Map of tool name -> implementation (used by the fallback engine path).
CORE_TOOLS = {
    "ai_detection": _run_ai_detection,
    "citation": _run_citation,
    "plagiarism": _run_plagiarism,
    "quality": _run_quality,
}


# --------------------------------------------------------------------------- #
# CrewAI tool wrappers (only defined if CrewAI is importable)
# --------------------------------------------------------------------------- #
def build_crewai_tools():
    """
    Return CrewAI tool objects wrapping the core implementations, or ``None`` if
    CrewAI is not installed. Kept lazy so the engine works without CrewAI.
    """
    try:
        from crewai.tools import tool
    except Exception:
        return None

    @tool("AI Detection Safety Net")
    def ai_detection_tool() -> str:
        """Run the AI-detection safety net (PyTorch detector + LLM linguistic
        analysis + conflict resolver) over the current paper. Returns a JSON
        summary with the overall AI score, classification, and how many
        paragraph-level conflicts were resolved."""
        return json.dumps(_run_ai_detection(), ensure_ascii=False)

    @tool("Citation Verifier")
    def citation_tool() -> str:
        """Verify every reference in the current paper: existence (CrossRef /
        Semantic Scholar) and whether the cited work supports the claim. Returns
        a JSON summary with a citation-health score and per-tier counts."""
        return json.dumps(_run_citation(), ensure_ascii=False)

    @tool("Plagiarism Scanner")
    def plagiarism_tool() -> str:
        """Scan the current paper for overlap with open web and open-access
        scholarly sources. Returns a JSON summary with a similarity score and
        the number of flagged paragraphs."""
        return json.dumps(_run_plagiarism(), ensure_ascii=False)

    @tool("Writing Quality Reviewer")
    def quality_tool() -> str:
        """Assess the current paper's academic writing quality: structure,
        readability, and prose. Returns a JSON summary with an overall quality
        score and structural completeness."""
        return json.dumps(_run_quality(), ensure_ascii=False)

    return {
        "ai_detection": ai_detection_tool,
        "citation": citation_tool,
        "plagiarism": plagiarism_tool,
        "quality": quality_tool,
    }
