"""
AI-Detection Safety Net runner.

Framework-independent core that ties the three AI-detection components together
over a document, paragraph by paragraph:

    Detector (PyTorch math)  ─┐
                              ├─►  Conflict Resolver  ─►  heatmap + verdict
    Linguistic (LLM context) ─┘

This is the reusable engine the CrewAI crew (and the CLI/UI) call into. Keeping
it separate from CrewAI means AI detection still works if CrewAI is unavailable,
and makes it trivially unit-testable (e.g. against ``ood_stress_test`` cases).

Both agents degrade gracefully, so the runner works in any of these modes:
  * Detector + Linguistic  -> full safety net (mode-collapse correction).
  * Linguistic only        -> contextual detection (no model installed).
  * Detector only          -> raw model, flagged as "no safety net".
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agents.base import iter_paragraphs, split_body_and_references
from agents.conflict_resolver import ConflictResolver
from agents.detector_agent import DetectorAgent
from agents.linguistic_agent import LinguisticAgent

_MIN_PARAGRAPH_CHARS = 40


def _select_paragraphs(text: str, max_paragraphs: Optional[int]) -> List[tuple]:
    body, _refs = split_body_and_references(text)
    paras = [
        (section, para)
        for section, para in (iter_paragraphs(body or text) if text else [])
        if section.lower() not in {"references", "bibliography"}
        and len(para.strip()) >= _MIN_PARAGRAPH_CHARS
    ]
    if max_paragraphs is not None:
        paras = paras[:max_paragraphs]
    return paras


def run_safety_net(
    text: str,
    detector: Optional[DetectorAgent] = None,
    linguistic: Optional[LinguisticAgent] = None,
    resolver: Optional[ConflictResolver] = None,
    max_paragraphs: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run the full AI-detection safety net over ``text``.

    Returns the ``ConflictResolver.resolve_document`` payload augmented with a
    ``components`` block describing which signals were available.
    """
    detector = detector if detector is not None else DetectorAgent()
    linguistic = linguistic if linguistic is not None else LinguisticAgent()
    resolver = resolver if resolver is not None else ConflictResolver()

    paragraphs = _select_paragraphs(text, max_paragraphs)

    inputs: List[Dict[str, Any]] = []
    for idx, (section, para) in enumerate(paragraphs, start=1):
        det = detector.score_text(para)
        lin = linguistic.score_text(para)
        inputs.append({
            "detector": det,
            "linguistic": lin,
            "meta": {"paragraph_index": idx, "section": section, "text": para},
        })

    result = resolver.resolve_document(inputs)
    result["components"] = {
        "detector_available": detector.score_text("probe").get("available", False),
        "linguistic_available": linguistic.available(),
        "detector_model": detector.model_name,
        "linguistic_backend": getattr(linguistic, "_backend", "unknown"),
    }
    return result


def score_single_paragraph(
    text: str,
    detector: Optional[DetectorAgent] = None,
    linguistic: Optional[LinguisticAgent] = None,
    resolver: Optional[ConflictResolver] = None,
) -> Dict[str, Any]:
    """Run the safety net on a single block of text (used by the CrewAI tool)."""
    detector = detector if detector is not None else DetectorAgent()
    linguistic = linguistic if linguistic is not None else LinguisticAgent()
    resolver = resolver if resolver is not None else ConflictResolver()

    det = detector.score_text(text)
    lin = linguistic.score_text(text, force=True)
    return resolver.resolve_paragraph(det, lin, {"paragraph_index": 1, "text": text})
