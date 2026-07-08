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
# Stylometric patchwork detection tuning.
_STYLO_MIN_PARAGRAPHS = 4       # need enough paragraphs for a stable baseline
# Robust (median/MAD) modified z-score threshold. Mean/std self-inflates on small
# samples (one outlier caps its own z near (n-1)/sqrt(n)); MAD avoids that.
_STYLO_MODZ_FLAG = 3.5
_STYLO_MIN_DISTANCE = 0.05      # ignore trivial deviations in ultra-uniform docs


def _detect_patchwork(
    paragraphs: List[tuple], detector: DetectorAgent
) -> Dict[str, Any]:
    """
    Stylometric drift / "Frankenstein" detection.

    Embeds each paragraph and flags those whose stylometric fingerprint deviates
    strongly (z-score of cosine distance to the document centroid) from the rest
    of the paper -- a signal that a passage was written by a different "hand"
    (e.g. AI text pasted into a human draft), which a per-paragraph AI score can
    miss when the passage looks locally plausible.
    """
    if len(paragraphs) < _STYLO_MIN_PARAGRAPHS:
        return {"available": False, "reason": "too few paragraphs", "outliers": []}
    try:
        import numpy as np
    except Exception:
        return {"available": False, "reason": "numpy unavailable", "outliers": []}

    vecs, meta = [], []
    for idx, (section, para) in enumerate(paragraphs, start=1):
        emb = detector.embed_text(para)
        if emb is not None:
            vecs.append(emb)
            meta.append((idx, section, para))
    if len(vecs) < _STYLO_MIN_PARAGRAPHS:
        return {"available": False, "reason": "embeddings unavailable", "outliers": []}

    mat = np.array(vecs, dtype=float)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = mat / norms
    centroid = unit.mean(axis=0)
    cnorm = np.linalg.norm(centroid) or 1.0
    centroid = centroid / cnorm
    distances = 1.0 - unit.dot(centroid)  # cosine distance to the document style

    mean_d = float(distances.mean())
    # Robust outlier detection via median absolute deviation (MAD).
    median_d = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median_d)))
    scale = mad if mad > 1e-6 else (float(distances.std()) or 1e-9)

    outliers = []
    for (idx, section, para), dist in zip(meta, distances):
        mod_z = 0.6745 * (float(dist) - median_d) / scale
        if mod_z >= _STYLO_MODZ_FLAG and float(dist) >= _STYLO_MIN_DISTANCE:
            outliers.append({
                "paragraph_index": idx,
                "section": section,
                "style_distance": round(float(dist), 4),
                "modified_zscore": round(mod_z, 2),
                "text_preview": para[:200],
            })

    return {
        "available": True,
        "paragraphs_analyzed": len(vecs),
        "mean_style_distance": round(mean_d, 4),
        "style_cohesion": round(max(0.0, 1.0 - mean_d), 4),  # 1 = very uniform style
        "outlier_count": len(outliers),
        "outliers": outliers,
        "note": (
            "Paragraphs whose stylometric fingerprint deviates strongly "
            f"(robust modified z >= {_STYLO_MODZ_FLAG}) from the document may "
            "indicate mixed authorship (human + pasted AI). Indicative, not definitive."
        ),
    }


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
    detector_ok = detector.score_text("probe").get("available", False)
    result["components"] = {
        "detector_available": detector_ok,
        "linguistic_available": linguistic.available(),
        "detector_model": detector.model_name,
        "linguistic_backend": getattr(linguistic, "_backend", "unknown"),
    }
    result["stylometry"] = (
        _detect_patchwork(paragraphs, detector) if detector_ok
        else {"available": False, "reason": "detector unavailable", "outliers": []}
    )
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
