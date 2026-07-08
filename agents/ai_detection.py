"""
AI-Detection engine (model-only).

AI detection is performed entirely by PaperGuard's fine-tuned DistilBERT
detector (``vediumsameer/paperguard-ai-detector``), scored via the calibrated
logit margin, plus embedding-based stylometric "patchwork" detection. No LLM is
used for AI detection: Gemini is reserved for agent orchestration and the other
tasks (citation claim checks, quality prose review, plagiarism similarity,
reference parsing).

Per paragraph it produces a heatmap entry; per document it produces an overall
AI score, the list of flagged paragraphs, and a stylometry block that flags
paragraphs whose style deviates from the rest of the paper (possible mixed
authorship / pasted AI).

Known blind spot: slang/style-masked AI can still read as human to the model
(the embedding patchwork check partially mitigates this when such text is pasted
into otherwise-human writing).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agents.base import iter_paragraphs, split_body_and_references
from agents.detector_agent import DetectorAgent

_MIN_PARAGRAPH_CHARS = 40
# Classification bands (AI probability 0-100).
_LIKELY_AI = 65
_LIKELY_HUMAN = 35
# Stylometric patchwork detection tuning.
_STYLO_MIN_PARAGRAPHS = 4       # need enough paragraphs for a stable baseline
# Robust (median/MAD) modified z-score threshold. Mean/std self-inflates on small
# samples (one outlier caps its own z near (n-1)/sqrt(n)); MAD avoids that.
_STYLO_MODZ_FLAG = 3.5
_STYLO_MIN_DISTANCE = 0.05      # ignore trivial deviations in ultra-uniform docs


def _classify(score: Optional[float]) -> str:
    if score is None:
        return "Uncertain"
    if score >= _LIKELY_AI:
        return "Likely AI"
    if score <= _LIKELY_HUMAN:
        return "Likely Human"
    return "Uncertain"


def _heat_level(score: Optional[float]) -> str:
    """Coarse band for UI heatmap colouring."""
    if score is None:
        return "unknown"
    if score >= _LIKELY_AI:
        return "high"
    if score <= _LIKELY_HUMAN:
        return "low"
    return "medium"


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


def _detect_patchwork(paragraphs: List[tuple], detector: DetectorAgent) -> Dict[str, Any]:
    """
    Stylometric drift / "Frankenstein" detection.

    Embeds each paragraph and flags those whose stylometric fingerprint deviates
    strongly (robust median/MAD modified z-score of cosine distance to the
    document centroid) from the rest of the paper -- a signal that a passage was
    written by a different "hand" (e.g. AI text pasted into a human draft).
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


def run_ai_detection(
    text: str,
    detector: Optional[DetectorAgent] = None,
    max_paragraphs: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run model-based AI detection over ``text``.

    Returns a document verdict with a per-paragraph heatmap and a stylometry
    (patchwork) block. Degrades gracefully: if the detector model is unavailable,
    scores are ``None`` and the result flags that.
    """
    detector = detector if detector is not None else DetectorAgent()
    paragraphs = _select_paragraphs(text, max_paragraphs)

    heatmap: List[Dict[str, Any]] = []
    scored: List[float] = []
    for idx, (section, para) in enumerate(paragraphs, start=1):
        s = detector.score_text(para)
        ai = s.get("ai_probability")
        if ai is not None:
            scored.append(ai)
        heatmap.append({
            "paragraph_index": idx,
            "section": section,
            "text_preview": para[:200],
            "final_ai_score": ai,
            "detector_score": ai,
            "raw_softmax_ai": s.get("raw_softmax_ai"),
            "logit_margin": s.get("logit_margin"),
            "classification": _classify(ai),
            "heat_level": _heat_level(ai),
        })

    overall = round(sum(scored) / len(scored), 2) if scored else None
    detector_ok = detector.score_text("probe").get("available", False)

    return {
        "overall_ai_score": overall,
        "classification": _classify(overall),
        "paragraphs_analyzed": len(heatmap),
        "flagged_paragraphs": [
            h["paragraph_index"] for h in heatmap if (h["final_ai_score"] or 0) >= _LIKELY_AI
        ],
        "method": "calibrated_distilbert_logit_margin",
        "components": {
            "detector_available": detector_ok,
            "detector_model": detector.model_name,
        },
        "stylometry": (
            _detect_patchwork(paragraphs, detector) if detector_ok
            else {"available": False, "reason": "detector unavailable", "outliers": []}
        ),
        "heatmap": heatmap,
    }
