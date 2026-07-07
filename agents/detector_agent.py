"""
Detector Agent  --  "The Math" half of the AI-detection Safety Net.

Runs PaperGuard's fine-tuned DistilBERT classifier
(``vediumsameer/paperguard-ai-detector``, v2.0 mega weights) to score how likely
a block of text is AI-generated. It looks *only* at statistical token patterns.

This agent is deliberately dumb-but-fast. It has documented blind spots
(mode collapse / logit saturation): it can panic to ~100% AI on rigid ESL
writing, or ~100% Human on style-masked AI text. Those mistakes are caught
downstream by the Linguistic Agent + Conflict Resolver -- that is the whole
point of the "safety net" architecture.

Label convention (from the model config): index 0 == "ai", index 1 == "human".

Heavy deps (torch/transformers) and the model itself are loaded lazily and
cached at module level. If they are unavailable, the agent degrades gracefully:
``enabled`` becomes False and every score is ``None`` so the orchestrator can
fall back to the Linguistic Agent alone.

CLI:  python -m agents.detector_agent path/to/paper.(pdf|md|txt)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from agents.base import (
    BaseAgent,
    Reference,
    iter_paragraphs,
    run_cli,
    split_body_and_references,
)

# Canonical model (the same HF repo now holds the v2.0 mega weights). Override
# with PAPERGUARD_DETECTOR_MODEL to point at a local dir or a different repo.
DEFAULT_MODEL = os.getenv(
    "PAPERGUARD_DETECTOR_MODEL", "vediumsameer/paperguard-ai-detector"
)
_MAX_LENGTH = 512
# Only score paragraphs with at least this many characters (skip headers/captions).
_MIN_PARAGRAPH_CHARS = 40
# Classification bands (AI probability 0-100), shared with the LLM side.
_LIKELY_AI = 65
_LIKELY_HUMAN = 35


# --------------------------------------------------------------------------- #
# Lazy, cached model loader
# --------------------------------------------------------------------------- #
class _ModelBundle:
    """Holds the tokenizer/model/device once loaded; None until first use."""

    tokenizer = None
    model = None
    device = None
    ai_index = 0
    load_error: Optional[str] = None
    loaded = False


_BUNDLE = _ModelBundle()


def _load_bundle(model_name: str = DEFAULT_MODEL) -> _ModelBundle:
    """Load (once) the tokenizer + model. Records any failure in load_error."""
    if _BUNDLE.loaded:
        return _BUNDLE
    _BUNDLE.loaded = True  # mark attempted so we never retry a hard failure in a loop
    try:
        import torch  # noqa: WPS433 (intentional lazy import)
        from transformers import (  # noqa: WPS433
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        # Resolve the "ai" logit index robustly from the model config.
        ai_index = 0
        label2id = getattr(model.config, "label2id", None) or {}
        for label, idx in label2id.items():
            if str(label).strip().lower() == "ai":
                ai_index = int(idx)
                break

        _BUNDLE.tokenizer = tokenizer
        _BUNDLE.model = model
        _BUNDLE.device = device
        _BUNDLE.ai_index = ai_index
    except Exception as exc:  # noqa: BLE001 - any failure => degraded mode
        _BUNDLE.load_error = f"{type(exc).__name__}: {exc}"
    return _BUNDLE


def detector_available() -> bool:
    """True if the PyTorch model is loadable/loaded in the current environment."""
    return _load_bundle().model is not None


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
class DetectorAgent(BaseAgent):
    """PyTorch statistical AI-text classifier (per-paragraph)."""

    name = "DetectorAgent"
    needs_references = False

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name

    # ------------------------------------------------------------------ #
    # Low-level scoring (used directly by the Orchestrator's safety net)
    # ------------------------------------------------------------------ #
    def score_text(self, text: str) -> Dict[str, Any]:
        """
        Score a single block of text.

        Returns ``{"ai_probability": float|None, "human_probability": float|None,
        "available": bool}``. Probabilities are 0-100; ``None`` when the model
        could not be loaded.
        """
        bundle = _load_bundle(self.model_name)
        if bundle.model is None or not (text and text.strip()):
            return {"ai_probability": None, "human_probability": None, "available": False}

        import torch

        inputs = bundle.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=_MAX_LENGTH,
        ).to(bundle.device)

        with torch.no_grad():
            logits = bundle.model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]

        ai_idx = bundle.ai_index
        human_idx = 1 - ai_idx if probs.shape[0] == 2 else ai_idx
        ai_prob = round(float(probs[ai_idx].item()) * 100, 2)
        human_prob = round(float(probs[human_idx].item()) * 100, 2)
        return {"ai_probability": ai_prob, "human_probability": human_prob, "available": True}

    def score_paragraphs(self, paragraphs: List[str]) -> List[Dict[str, Any]]:
        """Score a list of paragraph strings; returns one dict per paragraph."""
        return [self.score_text(p) for p in paragraphs]

    # ------------------------------------------------------------------ #
    # Standalone agent entry point
    # ------------------------------------------------------------------ #
    def run(self, text: str, references: Optional[List[Reference]] = None) -> "Any":
        body_text, _refs = split_body_and_references(text)
        paragraphs = [
            (section, para)
            for section, para in (iter_paragraphs(body_text or text) if text else [])
            if section.lower() not in {"references", "bibliography"}
            and len(para.strip()) >= _MIN_PARAGRAPH_CHARS
        ]

        bundle = _load_bundle(self.model_name)
        available = bundle.model is not None

        if not available:
            return self._result(
                status="warning",
                findings=[
                    "PyTorch detector unavailable "
                    f"({bundle.load_error or 'torch/transformers not installed'}); "
                    "AI detection will rely on the Linguistic Agent alone."
                ],
                metadata={
                    "enabled": False,
                    "model_name": self.model_name,
                    "load_error": bundle.load_error,
                    "overall_ai_score": None,
                    "paragraphs": [],
                },
            )

        if not paragraphs:
            return self._result(
                status="warning",
                findings=["No analyzable paragraphs were found."],
                metadata={"enabled": True, "model_name": self.model_name,
                          "overall_ai_score": None, "paragraphs": []},
            )

        para_scores: List[Dict[str, Any]] = []
        ai_values: List[float] = []
        for i, (section, para) in enumerate(paragraphs, start=1):
            scored = self.score_text(para)
            ai_prob = scored["ai_probability"]
            if ai_prob is not None:
                ai_values.append(ai_prob)
            para_scores.append({
                "paragraph_index": i,
                "section": section,
                "ai_probability": ai_prob,
                "human_probability": scored["human_probability"],
                "classification": self._classify(ai_prob),
                "text_preview": para[:160],
            })

        overall = round(sum(ai_values) / len(ai_values), 2) if ai_values else None
        classification = self._classify(overall)
        findings = self._compose_findings(para_scores, overall, classification)
        status = "warning" if (overall is not None and overall >= _LIKELY_AI) else "passed"

        metadata = {
            "enabled": True,
            "model_name": self.model_name,
            "device": bundle.device,
            "overall_ai_score": overall,
            "classification": classification,
            "method": "pytorch_distilbert_classifier",
            "disclaimer": (
                "Raw statistical classifier. Prone to mode collapse on out-of-"
                "distribution text (ESL, style-masked AI); paired with the "
                "Linguistic Agent and Conflict Resolver to correct this."
            ),
            "paragraphs": para_scores,
        }
        return self._result(status=status, findings=findings, metadata=metadata)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _classify(score: Optional[float]) -> str:
        if score is None:
            return "Uncertain"
        if score >= _LIKELY_AI:
            return "Likely AI"
        if score <= _LIKELY_HUMAN:
            return "Likely Human"
        return "Uncertain"

    @staticmethod
    def _compose_findings(
        para_scores: List[Dict[str, Any]],
        overall: Optional[float],
        classification: str,
    ) -> List[str]:
        findings: List[str] = []
        if overall is not None:
            findings.append(f"Detector overall AI score: {overall}% ({classification}).")
        flagged = [p for p in para_scores if (p["ai_probability"] or 0) >= _LIKELY_AI]
        for p in flagged[:10]:
            findings.append(
                f"Paragraph {p['paragraph_index']} ('{p['section']}'): "
                f"{p['ai_probability']}% AI (raw model)."
            )
        if not flagged:
            findings.append("No paragraph flagged as AI by the raw model.")
        return findings


if __name__ == "__main__":
    run_cli(DetectorAgent(), "PyTorch statistical AI-text detector (the Math).")
