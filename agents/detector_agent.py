"""
Detector Agent  --  "The Math" half of the AI-detection Safety Net.

Runs PaperGuard's fine-tuned DistilBERT classifier
(``vediumsameer/paperguard-ai-detector``, v2.0 mega weights) to score how likely
a block of text is AI-generated from statistical token patterns alone.

Important: the model's softmax is *saturated* (overconfident) -- it reports ~0%
AI even on genuine AI text, so the softmax alone is unusable. The real signal
lives in the logit MARGIN (human_logit - ai_logit), which cleanly separates
clean/academic AI (~6-8) from human text (~16-18). This agent therefore scores
off a logistic calibration of the margin, not the softmax (see the Calibration
section below). After calibration the detector correctly flags clean and
academic AI (~70-90%) while keeping human text low (~10%).

The one remaining blind spot is slang / style-masked AI (an LLM told to write
casually), which can still read as human. That case is caught downstream by the
Linguistic Agent + Conflict Resolver -- the whole point of the "safety net".

Label convention (from the model config): index 0 == "ai", index 1 == "human".

Heavy deps (torch/transformers) and the model itself are loaded lazily and
cached at module level. If they are unavailable, the agent degrades gracefully:
``enabled`` becomes False and every score is ``None`` so the orchestrator can
fall back to the Linguistic Agent alone.

CLI:  python -m agents.detector_agent path/to/paper.(pdf|md|txt)
"""

from __future__ import annotations

import math
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
# Calibration
# --------------------------------------------------------------------------- #
# The v2.0 model is *overconfident*: its softmax saturates (e.g. 0% AI even on
# genuine AI text) because the logits are large. The raw logit MARGIN
# (human_logit - ai_logit), however, still separates classes well:
#   clean / academic AI  ~=  6-8      (AI-leaning)
#   human text           ~= 16-18     (human-leaning)
# So we recover the signal by scoring off the margin with a logistic mapping
# instead of trusting the saturated softmax:
#   ai_prob = 100 * sigmoid((MIDPOINT - margin) / SCALE)
# Defaults were chosen from the observed logit distribution; for a production /
# research-grade calibration these should be fit on a labelled dev set
# (Platt/temperature scaling). Override via env vars if you re-fit them.
_CALIB_MIDPOINT = float(os.getenv("PAPERGUARD_DETECTOR_CALIB_MIDPOINT", "12.0"))
_CALIB_SCALE = float(os.getenv("PAPERGUARD_DETECTOR_CALIB_SCALE", "2.5"))
# Set PAPERGUARD_DETECTOR_RAW_SOFTMAX=1 to bypass calibration (debug only).
_USE_CALIBRATION = os.getenv("PAPERGUARD_DETECTOR_RAW_SOFTMAX", "0") != "1"


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


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

        Returns a dict with:
          ``ai_probability``     -- calibrated 0-100 AI likelihood (or None)
          ``human_probability``  -- 100 - ai_probability (or None)
          ``raw_softmax_ai``     -- the model's uncalibrated softmax (saturates)
          ``logit_margin``       -- human_logit - ai_logit (the real signal)
          ``calibrated``         -- whether calibration was applied
          ``available``          -- False when the model could not be loaded

        The primary ``ai_probability`` is derived from the logit margin via a
        logistic calibration (see module docstring), because the raw softmax is
        saturated and unusable on its own.
        """
        bundle = _load_bundle(self.model_name)
        if bundle.model is None or not (text and text.strip()):
            return {
                "ai_probability": None, "human_probability": None,
                "raw_softmax_ai": None, "logit_margin": None,
                "calibrated": False, "available": False,
            }

        import torch

        inputs = bundle.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=_MAX_LENGTH,
        ).to(bundle.device)

        with torch.no_grad():
            logits = bundle.model(**inputs).logits[0]
            probs = torch.nn.functional.softmax(logits, dim=-1)

        ai_idx = bundle.ai_index
        is_binary = probs.shape[0] == 2
        human_idx = 1 - ai_idx if is_binary else ai_idx

        raw_softmax_ai = round(float(probs[ai_idx].item()) * 100, 4)
        margin = None
        if is_binary:
            margin = round(float(logits[human_idx].item() - logits[ai_idx].item()), 4)

        if _USE_CALIBRATION and margin is not None:
            ai_prob = round(_sigmoid((_CALIB_MIDPOINT - margin) / _CALIB_SCALE) * 100, 2)
            calibrated = True
        else:
            ai_prob = round(raw_softmax_ai, 2)
            calibrated = False

        human_prob = round(100.0 - ai_prob, 2)
        return {
            "ai_probability": ai_prob,
            "human_probability": human_prob,
            "raw_softmax_ai": raw_softmax_ai,
            "logit_margin": margin,
            "calibrated": calibrated,
            "available": True,
        }

    def score_paragraphs(self, paragraphs: List[str]) -> List[Dict[str, Any]]:
        """Score a list of paragraph strings; returns one dict per paragraph."""
        return [self.score_text(p) for p in paragraphs]

    # ------------------------------------------------------------------ #
    # Embeddings (for stylometric drift / "Frankenstein" patchwork detection)
    # ------------------------------------------------------------------ #
    def embed_text(self, text: str) -> Optional[List[float]]:
        """
        Return a mean-pooled last-hidden-state embedding for ``text`` (a
        stylometric fingerprint), or ``None`` if the model is unavailable.

        Uses the transformer encoder's final hidden states (architecture-
        agnostic via ``output_hidden_states``), masked-mean-pooled over tokens.
        """
        bundle = _load_bundle(self.model_name)
        if bundle.model is None or not (text and text.strip()):
            return None

        import torch

        inputs = bundle.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=_MAX_LENGTH,
        ).to(bundle.device)
        with torch.no_grad():
            out = bundle.model(**inputs, output_hidden_states=True)
        hidden = out.hidden_states[-1][0]                 # (seq, hidden)
        mask = inputs["attention_mask"][0].unsqueeze(-1)  # (seq, 1)
        summed = (hidden * mask).sum(dim=0)
        counts = mask.sum(dim=0).clamp(min=1)
        pooled = (summed / counts)
        return pooled.detach().cpu().tolist()

    def embed_paragraphs(self, paragraphs: List[str]) -> List[Optional[List[float]]]:
        """Embed a list of paragraph strings (one vector per paragraph)."""
        return [self.embed_text(p) for p in paragraphs]

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
                "raw_softmax_ai": scored.get("raw_softmax_ai"),
                "logit_margin": scored.get("logit_margin"),
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
            "method": (
                "distilbert_logit_margin_calibrated" if _USE_CALIBRATION
                else "distilbert_raw_softmax"
            ),
            "calibration": {
                "enabled": _USE_CALIBRATION,
                "midpoint": _CALIB_MIDPOINT,
                "scale": _CALIB_SCALE,
                "note": (
                    "Score derived from the logit margin (human-ai) via logistic "
                    "calibration; the raw softmax is saturated and reported only "
                    "for transparency in each paragraph's 'raw_softmax_ai'."
                ),
            },
            "disclaimer": (
                "Statistical classifier. Slang/style-masked AI can still evade it "
                "(a known blind spot); paired with the Linguistic Agent and "
                "Conflict Resolver to correct such cases."
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
