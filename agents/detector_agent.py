"""
Detector Agent  --  PaperGuard's AI-text detector.

Runs ``desklib/ai-text-detector-v1.01`` (a fine-tuned microsoft/deberta-v3-large,
leads the RAID benchmark) to score how likely a block of text is AI-generated.
This is the sole AI-detection signal (no LLM is used for detection).

v3.0 model swap (2026-07-14): replaced the previous in-house DistilBERT
(``vediumsameer/paperguard-ai-detector``, v2.0) with desklib v1.01 after both
were benchmarked head-to-head, plus a third candidate (mdrakibali/deberta-ai-
detector-v3), on the SAME frozen 240-sample benchmark (``benchmark_samples.json``
/ see ``benchmark_results.md`` and ``PROJECT_REPORT.md`` Section 1 for the full
numbers). Desklib was the clear winner on the exact axis v2.0 failed:

    | Metric               | v2.0 (old) | desklib v1.01 (now) |
    |----------------------|-----------:|---------------------:|
    | AUC                  |      0.911 |                 0.968 |
    | Disguised-AI recall  |         0% |                   75% |
    | Human FPR @ ~92 cutoff |      0.5% |                  ~0.5% |

Unlike v2.0, this model's classifier head outputs a single logit through a
sigmoid (``P(AI)`` directly) rather than a 2-way softmax, and it was NOT
observed to be saturated/overconfident on our benchmark -- so no logit-margin
calibration is needed. We do still raise the "Likely AI" decision threshold
above the model's own 0.5 default (see ``_LIKELY_AI`` below): at cutoff ~50 the
benchmark measured a 7.0% human false-positive rate, but at ~90-95 FPR drops to
~0.5% while AI recall stays at 85-87.5% -- matching v2.0's low-FPR deployment
posture instead of the model card's naive 0.5 threshold.

The remaining blind spot is smaller but not zero: 25% of disguised/style-masked
2025-model AI still evades this model (down from 100% with v2.0). Embedding-
based stylometric patchwork detection (in ``agents.ai_detection``) partially
mitigates this when such text is pasted into otherwise-human writing.

Architecture: the HF repo does NOT use a standard ``AutoModelForSequenceClassification``
head -- per its own model card, it wraps ``microsoft/deberta-v3-large`` with a
mean-pooling layer and a single-logit linear classifier (custom ``PreTrainedModel``
subclass, defined below as ``_DesklibAIDetectionModel``).

Heavy deps (torch/transformers) and the model itself are loaded lazily and
cached at module level. If they are unavailable, the agent degrades gracefully:
``enabled`` becomes False and every score is ``None``.

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

# Canonical model: desklib/ai-text-detector-v1.01 (see module docstring for the
# benchmark that motivated the swap from v2.0). Override with
# PAPERGUARD_DETECTOR_MODEL to point at a local dir or a different repo -- note
# a different repo may need a different model class than _DesklibAIDetectionModel
# below (e.g. a standard AutoModelForSequenceClassification repo).
DEFAULT_MODEL = os.getenv(
    "PAPERGUARD_DETECTOR_MODEL", "desklib/ai-text-detector-v1.01"
)
# The model card's own recommended max sequence length.
_MAX_LENGTH = 768
# Only score paragraphs with at least this many characters (skip headers/captions).
_MIN_PARAGRAPH_CHARS = 40
# Classification bands (AI probability 0-100), shared with the LLM side.
# Raised from the model's naive 0.5 (50%) default to the ~90-95 operating point
# our frozen benchmark measured as the FPR/recall sweet spot (~0.5% FPR at
# ~85-87.5% recall, vs. 7.0% FPR at the raw 50% cutoff) -- see module docstring.
_LIKELY_AI = float(os.getenv("PAPERGUARD_DETECTOR_AI_THRESHOLD", "90"))
_LIKELY_HUMAN = float(os.getenv("PAPERGUARD_DETECTOR_HUMAN_THRESHOLD", "35"))


def _build_desklib_model_class():
    """
    Build the custom desklib model class lazily (needs torch imported first).

    Per the model card (desklib/ai-text-detector-v1.01): a
    microsoft/deberta-v3-large backbone + mean-pooling over token embeddings +
    a single-logit linear classifier head, trained with BCEWithLogitsLoss (so
    ``sigmoid(logit)`` directly gives ``P(AI)`` -- no softmax, no 2-way label
    index to resolve). Not a standard ``AutoModelForSequenceClassification``
    head, so it can't be loaded with that class; this mirrors the model card's
    own reference implementation exactly.
    """
    import torch
    import torch.nn as nn
    from transformers import AutoConfig, AutoModel, PreTrainedModel

    class _DesklibAIDetectionModel(PreTrainedModel):
        config_class = AutoConfig

        def __init__(self, config):
            super().__init__(config)
            self.model = AutoModel.from_config(config)
            self.classifier = nn.Linear(config.hidden_size, 1)
            self.init_weights()

        @property
        def all_tied_weights_keys(self):
            # Compatibility shim: newer `transformers` versions probe this
            # property during from_pretrained() weight-tying resolution; this
            # custom architecture (unlike the base HF model classes) doesn't
            # define it. An empty mapping is correct here since this model has
            # no tied weights. Verified against the same issue hit benchmarking
            # this exact model in this session (see PROJECT_REPORT.md Section 1).
            return {}

        def forward(self, input_ids, attention_mask=None, **kwargs):
            outputs = self.model(input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs[0]
            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            summed = torch.sum(last_hidden_state * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled = summed / counts
            logits = self.classifier(pooled)
            return {"logits": logits}

    return _DesklibAIDetectionModel


# --------------------------------------------------------------------------- #
# Lazy, cached model loader
# --------------------------------------------------------------------------- #
class _ModelBundle:
    """Holds the tokenizer/model/device once loaded; None until first use."""

    tokenizer = None
    model = None
    device = None
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
        from transformers import AutoTokenizer  # noqa: WPS433

        DesklibAIDetectionModel = _build_desklib_model_class()

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = DesklibAIDetectionModel.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        _BUNDLE.tokenizer = tokenizer
        _BUNDLE.model = model
        _BUNDLE.device = device
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
          ``ai_probability``     -- 0-100 AI likelihood, sigmoid(logit) * 100 (or None)
          ``human_probability``  -- 100 - ai_probability (or None)
          ``raw_softmax_ai``     -- kept for backward-compat with callers/UI;
                                     identical to ``ai_probability`` for this
                                     model (single sigmoid output, no softmax)
          ``logit_margin``       -- the raw classifier logit (pre-sigmoid);
                                     kept under the old field name so downstream
                                     consumers (heatmap "raw" display) still work
          ``calibrated``         -- False (this model's sigmoid output was not
                                     found to be saturated on our benchmark, so
                                     no margin-based recalibration is applied;
                                     see ``_LIKELY_AI``'s raised decision
                                     threshold instead)
          ``available``          -- False when the model could not be loaded
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
            padding="max_length",
        ).to(bundle.device)

        with torch.no_grad():
            logit = bundle.model(**inputs)["logits"][0, 0]
            ai_prob_raw = torch.sigmoid(logit).item()

        ai_prob = round(ai_prob_raw * 100, 2)
        human_prob = round(100.0 - ai_prob, 2)
        return {
            "ai_probability": ai_prob,
            "human_probability": human_prob,
            "raw_softmax_ai": ai_prob,
            "logit_margin": round(float(logit.item()), 4),
            "calibrated": False,
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

        Uses the underlying DeBERTa encoder directly (``bundle.model.model`` --
        the desklib wrapper's ``self.model`` attribute is the plain transformer
        backbone, per its own architecture; see ``_build_desklib_model_class``),
        masked-mean-pooled over tokens -- the same pooling the classifier head
        itself uses, so this is genuinely the representation the model reasons
        over, not an approximation.
        """
        bundle = _load_bundle(self.model_name)
        if bundle.model is None or not (text and text.strip()):
            return None

        import torch

        inputs = bundle.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=_MAX_LENGTH,
        ).to(bundle.device)
        with torch.no_grad():
            out = bundle.model.model(
                inputs["input_ids"], attention_mask=inputs["attention_mask"],
            )
        hidden = out[0][0]                                 # (seq, hidden)
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
            "method": "desklib_deberta_v3_large_sigmoid",
            "calibration": {
                "enabled": False,
                "ai_threshold": _LIKELY_AI,
                "note": (
                    "This model's sigmoid(logit) output was not found to be "
                    "saturated on our frozen benchmark, so no margin-based "
                    "recalibration is applied. Instead the 'Likely AI' decision "
                    "threshold is raised from the model's naive 50% default to "
                    f"{_LIKELY_AI}%, matching the FPR/recall operating point "
                    "measured on the benchmark (~0.5% human FPR at ~85-87.5% "
                    "AI recall around this cutoff vs. 7.0% FPR at 50%)."
                ),
            },
            "disclaimer": (
                "Statistical classifier. Slang/style-masked AI can still evade it "
                "(a known, reduced blind spot -- 75% disguised-AI recall, up from "
                "0% with the previous model); embedding-based patchwork detection "
                "helps when such text is mixed into human writing."
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
