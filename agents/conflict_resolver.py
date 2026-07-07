"""
Conflict Resolver -- the "safety net" that reconciles the AI-detection agents.

The raw PyTorch Detector (math) and the LLM Linguistic agent (context) each
score a paragraph's AI-probability independently. This module reconciles them:

  * Agreement                    -> keep the score, mark the paragraph.
  * Logit saturation (Scenario A): model says ~human but the LLM sees AI
    (style-masked / Frankenstein text) -> adopt the LLM verdict.
  * ESL false-positive (Scenario B): model says ~AI but the LLM sees a human
    (rigid non-native prose) -> adopt the LLM verdict.
  * Other strong disagreement (Scenario C) -> 40/60 weighted consensus that
    favours the contextual LLM.

It exposes three levels of API:
  * ``resolve``            -- one paragraph, raw dict in / dict out (original).
  * ``resolve_paragraph``  -- one paragraph -> a rich heatmap entry.
  * ``resolve_document``   -- many paragraphs -> heatmap + weighted verdict.

All methods are ``None``-safe: if only one signal is available the resolver
degrades gracefully (and flags the risky "detector-only, no safety net" case).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Classification bands (0-100 AI probability) -- shared with the agents.
_LIKELY_AI = 65
_LIKELY_HUMAN = 35


class ConflictResolver:
    """Reconciles the Detector (math) and Linguistic (context) AI scores."""

    def __init__(self, conflict_threshold: float = 30.0):
        # A divergence >= this (in AI-probability points) triggers a conflict.
        self.conflict_threshold = conflict_threshold

    # ------------------------------------------------------------------ #
    # Original single-paragraph API (kept backward-compatible, now None-safe)
    # ------------------------------------------------------------------ #
    def resolve(
        self,
        detector_payload: Dict[str, Any],
        linguistic_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Reconcile one paragraph's two scores.

        Each payload may carry ``ai_probability`` (0-100 or None) and, for the
        linguistic payload, ``reasoning``. Returns a verdict dict.
        """
        pytorch_ai = detector_payload.get("ai_probability") if detector_payload else None
        llm_ai = linguistic_payload.get("ai_probability") if linguistic_payload else None
        llm_reasoning = (
            (linguistic_payload or {}).get("reasoning") or "No context provided."
        )

        verdict: Dict[str, Any] = {
            "conflict_detected": False,
            "override_type": None,  # "logit_saturation" | "esl_false_positive" | "compromise" | None
            "original_pytorch_score": pytorch_ai,
            "original_llm_score": llm_ai,
            "final_consensus_score": None,
            "resolution_reasoning": "",
            "safety_net_active": pytorch_ai is not None and llm_ai is not None,
        }

        # -- Degraded modes: one or both signals missing ------------------ #
        if pytorch_ai is None and llm_ai is None:
            verdict["resolution_reasoning"] = "No AI-detection signal available."
            return verdict
        if pytorch_ai is None:
            verdict["final_consensus_score"] = llm_ai
            verdict["resolution_reasoning"] = (
                "Detector (PyTorch) unavailable; using the contextual LLM verdict alone."
            )
            return verdict
        if llm_ai is None:
            # No safety net -- the raw model's known blind spots are uncorrected.
            verdict["final_consensus_score"] = pytorch_ai
            verdict["resolution_reasoning"] = (
                "Linguistic agent unavailable; using the raw model score WITHOUT a "
                "safety net (mode-collapse errors cannot be corrected)."
            )
            return verdict

        # -- Both present: run the safety-net logic ----------------------- #
        difference = abs(pytorch_ai - llm_ai)

        if difference >= self.conflict_threshold:
            verdict["conflict_detected"] = True

            # Scenario A: logit saturation (model says human, LLM says AI).
            if pytorch_ai < 10.0 and llm_ai > 60.0:
                verdict["override_type"] = "logit_saturation"
                verdict["final_consensus_score"] = llm_ai
                verdict["resolution_reasoning"] = (
                    "OVERRIDE: the PyTorch model showed logit saturation (mode "
                    "collapse), scoring synthetic text as human. The Linguistic "
                    f"agent caught the structural anomalies. LLM: {llm_reasoning}"
                )
            # Scenario B: ESL false-positive (model says AI, LLM says human).
            elif pytorch_ai > 80.0 and llm_ai < 30.0:
                verdict["override_type"] = "esl_false_positive"
                verdict["final_consensus_score"] = llm_ai
                verdict["resolution_reasoning"] = (
                    "OVERRIDE: the PyTorch model falsely flagged rigid non-native "
                    "(ESL) human phrasing as AI. The Linguistic agent verified "
                    f"human semantic intent. LLM: {llm_reasoning}"
                )
            # Scenario C: strong disagreement, neither extreme -> compromise.
            else:
                verdict["override_type"] = "compromise"
                verdict["final_consensus_score"] = round(
                    pytorch_ai * 0.4 + llm_ai * 0.6, 2
                )
                verdict["resolution_reasoning"] = (
                    "COMPROMISE: agents disagreed without hitting an extreme bound; "
                    "applied a 40/60 weighted consensus favouring the contextual LLM. "
                    f"LLM: {llm_reasoning}"
                )
        else:
            # Agreement -> average the two (both are informative and close).
            verdict["final_consensus_score"] = round((pytorch_ai + llm_ai) / 2, 2)
            verdict["resolution_reasoning"] = "Agents agree; averaged consensus."

        return verdict

    # ------------------------------------------------------------------ #
    # Rich per-paragraph API (produces a heatmap entry)
    # ------------------------------------------------------------------ #
    def resolve_paragraph(
        self,
        detector_payload: Dict[str, Any],
        linguistic_payload: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Resolve one paragraph and return a heatmap-ready entry."""
        meta = meta or {}
        verdict = self.resolve(detector_payload, linguistic_payload)
        score = verdict["final_consensus_score"]
        classification = self._classify(score)
        return {
            "paragraph_index": meta.get("paragraph_index"),
            "section": meta.get("section"),
            "text_preview": (meta.get("text") or "")[:200],
            "detector_score": verdict["original_pytorch_score"],
            "linguistic_score": verdict["original_llm_score"],
            "final_ai_score": score,
            "classification": classification,
            "heat_level": self._heat_level(score),
            "conflict_detected": verdict["conflict_detected"],
            "override_type": verdict["override_type"],
            "safety_net_active": verdict["safety_net_active"],
            "reasoning": verdict["resolution_reasoning"],
        }

    # ------------------------------------------------------------------ #
    # Document-level API (heatmap + weighted verdict)
    # ------------------------------------------------------------------ #
    def resolve_document(self, paragraph_inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Reconcile every paragraph and roll up a document verdict.

        ``paragraph_inputs`` is a list of dicts, each with:
            ``detector``    -> detector payload  ({"ai_probability": ...})
            ``linguistic``  -> linguistic payload ({"ai_probability": ..., "reasoning": ...})
            ``meta``        -> {"paragraph_index", "section", "text"} (optional)
        """
        heatmap: List[Dict[str, Any]] = []
        scored: List[float] = []
        conflicts = 0
        overrides = 0
        any_safety_net = False

        for item in paragraph_inputs:
            entry = self.resolve_paragraph(
                item.get("detector") or {},
                item.get("linguistic") or {},
                item.get("meta") or {},
            )
            heatmap.append(entry)
            if entry["final_ai_score"] is not None:
                scored.append(entry["final_ai_score"])
            if entry["conflict_detected"]:
                conflicts += 1
            if entry["override_type"] in {"logit_saturation", "esl_false_positive"}:
                overrides += 1
            if entry["safety_net_active"]:
                any_safety_net = True

        overall = round(sum(scored) / len(scored), 2) if scored else None
        classification = self._classify(overall)

        return {
            "overall_ai_score": overall,           # 0-100 or None
            "classification": classification,
            "paragraphs_analyzed": len(heatmap),
            "conflicts_detected": conflicts,
            "overrides_applied": overrides,
            "safety_net_active": any_safety_net,
            "flagged_paragraphs": [
                h["paragraph_index"] for h in heatmap
                if (h["final_ai_score"] or 0) >= _LIKELY_AI
            ],
            "heatmap": heatmap,
        }

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
    def _heat_level(score: Optional[float]) -> str:
        """Coarse band for UI heatmap colouring."""
        if score is None:
            return "unknown"
        if score >= _LIKELY_AI:
            return "high"
        if score <= _LIKELY_HUMAN:
            return "low"
        return "medium"


# === PROMPT LOGIC FOR THE LINGUISTIC AGENT ===
LINGUISTIC_AGENT_PROMPT = """
You are the Linguistic Analysis Agent in a multi-agent society designed to
detect AI-generated text in academic work (from student essays to research
papers). Another agent -- a PyTorch sequence classifier -- has analyzed this
text mathematically, but it has severe blind spots: non-native (ESL) writers,
heavily formatted text, and stylistic masking (slang, lowercase, sarcasm).

Your task is to analyze the semantic intent, tone, and structural flow of the
text to prevent false positives and false negatives.

Look specifically for:
1. "Patchwriting" / "Frankensteining" -- abrupt shifts between robotic and
   casual tones within one passage.
2. Advanced AI masking -- an LLM instructed to use slang, lowercase, or sarcasm
   to hide its default structured voice.
3. "The ESL penalty" -- rigid, textbook-perfect academic transitions that read
   as machine-like but are written by a non-native human.

Return ONLY a JSON object:
{
    "ai_probability": <float between 0.0 and 100.0>,
    "reasoning": "<1 sentence justifying the score from structural/semantic markers>"
}
"""
