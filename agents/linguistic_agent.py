"""
Linguistic Agent  --  "The Brain" half of the AI-detection Safety Net.

Where the Detector Agent looks only at token statistics, this agent asks an LLM
to read a paragraph *contextually* -- tone, structure, semantic intent -- to
judge whether it reads like machine-generated text masquerading as human, or a
human (e.g. a non-native/ESL writer) whose rigid style fools the raw classifier.

It exists specifically to catch the Detector's blind spots:
  * "Patchwriting" / Frankenstein tone shifts.
  * Style-masked AI (ChatGPT told to use slang, lowercase, sarcasm).
  * The "ESL penalty" (textbook-perfect transitions written by a human).

The system prompt is the shared ``LINGUISTIC_AGENT_PROMPT`` from
``conflict_resolver`` so the two stay in lockstep. Output per paragraph is
``{"ai_probability": 0-100, "reasoning": "..."}``.

Without a Gemini key the agent degrades gracefully (scores become ``None``).
The LLM backend is pluggable via a callable so Qwen (or any other model) can be
dropped in later without touching the agent logic.

CLI:  python -m agents.linguistic_agent path/to/paper.(pdf|md|txt)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from agents.base import (
    BaseAgent,
    Reference,
    get_llm,
    iter_paragraphs,
    llm_available,
    run_cli,
    split_body_and_references,
)
from agents.conflict_resolver import LINGUISTIC_AGENT_PROMPT

# Only analyze paragraphs with at least this many characters.
_MIN_PARAGRAPH_CHARS = 40
# Cap LLM calls per paper to protect free-tier rate limits.
_MAX_LLM_CALLS = 40
# Truncate very long paragraphs before sending (token control).
_PARAGRAPH_CHAR_LIMIT = 4000
# Classification bands (shared with the Detector side).
_LIKELY_AI = 65
_LIKELY_HUMAN = 35

# A JSON-returning LLM callable: (prompt, system_instruction) -> dict | None
LLMCallable = Callable[[str, Optional[str]], Optional[Dict[str, Any]]]


class LinguisticAgent(BaseAgent):
    """Contextual LLM-based AI-text analyst (per-paragraph)."""

    name = "LinguisticAgent"
    needs_references = False

    def __init__(
        self,
        llm_json: Optional[LLMCallable] = None,
        max_calls: int = _MAX_LLM_CALLS,
    ):
        """
        Args:
            llm_json: optional callable ``(prompt, system_instruction) -> dict``.
                Defaults to Gemini (``services.gemini.call_llm_json``). Supplying
                a different callable is how a Qwen backend would be plugged in.
            max_calls: cap on LLM calls per paper.
        """
        self.max_calls = max_calls
        self._calls_made = 0
        if llm_json is not None:
            self._llm_json = llm_json
            self._backend = "custom"
        else:
            gemini = get_llm()
            self._llm_json = gemini.call_llm_json if gemini is not None else None
            self._backend = "gemini"

    # ------------------------------------------------------------------ #
    def available(self) -> bool:
        """True when a usable LLM backend is configured."""
        if self._llm_json is None:
            return False
        # For the default Gemini backend, also require an API key to be present.
        if self._backend == "gemini":
            return llm_available()
        return True

    # ------------------------------------------------------------------ #
    # Low-level scoring (used directly by the Orchestrator's safety net)
    # ------------------------------------------------------------------ #
    def score_text(self, text: str, force: bool = False) -> Dict[str, Any]:
        """
        Contextually score a single block of text.

        Returns ``{"ai_probability": float|None, "reasoning": str|None,
        "available": bool}``. ``force`` bypasses the per-paper call cap (used by
        the orchestrator when it must resolve a specific conflict).
        """
        if not self.available() or not (text and text.strip()):
            return {"ai_probability": None, "reasoning": None, "available": False}
        if not force and self._calls_made >= self.max_calls:
            return {"ai_probability": None, "reasoning": "call cap reached", "available": False}

        prompt = (
            "Analyze the following text block and return your JSON verdict.\n\n"
            f"TEXT BLOCK:\n\"\"\"\n{text[:_PARAGRAPH_CHAR_LIMIT]}\n\"\"\"\n"
        )
        try:
            self._calls_made += 1
            data = self._llm_json(prompt, LINGUISTIC_AGENT_PROMPT)
        except Exception:
            data = None

        if not data or "ai_probability" not in data:
            return {"ai_probability": None, "reasoning": None, "available": False}
        try:
            score = max(0.0, min(100.0, float(data["ai_probability"])))
        except (TypeError, ValueError):
            return {"ai_probability": None, "reasoning": data.get("reasoning"), "available": False}
        return {
            "ai_probability": round(score, 2),
            "reasoning": data.get("reasoning"),
            "available": True,
        }

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

        if not self.available():
            return self._result(
                status="warning",
                findings=[
                    "Linguistic Agent unavailable (no LLM backend / API key); "
                    "contextual AI analysis skipped."
                ],
                metadata={"enabled": False, "backend": self._backend,
                          "overall_ai_score": None, "paragraphs": []},
            )

        if not paragraphs:
            return self._result(
                status="warning",
                findings=["No analyzable paragraphs were found."],
                metadata={"enabled": True, "backend": self._backend,
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
                "reasoning": scored["reasoning"],
                "classification": self._classify(ai_prob),
                "text_preview": para[:160],
            })

        overall = round(sum(ai_values) / len(ai_values), 2) if ai_values else None
        classification = self._classify(overall)
        findings = self._compose_findings(para_scores, overall, classification)
        status = "warning" if (overall is not None and overall >= _LIKELY_AI) else "passed"

        metadata = {
            "enabled": True,
            "backend": self._backend,
            "overall_ai_score": overall,
            "classification": classification,
            "method": "llm_contextual_classifier",
            "llm_calls_made": self._calls_made,
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
            findings.append(f"Linguistic overall AI score: {overall}% ({classification}).")
        flagged = [p for p in para_scores if (p["ai_probability"] or 0) >= _LIKELY_AI]
        for p in flagged[:10]:
            reason = f" - {p['reasoning']}" if p.get("reasoning") else ""
            findings.append(
                f"Paragraph {p['paragraph_index']} ('{p['section']}'): "
                f"{p['ai_probability']}% AI (contextual){reason}"
            )
        if not flagged:
            findings.append("No paragraph flagged as AI by contextual analysis.")
        return findings


if __name__ == "__main__":
    run_cli(LinguisticAgent(), "Contextual LLM AI-text analyst (the Brain).")
