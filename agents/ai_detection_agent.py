"""
AI Detection Agent.

Estimates how likely each section was AI-generated, combining two independent
signals (per the implementation plan, deliberately avoiding a hard logprobs
dependency):

1. LLM classifier (primary, weight 0.7): one Gemini call per section scoring
   AI probability 0-100 with reasoning.
2. Burstiness math (secondary, weight 0.3): pure statistics - sentence-length
   variation (coefficient of variation) + vocabulary diversity. Low variation
   and low diversity are consistent with AI-generated text. No API needed.

Methodology sections are down-weighted in the document score because their
formulaic style naturally mimics AI patterns (a documented false-positive
source). Scores are always reported with a classification band and confidence,
never a binary verdict.

Without a Gemini key the agent runs burstiness-only at reduced confidence.

CLI:  python -m agents.ai_detection_agent path/to/paper.(pdf|md|txt)
"""

from __future__ import annotations

import re
import statistics
from typing import Any, Dict, List, Optional

from agents.base import (
    BaseAgent,
    Reference,
    chunk_text,
    get_llm,
    llm_available,
    run_cli,
)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
_MAX_SECTION_LLM_CALLS = 8
_SECTION_CHAR_LIMIT = 6000

# Sections whose formulaic style tends to inflate AI scores -> lower weight.
_LOW_WEIGHT_SECTIONS = {"methodology", "method", "methods", "references", "bibliography"}

# Classification bands (AI probability 0-100)
_LIKELY_AI = 65
_LIKELY_HUMAN = 35


class AIDetectionAgent(BaseAgent):
    """Combine an LLM classifier with burstiness math to estimate AI authorship."""

    name = "AIDetectionAgent"
    needs_references = False

    def __init__(self, max_section_calls: int = _MAX_SECTION_LLM_CALLS):
        self.max_section_calls = max_section_calls
        self._gemini = get_llm()

    # ------------------------------------------------------------------ #
    def run(self, text: str, references: Optional[List[Reference]] = None) -> "Any":
        structured = chunk_text(text) if text else {}
        section_texts = self._section_texts(structured)

        if not section_texts:
            return self._result(
                status="warning",
                findings=["No analyzable text sections were found."],
                metadata={"overall_ai_score": None, "sections": []},
            )

        llm_enabled = self._gemini is not None and llm_available()
        llm_calls = 0
        sections: List[Dict[str, Any]] = []

        for name, content in section_texts.items():
            if name.lower() in {"references", "bibliography"}:
                continue

            burstiness = self._burstiness_signal(content)

            llm_score = None
            llm_reason = None
            if llm_enabled and llm_calls < self.max_section_calls and burstiness["analyzable"]:
                llm_calls += 1
                llm_score, llm_reason = self._llm_section_score(name, content[:_SECTION_CHAR_LIMIT])

            combined = self._combine(llm_score, burstiness["ai_signal"])
            sections.append({
                "section": name,
                "ai_score": combined,  # 0-100
                "llm_score": llm_score,
                "llm_reasoning": llm_reason,
                "burstiness_signal": burstiness["ai_signal"],
                "stats": burstiness["stats"],
                "weight": self._section_weight(name),
                "classification": self._classify(combined),
            })

        overall = self._weighted_overall(sections)
        classification = self._classify(overall) if overall is not None else "Uncertain"
        confidence = "medium" if llm_enabled else "low"

        findings = self._compose_findings(sections, overall, classification, llm_enabled)
        status = "warning" if (overall is not None and overall >= _LIKELY_AI) else "passed"

        metadata = {
            "overall_ai_score": overall,  # 0-100
            "classification": classification,
            "confidence": confidence,
            "method": "llm_classifier(0.7) + burstiness(0.3)" if llm_enabled else "burstiness_only",
            "llm_enabled": llm_enabled,
            "sections": sections,
            "disclaimer": (
                "AI detection is an indicator, not a verdict. Technical or formulaic "
                "writing can be misclassified; treat scores as probabilistic."
            ),
        }
        return self._result(status=status, findings=findings, metadata=metadata)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _section_texts(structured: Dict[str, List[List[str]]]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for section, paragraphs in structured.items():
            parts = [" ".join(sentences) for sentences in paragraphs]
            text = "\n\n".join(p for p in parts if p.strip())
            if text.strip():
                out[section] = text
        return out

    # ------------------------------------------------------------------ #
    # Burstiness math (no API)
    # ------------------------------------------------------------------ #
    def _burstiness_signal(self, text: str) -> Dict[str, Any]:
        sentences = self._split_sentences(text)
        lengths = [len(_WORD_RE.findall(s)) for s in sentences if _WORD_RE.findall(s)]
        words = [w.lower() for w in _WORD_RE.findall(text)]

        stats: Dict[str, Any] = {
            "sentence_count": len(lengths),
            "mean_sentence_length": round(statistics.mean(lengths), 2) if lengths else 0,
            "sentence_length_cv": None,
            "vocabulary_diversity": round(len(set(words)) / len(words), 3) if words else 0,
        }

        # Need at least a few sentences for variance to be meaningful.
        if len(lengths) < 3:
            stats["sentence_length_cv"] = None
            return {"ai_signal": 50.0, "analyzable": False, "stats": stats}

        mean_len = statistics.mean(lengths)
        std_len = statistics.pstdev(lengths)
        cv = (std_len / mean_len) if mean_len else 0.0
        stats["sentence_length_cv"] = round(cv, 3)

        # Lower variation (cv) -> more AI-like. Map cv 0.6->~15, 0.2->~85.
        cv_ai = self._map(cv, hi_x=0.6, hi_y=15.0, lo_x=0.2, lo_y=85.0)
        # Lower diversity -> more AI-like. Map ttr 0.6->~15, 0.3->~80.
        ttr = stats["vocabulary_diversity"]
        div_ai = self._map(ttr, hi_x=0.6, hi_y=15.0, lo_x=0.3, lo_y=80.0)

        ai_signal = round(0.6 * cv_ai + 0.4 * div_ai, 1)
        return {"ai_signal": ai_signal, "analyzable": True, "stats": stats}

    @staticmethod
    def _map(x: float, hi_x: float, hi_y: float, lo_x: float, lo_y: float) -> float:
        """
        Linear map where x>=hi_x -> hi_y and x<=lo_x -> lo_y (hi_x>lo_x),
        clamped to [min(y), max(y)].
        """
        if x >= hi_x:
            return hi_y
        if x <= lo_x:
            return lo_y
        frac = (x - lo_x) / (hi_x - lo_x)
        return lo_y + frac * (hi_y - lo_y)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
        return [p.strip() for p in parts if p.strip()]

    # ------------------------------------------------------------------ #
    # LLM classifier
    # ------------------------------------------------------------------ #
    def _llm_section_score(self, name: str, content: str):
        prompt = (
            f"Analyze the '{name}' section of an academic paper for patterns "
            "consistent with AI-generated writing. Consider structural uniformity, "
            "vocabulary diversity, hedging language, and stylistic variation.\n\n"
            f"SECTION:\n\"\"\"\n{content}\n\"\"\"\n\n"
            "Respond ONLY as JSON with keys: "
            '{"ai_probability": <0-100 number>, "reasoning": "one or two sentences"}'
        )
        try:
            data = self._gemini.call_llm_json(
                prompt,
                system_instruction=(
                    "You are a careful AI-text-detection analyst. Provide a calibrated "
                    "probability, not a binary judgment."
                ),
            )
        except Exception:
            data = None
        if not data or "ai_probability" not in data:
            return None, None
        try:
            score = float(data["ai_probability"])
        except (TypeError, ValueError):
            return None, None
        return max(0.0, min(100.0, score)), data.get("reasoning")

    # ------------------------------------------------------------------ #
    # Combination + scoring
    # ------------------------------------------------------------------ #
    @staticmethod
    def _combine(llm_score: Optional[float], burstiness_signal: float) -> float:
        if llm_score is None:
            return round(burstiness_signal, 1)
        return round(0.7 * llm_score + 0.3 * burstiness_signal, 1)

    @staticmethod
    def _section_weight(name: str) -> float:
        return 0.5 if name.lower() in _LOW_WEIGHT_SECTIONS else 1.0

    def _weighted_overall(self, sections: List[Dict[str, Any]]) -> Optional[float]:
        if not sections:
            return None
        num = sum(s["ai_score"] * s["weight"] for s in sections)
        den = sum(s["weight"] for s in sections)
        return round(num / den, 1) if den else None

    @staticmethod
    def _classify(score: Optional[float]) -> str:
        if score is None:
            return "Uncertain"
        if score >= _LIKELY_AI:
            return "Likely AI"
        if score <= _LIKELY_HUMAN:
            return "Likely Human"
        return "Uncertain"

    def _compose_findings(
        self,
        sections: List[Dict[str, Any]],
        overall: Optional[float],
        classification: str,
        llm_enabled: bool,
    ) -> List[str]:
        findings: List[str] = []
        if overall is not None:
            findings.append(f"Overall AI-content score: {overall}% ({classification}).")
        flagged = [s for s in sections if s["ai_score"] >= _LIKELY_AI]
        for s in flagged:
            findings.append(
                f"Section '{s['section']}' scores {s['ai_score']}% "
                f"(patterns consistent with AI-generated text)."
            )
        if not llm_enabled:
            findings.append(
                "Burstiness-only analysis (no Gemini API key) - lower confidence; "
                "add a key for the LLM classifier signal."
            )
        if not flagged:
            findings.append("No section shows a strong AI-generation signal.")
        return findings


if __name__ == "__main__":
    run_cli(AIDetectionAgent(), "Estimate AI-generated content probability.")
