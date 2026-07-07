"""
Writing Quality Agent.

Assesses academic writing quality along three axes:

1. Structure   - are the expected sections present (Abstract, Introduction,
                 Methods, Results, Discussion, Conclusion, References)?
2. Readability - pure-math statistics (sentence length, long-sentence ratio,
                 vocabulary diversity). No API required, always available.
3. Prose       - per-section LLM assessment of grammar, academic tone, clarity,
                 hedging, and tense consistency, with actionable suggestions.

Without a Gemini key the agent still returns a full structural + readability
report and simply omits the LLM prose assessment.

CLI:  python -m agents.quality_agent path/to/paper.(pdf|md|txt)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from agents.base import (
    BaseAgent,
    Reference,
    chunk_text,
    get_llm,
    llm_available,
    run_cli,
    split_body_and_references,
)

# Sections we expect a complete empirical paper to contain.
# Each entry: canonical name -> list of case-insensitive keywords that satisfy it.
_EXPECTED_SECTIONS = {
    "Abstract": ["abstract"],
    "Introduction": ["introduction"],
    "Methodology": ["method", "methodology", "materials", "approach"],
    "Results": ["result", "findings", "evaluation", "experiment"],
    "Discussion": ["discussion"],
    "Conclusion": ["conclusion"],
    "References": ["reference", "bibliography"],
}

# Cap LLM section assessments to protect free-tier rate limits.
_MAX_SECTION_LLM_CALLS = 8
# Truncate very long sections before sending to the LLM (token control).
_SECTION_CHAR_LIMIT = 6000
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


class QualityAgent(BaseAgent):
    """Evaluate academic writing quality (structure, readability, prose)."""

    name = "QualityAgent"
    needs_references = False

    def __init__(self, max_section_calls: int = _MAX_SECTION_LLM_CALLS):
        self.max_section_calls = max_section_calls
        self._gemini = get_llm()

    # ------------------------------------------------------------------ #
    def run(self, text: str, references: Optional[List[Reference]] = None) -> "Any":
        structured = chunk_text(text) if text else {}
        section_texts = self._section_texts(structured)

        structure = self._analyze_structure(section_texts)
        body_text, _refs = split_body_and_references(text)
        readability = self._readability_stats(body_text or text)
        style = self._citation_style_consistency(body_text or text)

        llm_enabled = self._gemini is not None and llm_available()
        section_reports: List[Dict[str, Any]] = []
        llm_scores: List[float] = []
        if llm_enabled:
            section_reports, llm_scores = self._assess_sections_llm(section_texts)

        overall = self._overall_score(structure, readability, llm_scores)

        findings = self._compose_findings(structure, readability, style, section_reports, llm_enabled)
        status = self._status(structure, readability, overall)

        metadata = {
            "overall_quality_score": overall,  # 0-10
            "structure": structure,
            "readability": readability,
            "citation_style": style,
            "llm_prose_assessment_enabled": llm_enabled,
            "section_reports": section_reports,
        }
        return self._result(status=status, findings=findings, metadata=metadata)

    # ------------------------------------------------------------------ #
    # Section reconstruction
    # ------------------------------------------------------------------ #
    @staticmethod
    def _section_texts(structured: Dict[str, List[List[str]]]) -> Dict[str, str]:
        """Rebuild {section_name: joined_text} from the chunker output."""
        out: Dict[str, str] = {}
        for section, paragraphs in structured.items():
            parts = [" ".join(sentences) for sentences in paragraphs]
            text = "\n\n".join(p for p in parts if p.strip())
            if text.strip():
                out[section] = text
        return out

    # ------------------------------------------------------------------ #
    # Structure analysis
    # ------------------------------------------------------------------ #
    def _analyze_structure(self, section_texts: Dict[str, str]) -> Dict[str, Any]:
        present_names = [name.lower() for name in section_texts.keys()]
        present: List[str] = []
        missing: List[str] = []
        for canonical, keywords in _EXPECTED_SECTIONS.items():
            found = any(
                any(kw in name for kw in keywords) for name in present_names
            )
            (present if found else missing).append(canonical)

        score = round(len(present) / len(_EXPECTED_SECTIONS), 3)
        return {
            "expected": list(_EXPECTED_SECTIONS.keys()),
            "present": present,
            "missing": missing,
            "detected_sections": list(section_texts.keys()),
            "completeness_score": score,  # 0-1
        }

    # ------------------------------------------------------------------ #
    # Readability statistics (pure math)
    # ------------------------------------------------------------------ #
    def _readability_stats(self, text: str) -> Dict[str, Any]:
        sentences = self._split_sentences(text)
        words = _WORD_RE.findall(text)
        n_words = len(words)
        n_sentences = max(1, len(sentences))

        sent_lengths = [len(_WORD_RE.findall(s)) for s in sentences] or [0]
        avg_sentence_len = round(n_words / n_sentences, 2)
        long_sentences = [ln for ln in sent_lengths if ln > 40]
        long_ratio = round(len(long_sentences) / n_sentences, 3)

        lowered = [w.lower() for w in words]
        vocab_diversity = round(len(set(lowered)) / max(1, n_words), 3)
        avg_word_len = round(sum(len(w) for w in words) / max(1, n_words), 2)

        # Map stats to a 0-1 readability quality signal.
        # Peak quality near an 18-word average sentence; penalize long-sentence load.
        length_fit = max(0.0, 1.0 - abs(avg_sentence_len - 18) / 18)
        readability_score = round(max(0.0, length_fit - 0.5 * long_ratio), 3)

        return {
            "word_count": n_words,
            "sentence_count": len(sentences),
            "avg_sentence_length": avg_sentence_len,
            "avg_word_length": avg_word_len,
            "long_sentence_count": len(long_sentences),
            "long_sentence_ratio": long_ratio,
            "vocabulary_diversity": vocab_diversity,
            "readability_score": readability_score,  # 0-1
        }

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
        return [p.strip() for p in parts if p.strip()]

    # ------------------------------------------------------------------ #
    # Citation-style consistency (heuristic)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _citation_style_consistency(text: str) -> Dict[str, Any]:
        numeric = len(re.findall(r"\[\d+(?:\s*[,\-–]\s*\d+)*\]", text))
        author_year = len(re.findall(r"\([A-Z][A-Za-z]+(?: et al\.?)?,?\s+\d{4}[a-z]?\)", text))
        styles_used = [s for s, n in (("numeric", numeric), ("author-year", author_year)) if n]
        consistent = len(styles_used) <= 1
        return {
            "numeric_citations": numeric,
            "author_year_citations": author_year,
            "styles_used": styles_used,
            "consistent": consistent,
        }

    # ------------------------------------------------------------------ #
    # Per-section LLM prose assessment
    # ------------------------------------------------------------------ #
    def _assess_sections_llm(self, section_texts: Dict[str, str]):
        reports: List[Dict[str, Any]] = []
        scores: List[float] = []
        calls = 0
        for name, content in section_texts.items():
            if name.lower() in {"references", "bibliography"}:
                continue
            if len(content.split()) < 25:  # skip trivially short sections
                continue
            if calls >= self.max_section_calls:
                break
            calls += 1
            report = self._assess_one_section(name, content[:_SECTION_CHAR_LIMIT])
            if report is not None:
                reports.append(report)
                if isinstance(report.get("quality_score"), (int, float)):
                    scores.append(float(report["quality_score"]))
        return reports, scores

    def _assess_one_section(self, name: str, content: str) -> Optional[Dict[str, Any]]:
        prompt = (
            f"Assess the academic writing quality of the '{name}' section below.\n\n"
            f"SECTION:\n\"\"\"\n{content}\n\"\"\"\n\n"
            "Evaluate grammar, academic tone (vs casual), clarity/conciseness, "
            "appropriate hedging (e.g. 'suggests' vs 'proves'), and tense "
            "consistency. Respond ONLY as JSON with keys: "
            '{"quality_score": <0-10 number>, '
            '"tone": "academic|mixed|casual", '
            '"issues": [<short strings>], '
            '"suggestions": [<short actionable strings>]}'
        )
        try:
            data = self._gemini.call_llm_json(
                prompt,
                system_instruction="You are an experienced academic writing editor.",
            )
        except Exception:
            data = None
        if not data:
            return None
        return {
            "section": name,
            "quality_score": data.get("quality_score"),
            "tone": data.get("tone"),
            "issues": data.get("issues", []),
            "suggestions": data.get("suggestions", []),
        }

    # ------------------------------------------------------------------ #
    # Scoring + reporting
    # ------------------------------------------------------------------ #
    def _overall_score(
        self,
        structure: Dict[str, Any],
        readability: Dict[str, Any],
        llm_scores: List[float],
    ) -> float:
        struct10 = structure["completeness_score"] * 10
        read10 = readability["readability_score"] * 10
        if llm_scores:
            llm10 = sum(llm_scores) / len(llm_scores)
            overall = 0.5 * llm10 + 0.25 * struct10 + 0.25 * read10
        else:
            overall = 0.5 * struct10 + 0.5 * read10
        return round(overall, 2)

    def _compose_findings(
        self,
        structure: Dict[str, Any],
        readability: Dict[str, Any],
        style: Dict[str, Any],
        section_reports: List[Dict[str, Any]],
        llm_enabled: bool,
    ) -> List[str]:
        findings: List[str] = []
        if structure["missing"]:
            findings.append(
                "Missing expected section(s): " + ", ".join(structure["missing"]) + "."
            )
        else:
            findings.append("All expected sections are present.")

        if readability["long_sentence_ratio"] > 0.15:
            findings.append(
                f"{readability['long_sentence_count']} very long sentence(s) "
                f"(>40 words) may hurt readability."
            )
        if readability["avg_sentence_length"] > 30:
            findings.append(
                f"Average sentence length is high ({readability['avg_sentence_length']} "
                f"words); consider tightening prose."
            )

        if not style["consistent"]:
            findings.append(
                "Inconsistent citation style: both "
                + " and ".join(style["styles_used"])
                + " formats are used."
            )

        if llm_enabled:
            for rep in section_reports:
                issues = rep.get("issues") or []
                if rep.get("tone") == "casual":
                    findings.append(f"'{rep['section']}': tone reads as casual for academic writing.")
                if issues:
                    findings.append(f"'{rep['section']}': " + "; ".join(str(i) for i in issues[:2]))
        else:
            findings.append(
                "LLM prose assessment skipped (no Gemini API key); "
                "structure and readability were still analyzed."
            )
        return findings

    def _status(
        self,
        structure: Dict[str, Any],
        readability: Dict[str, Any],
        overall: float,
    ) -> str:
        if structure["missing"] or overall < 5:
            return "warning"
        if overall < 7:
            return "warning"
        return "passed"


if __name__ == "__main__":
    run_cli(QualityAgent(), "Assess academic writing quality.")
