"""
Plagiarism Agent.

Finds paper text that matches existing published sources, using only free
resources (per the implementation plan). For each substantial paragraph:

1. Extract a distinctive key phrase.
2. Web path      - Serper exact-phrase search (needs SERPER_API_KEY).
3. Scholarly path- CrossRef bibliographic search (no key) -> Semantic Scholar
                   abstract for the top candidate.
4. Similarity    - Gemini compares the paragraph against each retrieved snippet
                   / abstract and rates similarity 0-100.

The agent is explicit about its limitations: it cannot access Turnitin's private
student-paper database and only covers open web + open-access scholarly content.
It is a pre-submission self-check, not a Turnitin replacement.

Degrades gracefully: without a Serper key it uses the scholarly path only;
without a Gemini key it reports raw candidate sources without similarity scores.

CLI:  python -m agents.plagiarism_agent path/to/paper.(pdf|md|txt)
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from agents.base import (
    BaseAgent,
    Reference,
    get_llm,
    iter_paragraphs,
    llm_available,
    run_cli,
    split_body_and_references,
)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
# Only check paragraphs with at least this many words (skip headers/captions).
_MIN_PARAGRAPH_WORDS = 30
# Cap paragraphs checked per paper (protects Serper's 2,500/month credits).
_MAX_PARAGRAPHS = 10
# Similarity (0-100) at or above which a match is considered significant.
_SIMILARITY_THRESHOLD = 70

_LIMITATION_NOTE = (
    "Covers open web and open-access scholarly sources only. It cannot access "
    "Turnitin's proprietary student-paper database. Use as a pre-submission "
    "self-check, not a Turnitin replacement."
)


class PlagiarismAgent(BaseAgent):
    """Detect text overlap with open web + open-access scholarly sources."""

    name = "PlagiarismAgent"
    needs_references = False

    def __init__(self, max_paragraphs: int = _MAX_PARAGRAPHS):
        self.max_paragraphs = max_paragraphs
        self._gemini = get_llm()
        from services import serper, crossref, semantic_scholar

        self._serper = serper
        self._crossref = crossref
        self._s2 = semantic_scholar

    # ------------------------------------------------------------------ #
    def run(self, text: str, references: Optional[List[Reference]] = None) -> "Any":
        body_text, _refs = split_body_and_references(text)
        paragraphs = self._select_paragraphs(body_text or text)

        web_enabled = bool(os.getenv("SERPER_API_KEY"))
        llm_enabled = self._gemini is not None and llm_available()

        if not paragraphs:
            return self._result(
                status="warning",
                findings=["No substantial paragraphs found to check for plagiarism."],
                metadata={"plagiarism_score": None, "checked_paragraphs": 0, "matches": []},
            )

        checks: List[Dict[str, Any]] = []
        best_similarities: List[float] = []

        for i, (section, para) in enumerate(paragraphs, start=1):
            phrase = self._key_phrase(para)
            candidates = self._gather_candidates(phrase, web_enabled)

            best_sim = 0.0
            scored_sources: List[Dict[str, Any]] = []
            for cand in candidates:
                snippet = cand.get("snippet") or ""
                similarity = None
                if llm_enabled and snippet:
                    similarity = self._score_similarity(para, snippet, cand.get("title", ""))
                    if similarity is not None:
                        best_sim = max(best_sim, similarity)
                scored_sources.append({
                    "source": cand.get("source"),
                    "title": cand.get("title"),
                    "link": cand.get("link"),
                    "snippet": snippet[:300],
                    "similarity": similarity,
                })

            if llm_enabled:
                best_similarities.append(best_sim)

            checks.append({
                "paragraph_index": i,
                "section": section,
                "key_phrase": phrase,
                "candidate_count": len(scored_sources),
                "best_similarity": best_sim if llm_enabled else None,
                "flagged": bool(llm_enabled and best_sim >= _SIMILARITY_THRESHOLD),
                "sources": scored_sources,
            })

        score = (
            round(sum(best_similarities) / len(best_similarities), 1)
            if best_similarities else None
        )
        flagged = [c for c in checks if c["flagged"]]

        findings = self._compose_findings(checks, flagged, score, web_enabled, llm_enabled)
        status = self._status(score, flagged)

        metadata = {
            "plagiarism_score": score,  # 0-100 avg best-match similarity, or None
            "checked_paragraphs": len(checks),
            "flagged_paragraph_count": len(flagged),
            "web_search_enabled": web_enabled,
            "similarity_scoring_enabled": llm_enabled,
            "similarity_threshold": _SIMILARITY_THRESHOLD,
            "limitations": _LIMITATION_NOTE,
            "matches": checks,
        }
        return self._result(status=status, findings=findings, metadata=metadata)

    # ------------------------------------------------------------------ #
    # Paragraph selection + phrase extraction
    # ------------------------------------------------------------------ #
    def _select_paragraphs(self, text: str):
        paras = [
            (section, para)
            for section, para in iter_paragraphs(text)
            if section.lower() not in {"references", "bibliography"}
            and len(_WORD_RE.findall(para)) >= _MIN_PARAGRAPH_WORDS
        ]
        return paras[: self.max_paragraphs]

    @staticmethod
    def _key_phrase(paragraph: str, max_words: int = 12) -> str:
        """Pick the longest sentence and trim to a distinctive exact-search phrase."""
        sentences = re.split(r"(?<=[.!?])\s+", paragraph.replace("\n", " "))
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return paragraph[:120]
        longest = max(sentences, key=lambda s: len(_WORD_RE.findall(s)))
        words = longest.split()
        if len(words) <= max_words:
            return longest.strip('"')
        start = max(0, (len(words) - max_words) // 2)  # middle window, most distinctive
        return " ".join(words[start:start + max_words]).strip('"')

    # ------------------------------------------------------------------ #
    # Candidate retrieval
    # ------------------------------------------------------------------ #
    def _gather_candidates(self, phrase: str, web_enabled: bool) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []

        # Web path (Serper) - exact phrase search.
        if web_enabled:
            try:
                results = self._serper.search_web(f'"{phrase}"', num_results=5) or []
            except Exception:
                results = []
            for r in results[:5]:
                candidates.append({
                    "source": "web",
                    "title": r.get("title"),
                    "link": r.get("link"),
                    "snippet": r.get("snippet", ""),
                })

        # Scholarly path (CrossRef -> Semantic Scholar abstract). No key needed.
        try:
            cr = self._crossref.search_works_by_title(phrase)
        except Exception:
            cr = None
        item = self._top_crossref_item(cr)
        if item:
            abstract = self._scholarly_abstract(item)
            candidates.append({
                "source": "scholarly",
                "title": (item.get("title") or [""])[0],
                "link": item.get("URL") or (f"https://doi.org/{item.get('DOI')}" if item.get("DOI") else None),
                "snippet": abstract or "",
            })
        return candidates

    @staticmethod
    def _top_crossref_item(cr: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not cr or not isinstance(cr, dict) or "error" in cr:
            return None
        items = (cr.get("message") or {}).get("items") or []
        return items[0] if items else None

    def _scholarly_abstract(self, item: Dict[str, Any]) -> Optional[str]:
        doi = item.get("DOI")
        if not doi:
            return None
        try:
            data = self._s2.get_paper_by_doi(doi)
        except Exception:
            data = None
        if not data or not isinstance(data, dict) or "error" in data:
            return None
        abstract = (data.get("abstract") or "").strip()
        return abstract or None

    # ------------------------------------------------------------------ #
    # Similarity scoring
    # ------------------------------------------------------------------ #
    def _score_similarity(self, paragraph: str, source_text: str, source_title: str) -> Optional[float]:
        prompt = (
            "Compare the two passages and rate how similar they are in content and "
            "wording, as a plagiarism-style similarity score.\n\n"
            f"PASSAGE A (from the submitted paper):\n\"\"\"\n{paragraph[:1500]}\n\"\"\"\n\n"
            f"PASSAGE B (from '{source_title}'):\n\"\"\"\n{source_text[:1500]}\n\"\"\"\n\n"
            "Respond ONLY as JSON: "
            '{"similarity": <0-100 number>, "reasoning": "one sentence"}'
        )
        try:
            data = self._gemini.call_llm_json(
                prompt,
                system_instruction="You are a plagiarism-analysis assistant. Score conservatively.",
            )
        except Exception:
            data = None
        if not data or "similarity" not in data:
            return None
        try:
            return max(0.0, min(100.0, float(data["similarity"])))
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def _compose_findings(
        self,
        checks: List[Dict[str, Any]],
        flagged: List[Dict[str, Any]],
        score: Optional[float],
        web_enabled: bool,
        llm_enabled: bool,
    ) -> List[str]:
        findings: List[str] = []
        if score is not None:
            findings.append(f"Average best-match similarity across checked paragraphs: {score}%.")
        for c in flagged:
            top = max(c["sources"], key=lambda s: (s.get("similarity") or 0), default=None)
            src = f" (source: {top.get('link')})" if top and top.get("link") else ""
            findings.append(
                f"Paragraph {c['paragraph_index']} in '{c['section']}' closely matches "
                f"an existing source ({c['best_similarity']}%)" + src + "."
            )
        if not web_enabled:
            findings.append(
                "Web search disabled (no SERPER_API_KEY); used the open-access scholarly "
                "path only. Add a Serper key for broader web coverage."
            )
        if not llm_enabled:
            findings.append(
                "Similarity scoring skipped (no Gemini API key); candidate sources are "
                "listed without similarity percentages."
            )
        if not flagged and (web_enabled or llm_enabled):
            findings.append("No paragraph exceeded the similarity threshold.")
        findings.append("Limitation: " + _LIMITATION_NOTE)
        return findings

    @staticmethod
    def _status(score: Optional[float], flagged: List[Dict[str, Any]]) -> str:
        if flagged or (score is not None and score >= 50):
            return "failed"
        if score is not None and score >= 20:
            return "warning"
        return "passed"


if __name__ == "__main__":
    run_cli(PlagiarismAgent(), "Check for text overlap with open sources.")
