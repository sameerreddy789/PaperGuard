"""
Plagiarism Agent.

Finds paper text that matches existing published sources, using only free
resources (per the implementation plan). For each substantial paragraph:

1. Extract a distinctive key phrase.
2. Web path      - Serper exact-phrase search (needs SERPER_API_KEY).
3. Scholarly path- CrossRef bibliographic search (no key) -> Semantic Scholar
                   abstract for the top candidate.
4. Similarity is scored THREE ways per candidate, each degrading independently:
     a. n-gram/shingle overlap  - deterministic word-shingle Jaccard overlap
        (no LLM, no API key). Catches verbatim / near-verbatim copy-paste.
     b. Semantic embedding      - cosine similarity of the detector's own
        DistilBERT embeddings (reuses ``DetectorAgent.embed_text``; needs only
        torch/transformers, no API key). Catches paraphrased overlap.
     c. LLM judgment            - Gemini rates similarity 0-100 (needs a key).
        Most nuanced, but the only one that costs a network call + API key.
   The best (max) of whichever are available is used as the paragraph's
   overlap score, so plagiarism scoring degrades gracefully but never fully
   turns off (n-gram overlap always runs; semantic runs whenever the detector
   model loads, which AI-detection already requires).
5. Cross-agent dedupe: a match is downgraded (not counted as "flagged") when
   the paragraph is both quoted AND cites a reference already extracted from
   the paper -- i.e. it looks like a properly attributed quotation rather than
   plagiarism. See ``_looks_quoted_and_cited``.

The agent is explicit about its limitations: it cannot access Turnitin's private
student-paper database and only covers open web + open-access scholarly content.
It is a pre-submission self-check, not a Turnitin replacement.

CLI:  python -m agents.plagiarism_agent path/to/paper.(pdf|md|txt)
"""

from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from agents.base import (
    BaseAgent,
    Reference,
    extract_references,
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
# Word-shingle size for the deterministic n-gram overlap check.
_SHINGLE_SIZE = 8

_LIMITATION_NOTE = (
    "Covers open web and open-access scholarly sources only. It cannot access "
    "Turnitin's proprietary student-paper database. Use as a pre-submission "
    "self-check, not a Turnitin replacement."
)


# --------------------------------------------------------------------------- #
# Deterministic n-gram / shingle overlap (no LLM, no API key)
# --------------------------------------------------------------------------- #
def _shingles(text: str, n: int = _SHINGLE_SIZE) -> Set[int]:
    """Hashed word n-grams ("shingles") of ``text``, for Jaccard overlap."""
    words = [w.lower() for w in _WORD_RE.findall(text or "")]
    if len(words) < n:
        return {hash(tuple(words))} if words else set()
    return {hash(tuple(words[i:i + n])) for i in range(len(words) - n + 1)}


def _jaccard(a: Set[int], b: Set[int]) -> float:
    """Jaccard similarity of two shingle sets, in [0, 1]."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# --------------------------------------------------------------------------- #
# Semantic embedding similarity (reuses the detector's DistilBERT encoder)
# --------------------------------------------------------------------------- #
def _cosine(a: Optional[List[float]], b: Optional[List[float]]) -> Optional[float]:
    if not a or not b or len(a) != len(b):
        return None
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return None
    return dot / (na * nb)


# --------------------------------------------------------------------------- #
# Cross-agent dedupe: properly quoted + cited spans aren't plagiarism
# --------------------------------------------------------------------------- #
_QUOTE_RE = re.compile(r'["\u201c\u201d].{15,}?["\u201c\u201d]')


def _looks_quoted_and_cited(paragraph: str, references: List[Reference]) -> bool:
    """
    Heuristic cross-agent dedupe (per TASKS.md): if a paragraph both (a)
    contains a quoted span and (b) cites a reference that the citation agent's
    own extraction found in this paper, treat overlap as an attributed
    quotation rather than plagiarism. Deterministic and cheap: it does not
    require re-running the full CitationAgent (existence/claim verification),
    only the already-extracted reference list, so it works even when the
    citation and plagiarism agents run independently via the CrewAI tools.
    """
    if not references or not _QUOTE_RE.search(paragraph or ""):
        return False
    # Numeric citation: [12], (12), [3, 5-7]
    for m in re.finditer(r"[\[(]([\d,;\s\-\u2013]+)[\])]", paragraph):
        if any(ch.isdigit() for ch in m.group(1)):
            return True
    # Author-year citation: "Smith (2020)" / "(Smith, 2020)"
    for ref in references:
        if not ref.authors or not ref.year:
            continue
        lastname = ref.authors[0].split(",")[0].split()[-1] if ref.authors[0].split() else None
        if lastname and re.search(
            rf"{re.escape(lastname)}[^.]{{0,40}}\b{ref.year}\b", paragraph, re.IGNORECASE
        ):
            return True
    return False


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
        self._detector = None  # lazy: only load DistilBERT if we reach _select_paragraphs

    def _get_detector(self):
        """Lazily construct the shared DetectorAgent for semantic embeddings."""
        if self._detector is None:
            from agents.detector_agent import DetectorAgent
            self._detector = DetectorAgent()
        return self._detector

    # ------------------------------------------------------------------ #
    def run(self, text: str, references: Optional[List[Reference]] = None) -> "Any":
        if references is None:
            try:
                references = extract_references(text)
            except Exception:
                references = []

        body_text, _refs = split_body_and_references(text)
        paragraphs = self._select_paragraphs(body_text or text)

        web_enabled = bool(os.getenv("SERPER_API_KEY"))
        llm_enabled = self._gemini is not None and llm_available()
        detector = self._get_detector()
        semantic_enabled = detector.score_text("probe").get("available", False)

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
            para_shingles = _shingles(para)
            para_embedding = detector.embed_text(para) if semantic_enabled else None

            best_sim = 0.0
            best_method = None
            scored_sources: List[Dict[str, Any]] = []
            for cand in candidates:
                snippet = cand.get("snippet") or ""
                scores: Dict[str, Optional[float]] = {
                    "ngram_overlap": None, "semantic": None, "llm": None,
                }

                # (a) Deterministic n-gram/shingle overlap -- always available,
                # no LLM/API key needed. Catches verbatim/near-verbatim copying.
                if snippet:
                    jac = _jaccard(para_shingles, _shingles(snippet))
                    scores["ngram_overlap"] = round(jac * 100, 1)

                # (b) Semantic embedding cosine similarity -- reuses the
                # detector's own encoder (agents.detector_agent.DetectorAgent),
                # no separate model/dependency and no API key.
                if snippet and semantic_enabled and para_embedding is not None:
                    snip_emb = detector.embed_text(snippet)
                    cos = _cosine(para_embedding, snip_emb)
                    if cos is not None:
                        # cosine in [-1,1] -> map to a 0-100 "similarity" scale.
                        scores["semantic"] = round(max(0.0, cos) * 100, 1)

                # (c) LLM judgment -- most nuanced, costs a network call + key.
                if llm_enabled and snippet:
                    scores["llm"] = self._score_similarity(para, snippet, cand.get("title", ""))

                available = {k: v for k, v in scores.items() if v is not None}
                cand_best = max(available.values()) if available else None
                cand_best_method = (
                    max(available, key=lambda k: available[k]) if available else None
                )
                if cand_best is not None and cand_best > best_sim:
                    best_sim = cand_best
                    best_method = cand_best_method

                scored_sources.append({
                    "source": cand.get("source"),
                    "title": cand.get("title"),
                    "link": cand.get("link"),
                    "snippet": snippet[:300],
                    "similarity": cand_best,     # backward-compatible single score
                    "similarity_by_method": scores,
                })

            any_scoring_available = bool(scored_sources) and any(
                s.get("similarity") is not None for s in scored_sources
            )
            if any_scoring_available:
                best_similarities.append(best_sim)

            quoted_and_cited = _looks_quoted_and_cited(para, references)
            raw_flag = any_scoring_available and best_sim >= _SIMILARITY_THRESHOLD

            checks.append({
                "paragraph_index": i,
                "section": section,
                "key_phrase": phrase,
                # Full text (not just a preview) so the UI can align this
                # paragraph with the AI-detection heatmap by content match
                # rather than by index -- the two agents select different
                # (overlapping but not identical) paragraph subsets, so index
                # alone is not a reliable join key for a combined overlay.
                "paragraph_text": para,
                "candidate_count": len(scored_sources),
                "best_similarity": best_sim if any_scoring_available else None,
                "best_method": best_method,
                "quoted_and_cited": quoted_and_cited,
                "flagged": bool(raw_flag and not quoted_and_cited),
                "downgraded_attributed_quote": bool(raw_flag and quoted_and_cited),
                "sources": scored_sources,
            })

        score = (
            round(sum(best_similarities) / len(best_similarities), 1)
            if best_similarities else None
        )
        flagged = [c for c in checks if c["flagged"]]
        downgraded = [c for c in checks if c["downgraded_attributed_quote"]]

        findings = self._compose_findings(
            checks, flagged, downgraded, score, web_enabled, llm_enabled, semantic_enabled
        )
        status = self._status(score, flagged)

        metadata = {
            "plagiarism_score": score,  # 0-100 avg best-match similarity, or None
            "checked_paragraphs": len(checks),
            "flagged_paragraph_count": len(flagged),
            "downgraded_attributed_quote_count": len(downgraded),
            "web_search_enabled": web_enabled,
            "similarity_scoring_enabled": bool(llm_enabled or semantic_enabled),
            "ngram_overlap_enabled": True,
            "semantic_similarity_enabled": semantic_enabled,
            "llm_similarity_enabled": llm_enabled,
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
        downgraded: List[Dict[str, Any]],
        score: Optional[float],
        web_enabled: bool,
        llm_enabled: bool,
        semantic_enabled: bool,
    ) -> List[str]:
        findings: List[str] = []
        if score is not None:
            findings.append(f"Average best-match similarity across checked paragraphs: {score}%.")
        for c in flagged:
            top = max(c["sources"], key=lambda s: (s.get("similarity") or 0), default=None)
            src = f" (source: {top.get('link')})" if top and top.get("link") else ""
            method = f" via {c.get('best_method')}" if c.get("best_method") else ""
            findings.append(
                f"Paragraph {c['paragraph_index']} in '{c['section']}' closely matches "
                f"an existing source ({c['best_similarity']}%{method})" + src + "."
            )
        for c in downgraded:
            findings.append(
                f"Paragraph {c['paragraph_index']} in '{c['section']}' overlaps a source "
                f"({c['best_similarity']}%) but appears to be a quoted, properly cited "
                f"passage - not counted as plagiarism (cross-agent dedupe)."
            )
        if not web_enabled:
            findings.append(
                "Web search disabled (no SERPER_API_KEY); used the open-access scholarly "
                "path only. Add a Serper key for broader web coverage."
            )
        if not semantic_enabled:
            findings.append(
                "Semantic-embedding similarity unavailable (detector model not loaded); "
                "using deterministic n-gram overlap only for those candidates."
            )
        if not llm_enabled:
            findings.append(
                "LLM similarity judgment skipped (no Gemini API key); relying on "
                "n-gram overlap" + (" + semantic embeddings" if semantic_enabled else "") + "."
            )
        if not flagged and (web_enabled or llm_enabled or semantic_enabled):
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
