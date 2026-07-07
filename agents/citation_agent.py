"""
Citation Verification Agent  (PaperGuard's killer feature).

For every reference in a paper this agent answers two questions:

1. Does the cited work actually EXIST?         -> CrossRef (unlimited, no key)
                                                  + Semantic Scholar fallback
2. Does the cited work SUPPORT the claim it     -> Semantic Scholar abstract
   is used for in the paper?                       + Gemini claim verification

Each reference is placed into one of four tiers:

    VERIFIED           exists AND its abstract supports the in-text claim
    PARTIALLY_VERIFIED exists, abstract retrieved, but claim not confirmed
                       (vague/unrelated abstract, or LLM unavailable)
    EXISTENCE_ONLY     exists, but no abstract/content available to check claim
    NOT_FOUND          no DOI match and no title match -> possibly fabricated

Pattern signal: if >50% of citations are PARTIALLY_VERIFIED or EXISTENCE_ONLY,
the paper is flagged for heavy reliance on unverifiable sources.

The agent works without any API keys (CrossRef needs none); claim verification
and abstract-based checks simply downgrade gracefully when keys are absent.

CLI:  python -m agents.citation_agent path/to/paper.(pdf|md|txt)
"""

from __future__ import annotations

import re
import difflib
from typing import Any, Dict, List, Optional, Tuple

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

# Verification tiers
VERIFIED = "VERIFIED"
PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"
EXISTENCE_ONLY = "EXISTENCE_ONLY"
NOT_FOUND = "NOT_FOUND"

# Claim verdicts
SUPPORTS = "SUPPORTS"
CONTRADICTS = "CONTRADICTS"
UNRELATED = "UNRELATED"
CANNOT_DETERMINE = "CANNOT_DETERMINE"

# Fuzzy title match threshold (0-1)
_TITLE_MATCH_THRESHOLD = 0.75
# Cap on LLM claim checks per paper (protects free-tier rate limits)
_MAX_CLAIM_CHECKS = 30


def _normalize_title(title: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace for fuzzy comparison."""
    title = re.sub(r"[^a-z0-9\s]", " ", (title or "").lower())
    return re.sub(r"\s+", " ", title).strip()


def _title_similarity(a: str, b: str) -> float:
    na, nb = _normalize_title(a), _normalize_title(b)
    if not na or not nb:
        return 0.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


def _clean_service_result(result: Any) -> Optional[Dict[str, Any]]:
    """
    Normalize a services.* response into a usable dict or None.

    The service wrappers may return ``None``, a valid payload, or a cached
    ``{"error": "not_found", ...}`` sentinel; the latter two must both be
    treated as "no data".
    """
    if not result or not isinstance(result, dict):
        return None
    if "error" in result:
        return None
    return result


def _first_author_lastname(ref: Reference) -> Optional[str]:
    """Best-effort extraction of the first author's surname."""
    if not ref.authors:
        return None
    first = ref.authors[0].strip()
    if not first:
        return None
    # Handle "Smith, J." and "John Smith" and "J. Smith".
    if "," in first:
        return first.split(",")[0].strip()
    parts = first.split()
    return parts[-1].strip() if parts else None


def _reference_number(ref: Reference) -> Optional[int]:
    """Extract a leading citation number from the raw reference text (e.g. [12] or 12.)."""
    m = re.match(r"^\s*\[?(\d{1,3})\]?[.)]?\s+\S", ref.raw_text or "")
    if m:
        return int(m.group(1))
    return None


def _expand_bracket_numbers(inner: str) -> List[int]:
    """Expand the contents of a bracket like '3, 5-7, 9' into [3,5,6,7,9]."""
    nums: List[int] = []
    for part in re.split(r"[,;]", inner):
        part = part.strip()
        rng = re.match(r"^(\d{1,3})\s*[-–]\s*(\d{1,3})$", part)
        if rng:
            lo, hi = int(rng.group(1)), int(rng.group(2))
            if 0 < hi - lo < 100:
                nums.extend(range(lo, hi + 1))
        elif part.isdigit():
            nums.append(int(part))
    return nums


class CitationAgent(BaseAgent):
    """Verifies existence and claim-support for every reference."""

    name = "CitationAgent"
    needs_references = True

    def __init__(self, max_claim_checks: int = _MAX_CLAIM_CHECKS):
        self.max_claim_checks = max_claim_checks
        # Lazy service handles (import here so missing gemini never breaks import)
        from services import crossref, semantic_scholar

        self._crossref = crossref
        self._s2 = semantic_scholar
        self._gemini = get_llm()

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def run(
        self,
        text: str,
        references: Optional[List[Reference]] = None,
    ) -> "Any":
        if references is None:
            references = extract_references(text)

        # Resolve LLM availability at run time (after .env is loaded by run_cli).
        self._llm_enabled = self._gemini is not None and llm_available()

        body_text, _refs_text = split_body_and_references(text)
        paragraphs = [p for _section, p in iter_paragraphs(body_text)] if body_text else []

        if not references:
            return self._result(
                status="warning",
                findings=["No references were found or extracted from the paper."],
                metadata={"reference_count": 0, "results": []},
            )

        claim_checks_done = 0
        results: List[Dict[str, Any]] = []

        for idx, ref in enumerate(references, start=1):
            existence = self._check_existence(ref)
            entry: Dict[str, Any] = {
                "index": idx,
                "title": ref.title,
                "authors": ref.authors,
                "year": ref.year,
                "doi": ref.doi or existence.get("resolved_doi"),
                "exists": existence["exists"],
                "existence_source": existence["source"],
                "tier": NOT_FOUND,
                "claim_verdict": None,
                "claim_reasoning": None,
                "abstract_available": False,
                "in_text_context_found": False,
            }

            if not existence["exists"]:
                entry["tier"] = NOT_FOUND
                results.append(entry)
                continue

            # Reference exists -> attempt claim verification.
            abstract = self._get_abstract(ref, existence.get("resolved_doi"))
            entry["abstract_available"] = bool(abstract)

            context = self._find_in_text_context(ref, idx, paragraphs)
            entry["in_text_context_found"] = bool(context)

            if not abstract:
                entry["tier"] = EXISTENCE_ONLY
                results.append(entry)
                continue

            if context and self._llm_enabled and claim_checks_done < self.max_claim_checks:
                verdict, reasoning = self._verify_claim(context, abstract, ref)
                claim_checks_done += 1
                entry["claim_verdict"] = verdict
                entry["claim_reasoning"] = reasoning
                if verdict == SUPPORTS:
                    entry["tier"] = VERIFIED
                elif verdict == CONTRADICTS:
                    entry["tier"] = PARTIALLY_VERIFIED  # exists but misused -> flagged below
                else:  # UNRELATED / CANNOT_DETERMINE
                    entry["tier"] = PARTIALLY_VERIFIED
            else:
                # Abstract exists but we can't run a claim check (no context / no LLM).
                entry["tier"] = PARTIALLY_VERIFIED

            results.append(entry)

        return self._build_result(results, claim_checks_done)

    # ------------------------------------------------------------------ #
    # Existence checking
    # ------------------------------------------------------------------ #
    def _check_existence(self, ref: Reference) -> Dict[str, Any]:
        """Return {'exists': bool, 'source': str, 'resolved_doi': Optional[str]}."""
        # 1. DOI -> CrossRef (authoritative existence check, no key needed).
        if ref.doi:
            try:
                data = _clean_service_result(self._crossref.get_work_by_doi(ref.doi))
            except Exception:
                data = None
            if data and data.get("message"):
                return {"exists": True, "source": "crossref_doi", "resolved_doi": ref.doi}
            # DOI present but did not resolve -> strong fabrication signal.
            return {"exists": False, "source": "crossref_doi_404", "resolved_doi": None}

        # 2. No DOI -> CrossRef bibliographic search by title (+ author).
        author = _first_author_lastname(ref)
        try:
            search = _clean_service_result(
                self._crossref.search_works_by_title(ref.title, author)
            )
        except Exception:
            search = None
        resolved = self._match_crossref_search(ref, search)
        if resolved:
            return {"exists": True, "source": "crossref_title", "resolved_doi": resolved}

        # 3. Semantic Scholar title search as a secondary source.
        try:
            s2 = _clean_service_result(self._s2.search_paper_by_title(ref.title))
        except Exception:
            s2 = None
        if s2 and _title_similarity(ref.title, s2.get("title", "")) >= _TITLE_MATCH_THRESHOLD:
            resolved_doi = (s2.get("externalIds") or {}).get("DOI")
            return {"exists": True, "source": "semantic_scholar_title", "resolved_doi": resolved_doi}

        return {"exists": False, "source": "no_match", "resolved_doi": None}

    def _match_crossref_search(
        self, ref: Reference, search: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Return a matching DOI from a CrossRef title-search response, if any."""
        if not search:
            return None
        items = (search.get("message") or {}).get("items") or []
        for item in items:
            item_title = (item.get("title") or [""])[0]
            if _title_similarity(ref.title, item_title) >= _TITLE_MATCH_THRESHOLD:
                return item.get("DOI")
        return None

    # ------------------------------------------------------------------ #
    # Abstract retrieval
    # ------------------------------------------------------------------ #
    def _get_abstract(self, ref: Reference, resolved_doi: Optional[str]) -> Optional[str]:
        doi = ref.doi or resolved_doi
        data = None
        if doi:
            try:
                data = _clean_service_result(self._s2.get_paper_by_doi(doi))
            except Exception:
                data = None
        if not data:
            try:
                data = _clean_service_result(self._s2.search_paper_by_title(ref.title))
            except Exception:
                data = None
        if not data:
            return None
        abstract = (data.get("abstract") or "").strip()
        # Guard against too-short/uninformative abstracts.
        if len(abstract) < 40:
            return None
        return abstract

    # ------------------------------------------------------------------ #
    # In-text citation context
    # ------------------------------------------------------------------ #
    def _find_in_text_context(
        self, ref: Reference, idx: int, paragraphs: List[str]
    ) -> Optional[str]:
        """Locate the paragraph that cites this reference (numeric or author-year)."""
        if not paragraphs:
            return None

        ref_num = _reference_number(ref) or idx

        # Numeric citation style: [12], [3, 5-7], (12)
        for para in paragraphs:
            for m in re.finditer(r"[\[(]([\d,;\s\-–]+)[\])]", para):
                if ref_num in _expand_bracket_numbers(m.group(1)):
                    return para

        # Author-year style: "Smith (2020)" / "(Smith, 2020)" / "Smith et al., 2020"
        lastname = _first_author_lastname(ref)
        if lastname and ref.year:
            pattern = re.compile(
                rf"{re.escape(lastname)}[^.]{{0,40}}\b{ref.year}\b",
                re.IGNORECASE,
            )
            for para in paragraphs:
                if pattern.search(para):
                    return para
        return None

    # ------------------------------------------------------------------ #
    # LLM claim verification
    # ------------------------------------------------------------------ #
    def _verify_claim(
        self, context: str, abstract: str, ref: Reference
    ) -> Tuple[str, Optional[str]]:
        """Ask Gemini whether the abstract supports how the citation is used."""
        label = ref.title or (_first_author_lastname(ref) or "the cited work")
        prompt = (
            "You are verifying an academic citation.\n\n"
            "PARAGRAPH FROM THE SUBMITTED PAPER (contains the citation):\n"
            f"\"\"\"\n{context.strip()}\n\"\"\"\n\n"
            f"ABSTRACT OF THE CITED WORK ({label}):\n"
            f"\"\"\"\n{abstract.strip()}\n\"\"\"\n\n"
            "Question: Does the abstract of the cited work SUPPORT, CONTRADICT, or "
            "remain UNRELATED to the way the citation is used in the paragraph? "
            "If the abstract is too vague to tell, answer CANNOT_DETERMINE.\n\n"
            "Respond ONLY as JSON with keys: "
            '{"verdict": "SUPPORTS|CONTRADICTS|UNRELATED|CANNOT_DETERMINE", '
            '"reasoning": "one or two sentences"}'
        )
        try:
            data = self._gemini.call_llm_json(
                prompt,
                system_instruction=(
                    "You are a meticulous research-integrity assistant. "
                    "Base your judgment only on the provided texts."
                ),
            )
        except Exception:
            data = None

        if not data or "verdict" not in data:
            return CANNOT_DETERMINE, None
        verdict = str(data.get("verdict", "")).upper().strip()
        if verdict not in {SUPPORTS, CONTRADICTS, UNRELATED, CANNOT_DETERMINE}:
            verdict = CANNOT_DETERMINE
        return verdict, data.get("reasoning")

    # ------------------------------------------------------------------ #
    # Aggregation / report building
    # ------------------------------------------------------------------ #
    def _build_result(self, results: List[Dict[str, Any]], claim_checks_done: int) -> "Any":
        total = len(results)
        tier_counts = {VERIFIED: 0, PARTIALLY_VERIFIED: 0, EXISTENCE_ONLY: 0, NOT_FOUND: 0}
        for r in results:
            tier_counts[r["tier"]] += 1

        not_found = tier_counts[NOT_FOUND]
        unverifiable = tier_counts[PARTIALLY_VERIFIED] + tier_counts[EXISTENCE_ONLY]
        contradicts = [r for r in results if r["claim_verdict"] == CONTRADICTS]

        # Citation health: existence-weighted + claim-support bonus, 0-100.
        exists_count = total - not_found
        health = 0.0
        if total:
            health = (
                0.6 * (exists_count / total)
                + 0.4 * (tier_counts[VERIFIED] / total)
            ) * 100
        health = round(health, 1)

        findings: List[str] = []
        if not_found:
            findings.append(
                f"{not_found} reference(s) could not be found in CrossRef or "
                f"Semantic Scholar - possibly fabricated."
            )
            for r in results:
                if r["tier"] == NOT_FOUND:
                    findings.append(f"  - Ref #{r['index']}: NOT FOUND - \"{r['title']}\"")
        if contradicts:
            for r in contradicts:
                findings.append(
                    f"Ref #{r['index']} may not support the claim it is cited for "
                    f"(abstract appears to contradict usage)."
                )
        if tier_counts[VERIFIED]:
            findings.append(f"{tier_counts[VERIFIED]} reference(s) fully verified (exist + claim supported).")

        # Pattern signal: heavy reliance on unverifiable sources.
        pattern_flag = False
        if total and (unverifiable / total) > 0.5:
            pattern_flag = True
            findings.append(
                f"PATTERN: {unverifiable}/{total} citations are only partially "
                f"verifiable - heavy reliance on unverifiable sources."
            )

        if not findings:
            findings.append("All references processed; no fabrication signals detected.")

        # Status
        if not_found or contradicts:
            status = "failed"
        elif pattern_flag or unverifiable:
            status = "warning"
        else:
            status = "passed"

        llm_used = getattr(self, "_llm_enabled", False)
        metadata = {
            "reference_count": total,
            "citation_health_score": health,
            "tier_counts": tier_counts,
            "not_found_count": not_found,
            "unverifiable_count": unverifiable,
            "contradiction_count": len(contradicts),
            "pattern_flag_over_50pct_unverifiable": pattern_flag,
            "claim_checks_performed": claim_checks_done,
            "llm_claim_verification_enabled": bool(llm_used),
            "results": results,
        }
        return self._result(status=status, findings=findings, metadata=metadata)


if __name__ == "__main__":
    run_cli(CitationAgent(), "Verify citation existence and claim support.")
