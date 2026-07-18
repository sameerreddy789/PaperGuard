"""
OpenAlex service — replaces Semantic Scholar as PaperGuard's secondary
existence/abstract source (2026-07-16, see PROJECT_REPORT.md for why).

Why the swap: Semantic Scholar's public API returned persistent 429 (rate
limit) errors in this project's testing, and its own docs prioritize a
separate, approved API-key application over general public access.
OpenAlex (https://openalex.org) is a fully open, CC0-licensed index of
250M+ scholarly works that works with ZERO signup and ZERO API key for
basic queries (confirmed live against their own quickstart docs) -- a
genuine no-friction replacement, not just "another key to wait for."
An optional ``mailto`` query param (like CrossRef's polite pool) gets
faster/more reliable service without requiring a key at all.

Return shapes are intentionally kept compatible with the old
``services.semantic_scholar`` module (``title``, ``abstract``, ``authors``,
``year``, ``url``, ``externalIds": {"DOI": ...}``, and the same
``{"error": "not_found"}`` sentinel for negative results) so callers
(``agents/citation_agent.py``, ``agents/plagiarism_agent.py``) needed only a
symbol swap, not a rewrite.
"""

import os
import requests
from typing import Dict, Any, List, Optional
from services import cache

OPENALEX_API_BASE = "https://api.openalex.org"


def _polite_params() -> Dict[str, str]:
    """
    OpenAlex's "polite pool" -- like CrossRef's -- gives faster, more reliable
    service to requests that self-identify with a contact email. No key, no
    approval process; reuses the same CROSSREF_EMAIL env var since this
    project already asks the user for one contact email, not two.
    """
    email = os.getenv("CROSSREF_EMAIL") or os.getenv("OPENALEX_EMAIL")
    return {"mailto": email} if email else {}


def _reconstruct_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    """
    OpenAlex stores abstracts as an ``{word: [positions]}`` inverted index
    (a licensing/legal constraint on their end, not a design choice we can
    avoid) rather than plain text. Rebuild the plain-text abstract from it.
    """
    if not inverted_index:
        return ""
    positioned: List[tuple] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            positioned.append((pos, word))
    positioned.sort(key=lambda p: p[0])
    return " ".join(word for _, word in positioned)


def _normalize_work(work: Dict[str, Any]) -> Dict[str, Any]:
    """Map an OpenAlex work object to the shape callers already expect."""
    doi = (work.get("doi") or "").replace("https://doi.org/", "") or None
    authors = [
        (a.get("author") or {}).get("display_name", "")
        for a in (work.get("authorships") or [])
        if (a.get("author") or {}).get("display_name")
    ]
    return {
        "title": work.get("title") or work.get("display_name"),
        "abstract": _reconstruct_abstract(work.get("abstract_inverted_index")),
        "authors": authors,
        "year": work.get("publication_year"),
        "url": work.get("doi") or (work.get("primary_location") or {}).get("landing_page_url"),
        "externalIds": {"DOI": doi} if doi else {},
    }


def search_paper_by_title(title: str) -> Optional[Dict[str, Any]]:
    """
    Search for a work by title and return its metadata, including a
    reconstructed abstract. Same public contract as the retired
    ``services.semantic_scholar.search_paper_by_title``.
    """
    if not title:
        return None

    namespace = "openalex_search"
    cached_result = cache.get(title, namespace=namespace)
    if cached_result is not None:
        return cached_result

    url = f"{OPENALEX_API_BASE}/works"
    params = {"search": title, "per_page": 1, **_polite_params()}

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            results = data.get("results") or []
            if results:
                result = _normalize_work(results[0])
                cache.set(title, result, namespace=namespace)
                return result
            cache.set(title, {"error": "not_found"}, namespace=namespace)
            return None
        else:
            print(f"OpenAlex API error: {response.status_code} - {response.text[:200]}")
    except requests.RequestException as e:
        print(f"OpenAlex API request failed: {e}")

    return None


def get_paper_by_doi(doi: str) -> Optional[Dict[str, Any]]:
    """
    Fetch work metadata (including abstract) by DOI. Same public contract as
    the retired ``services.semantic_scholar.get_paper_by_doi``.
    """
    if not doi:
        return None

    namespace = "openalex_doi"
    cached_result = cache.get(doi, namespace=namespace)
    if cached_result is not None:
        return cached_result

    clean_doi = doi.strip().replace("https://doi.org/", "")
    url = f"{OPENALEX_API_BASE}/works/doi:{clean_doi}"

    try:
        response = requests.get(url, params=_polite_params(), timeout=10)
        if response.status_code == 200:
            result = _normalize_work(response.json())
            cache.set(doi, result, namespace=namespace)
            return result
        elif response.status_code == 404:
            cache.set(doi, {"error": "not_found"}, namespace=namespace)
            return None
        else:
            print(f"OpenAlex API error for DOI {doi}: {response.status_code}")
    except requests.RequestException as e:
        print(f"OpenAlex API request failed for DOI {doi}: {e}")

    return None


if __name__ == "__main__":
    import sys
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except ImportError:
        pass

    if len(sys.argv) > 1:
        query = sys.argv[1]
        print(f"Searching for '{query}'...")
        if query.startswith("10."):
            result = get_paper_by_doi(query)
        else:
            result = search_paper_by_title(query)

        if result and "error" not in result:
            print(f"Found: {result.get('title')}")
            print(f"Abstract: {(result.get('abstract') or 'No abstract available')[:200]}...")
        else:
            print("Not found or error.")
