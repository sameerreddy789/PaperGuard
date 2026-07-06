import os
import requests
import urllib.parse
from typing import Dict, Any, Optional
from services import cache

CROSSREF_API_BASE = "https://api.crossref.org"

def get_headers() -> Dict[str, str]:
    email = os.getenv("CROSSREF_EMAIL", "paperguard@example.com")
    return {
        "User-Agent": f"PaperGuard/1.0 (mailto:{email})"
    }

def get_work_by_doi(doi: str) -> Optional[Dict[str, Any]]:
    """
    Fetch metadata for a specific DOI from CrossRef.
    Returns None if not found or on error.
    """
    if not doi:
        return None
        
    namespace = "crossref_doi"
    cached_result = cache.get(doi, namespace=namespace)
    if cached_result is not None:
        return cached_result
        
    encoded_doi = urllib.parse.quote(doi)
    url = f"{CROSSREF_API_BASE}/works/{encoded_doi}"
    
    try:
        response = requests.get(url, headers=get_headers(), timeout=10)
        if response.status_code == 200:
            data = response.json()
            # Cache the successful result
            cache.set(doi, data, namespace=namespace)
            return data
        elif response.status_code == 404:
            # Cache the fact that it doesn't exist to prevent repeated 404s
            cache.set(doi, {"error": "not_found", "status": 404}, namespace=namespace)
            return None
    except requests.RequestException as e:
        print(f"CrossRef API error for DOI {doi}: {e}")
    
    return None

def search_works_by_title(title: str, author: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Search for a work by title and optional author.
    """
    query = title
    if author:
        query += f" {author}"
        
    namespace = "crossref_search"
    cached_result = cache.get(query, namespace=namespace)
    if cached_result is not None:
        return cached_result
        
    url = f"{CROSSREF_API_BASE}/works"
    params = {
        "query.bibliographic": query,
        "rows": 3,  # Top 3 matches
        "select": "DOI,title,author,URL,published"
    }
    
    try:
        response = requests.get(url, params=params, headers=get_headers(), timeout=10)
        if response.status_code == 200:
            data = response.json()
            cache.set(query, data, namespace=namespace)
            return data
    except requests.RequestException as e:
        print(f"CrossRef API error for query '{query}': {e}")
        
    return None

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        doi = sys.argv[1]
        print(f"Fetching metadata for {doi}...")
        result = get_work_by_doi(doi)
        if result and "error" not in result:
            msg = result.get("message", {})
            title = msg.get("title", ["Unknown"])[0]
            print(f"Found: {title}")
        else:
            print("Not found or error.")
