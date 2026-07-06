import os
import requests
from typing import Dict, Any, Optional
from services import cache

SEMANTIC_SCHOLAR_API_BASE = "https://api.semanticscholar.org/graph/v1"

def get_headers() -> Dict[str, str]:
    headers = {}
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    return headers

def search_paper_by_title(title: str) -> Optional[Dict[str, Any]]:
    """
    Search for a paper by title and return its metadata, including the abstract.
    """
    if not title:
        return None
        
    namespace = "semantic_scholar_search"
    cached_result = cache.get(title, namespace=namespace)
    if cached_result is not None:
        return cached_result
        
    url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"
    params = {
        "query": title,
        "limit": 1,
        "fields": "paperId,title,abstract,authors,year,url,externalIds"
    }
    
    try:
        response = requests.get(url, params=params, headers=get_headers(), timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("data") and len(data["data"]) > 0:
                result = data["data"][0]
                cache.set(title, result, namespace=namespace)
                return result
            else:
                # Cache negative result
                cache.set(title, {"error": "not_found"}, namespace=namespace)
                return None
        else:
            print(f"Semantic Scholar API error: {response.status_code} - {response.text}")
    except requests.RequestException as e:
        print(f"Semantic Scholar API request failed: {e}")
        
    return None

def get_paper_by_doi(doi: str) -> Optional[Dict[str, Any]]:
    """
    Fetch paper metadata (including abstract) using a DOI.
    """
    if not doi:
        return None
        
    namespace = "semantic_scholar_doi"
    cached_result = cache.get(doi, namespace=namespace)
    if cached_result is not None:
        return cached_result
        
    # Semantic Scholar allows fetching by DOI directly
    url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/DOI:{doi}"
    params = {
        "fields": "paperId,title,abstract,authors,year,url,externalIds"
    }
    
    try:
        response = requests.get(url, params=params, headers=get_headers(), timeout=10)
        if response.status_code == 200:
            data = response.json()
            cache.set(doi, data, namespace=namespace)
            return data
        elif response.status_code == 404:
            cache.set(doi, {"error": "not_found"}, namespace=namespace)
            return None
        else:
            print(f"Semantic Scholar API error for DOI {doi}: {response.status_code}")
    except requests.RequestException as e:
        print(f"Semantic Scholar API request failed for DOI {doi}: {e}")
        
    return None

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query = sys.argv[1]
        print(f"Searching for '{query}'...")
        if query.startswith("10."):
            result = get_paper_by_doi(query)
        else:
            result = search_paper_by_title(query)
            
        if result and "error" not in result:
            print(f"Found: {result.get('title')}")
            print(f"Abstract: {result.get('abstract', 'No abstract available')[:100]}...")
        else:
            print("Not found or error.")
