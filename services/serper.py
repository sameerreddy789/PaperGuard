import os
import requests
import json
from typing import Dict, Any, Optional, List
from services import cache

SERPER_API_URL = "https://google.serper.dev/search"

def get_headers() -> Dict[str, str]:
    api_key = os.getenv("SERPER_API_KEY")
    return {
        "X-API-KEY": api_key or "",
        "Content-Type": "application/json"
    }

def search_web(query: str, num_results: int = 5) -> Optional[List[Dict[str, Any]]]:
    """
    Search the web using Serper API for plagiarism checking.
    Returns a list of search results.
    """
    if not query:
        return None
        
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        print("Warning: SERPER_API_KEY not found in environment.")
        return None
        
    namespace = "serper_search"
    cached_result = cache.get(query, namespace=namespace)
    if cached_result is not None:
        return cached_result
        
    payload = json.dumps({
        "q": query,
        "num": num_results
    })
    
    try:
        response = requests.post(SERPER_API_URL, headers=get_headers(), data=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            results = data.get("organic", [])
            cache.set(query, results, namespace=namespace)
            return results
        else:
            print(f"Serper API error: {response.status_code} - {response.text}")
    except requests.RequestException as e:
        print(f"Serper API request failed: {e}")
        
    return None

if __name__ == "__main__":
    import sys
    # Load .env for testing
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
        
    if len(sys.argv) > 1:
        query = sys.argv[1]
        print(f"Searching web for '{query}'...")
        results = search_web(query)
        if results:
            for i, res in enumerate(results[:3]):
                print(f"{i+1}. {res.get('title')}\n   {res.get('link')}\n   {res.get('snippet')}\n")
        else:
            print("No results or error.")
