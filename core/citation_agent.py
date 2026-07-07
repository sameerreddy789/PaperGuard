import time
from typing import Dict, Any, Optional
from services import crossref, semantic_scholar, gemini

class CitationVerificationAgent:
    def __init__(self):
        self.max_retries = 3
        self.retry_delay = 2  # seconds (exponential backoff base)

    def verify_citation(self, claim: str, citation_text: str) -> Dict[str, Any]:
        """
        Runs the 4-tier citation verification pipeline.
        
        Returns:
            {
                "status": "VERIFIED" | "PARTIALLY_VERIFIED" | "EXISTENCE_ONLY" | "NOT_FOUND",
                "reasoning": str,
                "metadata": dict
            }
        """
        print(f"\n🔍 Verifying Citation: '{citation_text[:50]}...'")
        
        # STEP 1: Find DOI via CrossRef
        print("   [1/3] Searching CrossRef for DOI...")
        work_metadata = crossref.search_works_by_title(citation_text)
        
        if not work_metadata or "message" not in work_metadata or not work_metadata["message"]["items"]:
            return {
                "status": "NOT_FOUND",
                "reasoning": "Citation could not be found in the CrossRef academic database. Potential AI hallucination.",
                "metadata": {}
            }
            
        # Get the top match
        top_match = work_metadata["message"]["items"][0]
        doi = top_match.get("DOI")
        title = top_match.get("title", ["Unknown Title"])[0]
        
        if not doi:
            return {
                "status": "NOT_FOUND",
                "reasoning": f"Found a match ('{title}') but no valid DOI is associated with it.",
                "metadata": {"title": title}
            }

        print(f"   ✓ Found DOI: {doi} ({title})")
        
        # STEP 2: Fetch Abstract via Semantic Scholar (with Retry Logic)
        print("   [2/3] Fetching Abstract from Semantic Scholar...")
        paper_data = self._fetch_abstract_with_retry(doi)
        
        if not paper_data or not paper_data.get("abstract"):
            return {
                "status": "EXISTENCE_ONLY",
                "reasoning": f"The citation exists (DOI: {doi}), but no open-access abstract could be retrieved to verify the claim.",
                "metadata": {"doi": doi, "title": title}
            }
            
        real_abstract = paper_data["abstract"]
        print("   ✓ Abstract successfully retrieved!")

        # STEP 3: LLM Verification (Gemini 3.1 Flash Lite)
        print("   [3/3] Analyzing claim against abstract via Gemini LLM...")
        llm_result = self._check_claim_with_llm(claim, real_abstract)
        
        if not llm_result:
            return {
                "status": "EXISTENCE_ONLY",
                "reasoning": "Failed to connect to Gemini LLM for verification. The paper exists, but claim matching failed.",
                "metadata": {"doi": doi, "title": title}
            }

        is_verified = llm_result.get("is_supported", False)
        explanation = llm_result.get("explanation", "No explanation provided by LLM.")

        final_status = "VERIFIED" if is_verified else "PARTIALLY_VERIFIED"
        
        return {
            "status": final_status,
            "reasoning": explanation,
            "metadata": {
                "doi": doi,
                "title": title,
                "abstract_snippet": real_abstract[:200] + "..."
            }
        }

    def _fetch_abstract_with_retry(self, doi: str) -> Optional[Dict[str, Any]]:
        """Handles Semantic Scholar rate limits gracefully with exponential backoff."""
        for attempt in range(self.max_retries):
            result = semantic_scholar.get_paper_by_doi(doi)
            
            # If result is None, it might be a rate limit or a hard error (404 is cached as {"error": "not_found"})
            if result is not None:
                if "error" in result:
                    return None # e.g. 404 Not Found
                return result
                
            # If we get None, assume rate limit/network error and retry
            sleep_time = self.retry_delay * (2 ** attempt)
            print(f"     ⚠️ Rate limited or network error. Retrying in {sleep_time}s...")
            time.sleep(sleep_time)
            
        return None

    def _check_claim_with_llm(self, claim: str, abstract: str) -> Optional[Dict[str, Any]]:
        """
        Constructs the Gemini prompt to determine if the abstract supports the claim.
        Requests a structured JSON response.
        """
        system_instruction = (
            "You are an expert academic peer-reviewer. Your job is to strictly verify if a student's "
            "claim is explicitly supported by the provided abstract of a research paper. "
            "You must return a JSON object with two keys: "
            "'is_supported' (boolean) and 'explanation' (string, max 1 sentence)."
        )
        
        prompt = f"""
        STUDENT CLAIM:
        "{claim}"
        
        REAL ABSTRACT OF THE CITED PAPER:
        "{abstract}"
        
        Does the abstract support the student's claim?
        """
        
        return gemini.call_llm_json(prompt=prompt, system_instruction=system_instruction)


if __name__ == "__main__":
    # Test execution
    agent = CitationVerificationAgent()
    
    test_claim = "Quantum entanglement entropy decreases exponentially over large spatial distances."
    test_citation = "Einstein, A., Podolsky, B., & Rosen, N. (1935). Can Quantum-Mechanical Description of Physical Reality Be Considered Complete?. Physical review, 47(10), 777."
    
    print("=========================================")
    print(f"TEST CLAIM: {test_claim}")
    print(f"TEST CITATION: {test_citation}")
    print("=========================================")
    
    result = agent.verify_citation(test_claim, test_citation)
    
    print("\n=========================================")
    print(f"FINAL STATUS: {result['status']}")
    print(f"REASONING:    {result['reasoning']}")
    print("=========================================")
