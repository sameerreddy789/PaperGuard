"""
Reference Parser.
Extracts structured references from a raw References section.
Uses Gemini LLM for structured extraction with a fallback heuristic parser.
"""

import os
import re
import sys
import json
import argparse
from typing import List, Optional
from pydantic import ValidationError

# Adding parent dir to path to allow absolute imports when running as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.reference import Reference
# Provider-agnostic (Gemini or Qwen/DashScope, see services/llm.py) rather than
# hardcoding Gemini, so reference extraction also runs on the Alibaba backend.
from services import llm as gemini_service

class ReferenceParser:
    """
    Parses the raw text of a References section into structured Reference objects.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initializes the parser. If api_key is provided, configures the Gemini client.
        
        Args:
            api_key: Optional Gemini API Key.
        """
        # LLM calls are routed through services.llm, which picks Gemini or
        # Qwen/DashScope based on which key is configured (see services/llm.py).
        # An explicit api_key argument is assumed to be a Gemini key (the
        # historical default for this constructor) and is exposed via env so
        # the shared client picks it up.
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
        self.llm_enabled = bool(self.api_key)

    def parse_references(self, references_text: str) -> List[Reference]:
        """
        Parses raw references text into a list of Reference models.
        Attempts LLM extraction first if available, falls back to heuristic.
        
        Args:
            references_text: The raw text of the References section.
            
        Returns:
            List of Reference objects.
        """
        if not references_text.strip():
            return []
            
        if self.llm_enabled:
            try:
                return self._parse_with_llm(references_text)
            except Exception as e:
                print(f"Warning: LLM extraction failed ({e}). Falling back to heuristics.", file=sys.stderr)
                
        return self._parse_heuristically(references_text)

    def _parse_with_llm(self, text: str) -> List[Reference]:
        """Uses Gemini to extract structured JSON matching the Reference schema."""
        prompt = f"""
        Extract the academic references from the following text.
        Return ONLY a JSON list of objects. Each object MUST have these exact keys:
        - authors (list of strings)
        - title (string)
        - year (integer or null)
        - journal (string or null)
        - doi (string or null)
        - volume (string or null)
        - pages (string or null)
        - raw_text (string, the original unedited reference text)
        
        References text:
        {text}
        """
        
        raw_json = gemini_service.call_llm(prompt, response_format="json")
        if not raw_json:
            raise RuntimeError("LLM returned no content for reference extraction.")

        # Clean markdown fences if present
        raw_json = raw_json.strip()
        if raw_json.startswith("```json"):
            raw_json = raw_json[7:]
        if raw_json.startswith("```"):
            raw_json = raw_json[3:]
        if raw_json.endswith("```"):
            raw_json = raw_json[:-3]

        parsed_data = json.loads(raw_json.strip())
        
        references = []
        for item in parsed_data:
            try:
                references.append(Reference(**item))
            except ValidationError as e:
                print(f"Skipping malformed reference: {e}", file=sys.stderr)
                
        return references

    def _parse_heuristically(self, text: str) -> List[Reference]:
        """
        Regex/heuristic parsing for references when the LLM is unavailable.

        Extracts year, DOI, and a best-effort title + authors from the common
        citation styles (APA "Authors (YEAR). Title. Venue", IEEE quoted titles,
        and numeric "Authors. Title. Venue"). A real title/author matters: it
        lets the Citation Agent do CrossRef/OpenAlex lookups even without
        an LLM key.
        """
        references = []
        # Split by blank lines or numbered items like "[1]", "1."
        blocks = re.split(r'\n\s*\n|(?=\n\s*\[\d+\])|\n(?=\d+\.\s)', text)

        for block in blocks:
            block = " ".join(block.split())  # collapse internal newlines/whitespace
            if not block or len(block) < 10:
                continue

            year_match = re.search(r'\b(19\d\d|20\d\d)\b', block)
            year = int(year_match.group(1)) if year_match else None

            doi_match = re.search(r'\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b', block, re.IGNORECASE)
            doi = doi_match.group(1).rstrip('.') if doi_match else None

            title, authors = self._extract_title_authors(block, year)

            ref = Reference(
                authors=authors,
                title=title or "Unknown Title (Heuristic Parse)",
                year=year,
                doi=doi,
                raw_text=block,
            )
            references.append(ref)

        return references

    @staticmethod
    def _extract_title_authors(block: str, year: Optional[int]) -> tuple:
        """Best-effort (title, authors_list) from a single reference string."""
        # Drop a leading citation marker: "[12] " or "12. "
        clean = re.sub(r'^\s*\[?\d{1,3}\]?[.)]?\s*', '', block).strip()

        # 1. IEEE style: the title is usually in quotes.
        q = re.search(r'["“\u201c]([^"”\u201d]{6,})["”\u201d]', clean)
        if q:
            title = q.group(1).strip().rstrip('.,')
            authors = clean[:q.start()].strip().rstrip(',')
            return title, ([authors] if authors else [])

        # 2. APA style: "Authors (YEAR). Title. Venue"
        apa = re.search(r'\((?:19|20)\d\d[a-z]?\)\.?\s*(.+?)(?:\.\s|\Z)', clean)
        if apa:
            title = apa.group(1).strip().rstrip('.')
            authors_seg = clean[:apa.start()].strip().rstrip('(').strip().rstrip(',.')
            return title, ([authors_seg] if authors_seg else [])

        # 3. Numeric/other: "Authors. Title. Venue" -> take the 2nd sentence chunk.
        chunks = [c.strip() for c in re.split(r'\.\s+', clean) if c.strip()]
        if len(chunks) >= 2:
            # If the first chunk looks like authors (has commas/initials), title is next.
            return chunks[1].rstrip('.'), [chunks[0].rstrip(',.')]
        if chunks:
            return chunks[0].rstrip('.'), []
        return None, []

def main():
    parser = argparse.ArgumentParser(description="Parse references from raw text using Gemini or heuristics.")
    parser.add_argument("input_file", type=str, help="Path to text file containing references")
    args = parser.parse_args()
    
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        ref_parser = ReferenceParser()
        refs = ref_parser.parse_references(content)
        
        print(json.dumps([ref.model_dump() for ref in refs], indent=2))
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
