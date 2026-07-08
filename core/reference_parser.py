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
import google.generativeai as genai
from pydantic import ValidationError

# Adding parent dir to path to allow absolute imports when running as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.reference import Reference

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
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            # Keep the model name consistent with services/gemini.py.
            model_name = os.getenv("PAPERGUARD_GEMINI_MODEL", "gemini-2.5-flash")
            self.model = genai.GenerativeModel(model_name)
        else:
            self.model = None

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
            
        if self.model:
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
        
        response = self.model.generate_content(prompt)
        
        # Clean markdown formatting if present
        raw_json = response.text
        if raw_json.startswith("```json"):
            raw_json = raw_json[7:-3]
        elif raw_json.startswith("```"):
            raw_json = raw_json[3:-3]
            
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
        Basic regex/heuristic parsing for references when LLM is unavailable.
        Assumes each paragraph/newline-separated block is a distinct reference.
        """
        references = []
        # Split by double newline or numbered items like "[1]", "1."
        blocks = re.split(r'\n\n|(?=\n\[\d+\])|\n(?=\d+\.\s)', text)
        
        for block in blocks:
            block = block.strip()
            if not block:
                continue
                
            # Very basic heuristic extraction
            year_match = re.search(r'\b(19\d\d|20\d\d)\b', block)
            year = int(year_match.group(1)) if year_match else None
            
            doi_match = re.search(r'\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b', block, re.IGNORECASE)
            doi = doi_match.group(1) if doi_match else None
            
            # Create a placeholder reference
            ref = Reference(
                authors=[],
                title="Unknown Title (Heuristic Parse)",
                year=year,
                doi=doi,
                raw_text=block
            )
            references.append(ref)
            
        return references

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
