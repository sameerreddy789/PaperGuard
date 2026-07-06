"""
Text Chunker for academic papers.
Splits Markdown text into sections, paragraphs, and sentences.
"""

import re
from typing import Dict, List, Any

class TextChunker:
    """
    Chunks extracted academic paper Markdown into structured components.
    """
    
    # Common academic paper section headers
    # Matches markdown headers like ## Abstract, **1. Introduction**, or ALL CAPS
    SECTION_REGEX = re.compile(
        r'^(?:#{1,6}\s+)?(?:\*\*?)?(?:(?:\d+\.?\s*)?(Abstract|Introduction|Background|Related Work|'
        r'Method(?:ology)?|Methods|Materials and Methods|Results|Discussion|Conclusion(?:s)?|'
        r'References|Bibliography))(?:\*\*?)?\s*$',
        re.IGNORECASE | re.MULTILINE
    )
    
    # Simple sentence boundary detection (not perfect, but works for general use)
    SENTENCE_REGEX = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s')

    def __init__(self):
        pass

    def chunk(self, text: str) -> Dict[str, Any]:
        """
        Takes raw markdown text and chunks it into sections, paragraphs, and sentences.
        
        Args:
            text (str): The Markdown text of the paper.
            
        Returns:
            Dict[str, Any]: A dictionary where keys are section names and values are 
                            lists of paragraphs, which are lists of sentences.
        """
        if not text:
            return {}

        sections = self._split_into_sections(text)
        
        structured_data = {}
        
        for section_name, section_content in sections.items():
            paragraphs = self._split_into_paragraphs(section_content)
            
            structured_paragraphs = []
            for para in paragraphs:
                if not para.strip():
                    continue
                sentences = self._split_into_sentences(para)
                structured_paragraphs.append(sentences)
                
            structured_data[section_name] = structured_paragraphs
            
        return structured_data

    def _split_into_sections(self, text: str) -> Dict[str, str]:
        """Splits text into major academic sections."""
        lines = text.split('\n')
        sections = {}
        current_section = "Uncategorized"
        current_content = []
        
        for line in lines:
            match = self.SECTION_REGEX.match(line.strip())
            if match:
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                
                # Extract clean section name
                raw_name = match.group(1).title() if match.group(1) else line.strip()
                # Normalize common names
                if "Method" in raw_name:
                    current_section = "Methodology"
                elif "Conclusion" in raw_name:
                    current_section = "Conclusion"
                else:
                    current_section = raw_name
                
                current_content = []
            else:
                # Handle ALL CAPS headers (fallback)
                cleaned = line.strip().strip('*# ')
                if cleaned.replace(' ', '').isupper() and len(cleaned.split()) <= 4 and cleaned.replace(' ', '').isalpha():
                    if current_content:
                        sections[current_section] = '\n'.join(current_content)
                    current_section = cleaned.title()
                    current_content = []
                else:
                    current_content.append(line)
                    
        # Add the last section
        if current_content:
            sections[current_section] = '\n'.join(current_content)
            
        return sections
        
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Splits section text into paragraphs based on double newlines."""
        return [p.strip() for p in text.split('\n\n') if p.strip()]
        
    def _split_into_sentences(self, paragraph: str) -> List[str]:
        """Splits a paragraph into sentences."""
        sentences = self.SENTENCE_REGEX.split(paragraph)
        return [s.strip() for s in sentences if s.strip()]

def main():
    import argparse
    from pathlib import Path
    import json
    
    parser = argparse.ArgumentParser(description="Chunk Markdown text into sections and sentences.")
    parser.add_argument("markdown_file", type=str, help="Path to the Markdown file")
    args = parser.parse_args()
    
    chunker = TextChunker()
    try:
        content = Path(args.markdown_file).read_text(encoding='utf-8')
        result = chunker.chunk(content)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
