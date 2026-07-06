"""
PDF Parser using pymupdf4llm for layout-aware extraction.
Converts academic papers (including two-column formats) into clean Markdown.
"""

import sys
import argparse
from pathlib import Path
import pymupdf4llm

class PDFParser:
    """
    Handles extracting text from PDF files into Markdown format.
    """
    
    def __init__(self):
        pass

    def parse(self, pdf_path: str | Path) -> str:
        """
        Parses a PDF file and returns its content as Markdown.
        
        Args:
            pdf_path (str | Path): Path to the PDF file.
            
        Returns:
            str: The extracted text formatted in Markdown.
            
        Raises:
            FileNotFoundError: If the PDF file does not exist.
            Exception: For other parsing errors.
        """
        path_obj = Path(pdf_path)
        if not path_obj.exists() or not path_obj.is_file():
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
            
        try:
            # pymupdf4llm handles two-column layouts and reading order automatically
            md_text = pymupdf4llm.to_markdown(str(path_obj))
            return md_text
        except Exception as e:
            raise Exception(f"Failed to parse PDF {pdf_path}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Extract Markdown from PDF academic papers.")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file to parse")
    parser.add_argument("-o", "--output", type=str, help="Optional output path for the Markdown file", default=None)
    
    args = parser.parse_args()
    
    pdf_parser = PDFParser()
    try:
        print(f"Parsing {args.pdf_path}...")
        md_content = pdf_parser.parse(args.pdf_path)
        
        if args.output:
            out_path = Path(args.output)
            out_path.write_text(md_content, encoding='utf-8')
            print(f"Success! Markdown saved to {out_path}")
        else:
            print("\n--- Extracted Content ---\n")
            print(md_content)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
