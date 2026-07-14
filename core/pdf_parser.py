"""
PDF Parser using pymupdf4llm for layout-aware extraction.
Converts academic papers (including two-column formats) into clean Markdown.

Also provides ``highlight_pdf`` -- a Turnitin-style annotated PDF: takes the
ORIGINAL PDF bytes plus a list of flagged spans (paragraph text + a color +
label) and returns a new PDF with those spans highlighted in place, using
PyMuPDF's own text search (``page.search_for``) rather than pymupdf4llm's
Markdown extraction. This deliberately avoids building a full text-to-bbox
coordinate pipeline: pymupdf4llm's Markdown conversion discards page/bbox
metadata, but ``fitz.Page.search_for`` re-locates the same text directly on
the rendered page, which is enough to place highlight annotations without
having to reconcile two different text-extraction paths.
"""

import io
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF (module name is historical; same package as pymupdf)
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

# --------------------------------------------------------------------------- #
# Annotated PDF export (Turnitin-style highlighted spans on the ORIGINAL PDF)
# --------------------------------------------------------------------------- #
# RGB (0-1 float) highlight colors, matching the app's heat/badge palette.
HIGHLIGHT_COLORS = {
    "ai": (0.98, 0.55, 0.55),          # red-ish  - likely-AI paragraph
    "plagiarism": (0.99, 0.75, 0.30),  # amber    - plagiarism match
    "patchwork": (0.70, 0.55, 0.95),   # purple   - stylometric outlier
}

# Cap how much of a long paragraph we try to search for verbatim. PDF text
# reflow (hyphenation, column breaks, ligatures) means a full 200-word
# paragraph will often NOT match exactly; a shorter, distinctive substring
# from the middle of the paragraph is far more likely to be found intact.
_SEARCH_SNIPPET_WORDS = 12


def _search_snippet(text: str, max_words: int = _SEARCH_SNIPPET_WORDS) -> str:
    """Pick a short, distinctive middle window of ``text`` for page.search_for."""
    words = (text or "").split()
    if len(words) <= max_words:
        return " ".join(words)
    start = max(0, (len(words) - max_words) // 2)
    return " ".join(words[start:start + max_words])


def highlight_pdf(
    pdf_bytes: bytes,
    spans: List[Dict[str, Any]],
) -> Tuple[bytes, Dict[str, int]]:
    """
    Return (annotated_pdf_bytes, stats) with highlight annotations placed on
    the original PDF for each flagged span.

    ``spans`` is a list of dicts: ``{"text": str, "kind": "ai"|"plagiarism"|
    "patchwork", "label": str}``. For each span, a short distinctive snippet
    is searched for on every page via ``fitz.Page.search_for`` (PyMuPDF's own
    text search over the ORIGINAL page content -- exact-substring, case-
    sensitive by default so we lower-case both sides first) and a highlight
    annotation with a tooltip (the ``label``) is added over every match found.

    Degrades gracefully: a span whose text cannot be located (common with
    hyphenation/column-reflow in dense two-column PDFs) is simply skipped and
    counted in ``stats['not_found']`` -- this is a best-effort visual aid, not
    a claim of pixel-perfect coverage.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    stats = {"requested": len(spans), "highlighted": 0, "not_found": 0}

    try:
        for span in spans:
            snippet = _search_snippet(span.get("text", ""))
            if not snippet.strip():
                stats["not_found"] += 1
                continue
            color = HIGHLIGHT_COLORS.get(span.get("kind", ""), (1.0, 1.0, 0.4))
            found_any = False
            for page in doc:
                rects = page.search_for(snippet, quads=False)
                for rect in rects:
                    annot = page.add_highlight_annot(rect)
                    annot.set_colors(stroke=color)
                    if span.get("label"):
                        annot.set_info(title="PaperGuard", content=span["label"])
                    annot.update()
                    found_any = True
            if found_any:
                stats["highlighted"] += 1
            else:
                stats["not_found"] += 1

        out = io.BytesIO()
        doc.save(out)
        return out.getvalue(), stats
    finally:
        doc.close()


def spans_from_report(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build the ``highlight_pdf`` span list from a PaperGuard report dict:
    likely-AI paragraphs, flagged plagiarism matches, and stylometric
    patchwork outliers. Kept here (rather than in app.py) so the CLI and the
    Streamlit UI produce identical annotated PDFs from the same report.
    """
    spans: List[Dict[str, Any]] = []
    for ar in report.get("agent_results", []):
        meta = ar.get("metadata") or {}
        if ar.get("agent_name") == "AIDetection":
            for entry in meta.get("heatmap") or []:
                if entry.get("heat_level") == "high":
                    text = entry.get("text_preview") or ""
                    if text:
                        spans.append({
                            "text": text, "kind": "ai",
                            "label": f"Likely AI ({entry.get('final_ai_score')}%)",
                        })
            outliers = (meta.get("stylometry") or {}).get("outliers") or []
            for o in outliers:
                text = o.get("text_preview") or ""
                if text:
                    spans.append({
                        "text": text, "kind": "patchwork",
                        "label": "Stylometric patchwork outlier",
                    })
        elif ar.get("agent_name") == "PlagiarismAgent":
            for m in meta.get("matches") or []:
                if m.get("flagged"):
                    text = m.get("paragraph_text") or ""
                    if text:
                        spans.append({
                            "text": text, "kind": "plagiarism",
                            "label": f"Plagiarism match ({m.get('best_similarity')}%)",
                        })
    return spans


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
