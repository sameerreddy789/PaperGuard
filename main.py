"""
PaperGuard CLI -- run the full multi-agent analysis on a paper.

Usage:
    python main.py path/to/paper.pdf
    python main.py path/to/paper.md --json -o report.json
    python main.py path/to/paper.pdf --no-crew      # skip CrewAI, engine only

Outputs a human-readable report by default, or the full structured JSON with
``--json``. Works with .pdf, .md, and .txt inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _print_human(report) -> None:
    md = report.model_dump()
    line = "=" * 68
    print(line)
    print(f"PaperGuard Report  --  {md.get('file_name')}")
    if md.get("paper_title"):
        print(f"Title: {md['paper_title']}")
    print(line)
    print("\nEXECUTIVE SUMMARY\n-----------------")
    print(md.get("summary", "(none)"))

    print("\nAGENT RESULTS\n-------------")
    for ar in md.get("agent_results", []):
        print(f"\n[{ar['agent_name']}]  status={ar['status']}")
        for f in ar.get("findings", []):
            print(f"  - {f}")

    refs = md.get("extracted_references", [])
    print(f"\nREFERENCES EXTRACTED: {len(refs)}")
    print("\n" + line)


def main() -> None:
    parser = argparse.ArgumentParser(description="PaperGuard multi-agent paper analysis.")
    parser.add_argument("input", help="Path to the paper (.pdf, .md, or .txt)")
    parser.add_argument("--json", action="store_true", help="Output full structured JSON.")
    parser.add_argument("-o", "--output", default=None, help="Write output to this file.")
    parser.add_argument("--no-crew", action="store_true",
                        help="Skip the CrewAI layer; run the deterministic engine only.")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Imported here so `python main.py --help` works even if heavy deps are missing.
    from agents.orchestrator import analyze_paper

    report = analyze_paper(args.input, use_crew=not args.no_crew)

    if args.json:
        payload = json.dumps(report.model_dump(), indent=2, ensure_ascii=False)
        if args.output:
            Path(args.output).write_text(payload, encoding="utf-8")
            print(f"Report written to {args.output}")
        else:
            print(payload)
    else:
        _print_human(report)
        if args.output:
            Path(args.output).write_text(
                json.dumps(report.model_dump(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"\n(Full JSON written to {args.output})")


if __name__ == "__main__":
    main()
