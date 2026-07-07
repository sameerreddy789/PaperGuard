"""
PaperGuard agents package.

Each agent is a standalone module that accepts paper text (and, where relevant,
extracted references) and returns a structured `AgentResult`. Every agent also
exposes a CLI entry point so it can be run independently, e.g.:

    python -m agents.citation_agent path/to/paper.pdf
    python -m agents.quality_agent  path/to/paper.md

Agents degrade gracefully when optional dependencies (Gemini SDK) or API keys
are unavailable, returning partial results rather than crashing.
"""
