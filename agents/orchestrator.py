"""
Orchestrator -- the "manager" of PaperGuard's agent society.

Runs the full verification pipeline over a paper and assembles a single
``models.report.Report``. It coordinates a CrewAI crew of specialist agents
(each equipped with a deterministic tool from ``crew_tools``) plus a lead
"Research Integrity Editor" that synthesises the findings and flags cross-agent
conflicts.

Robust by design:
  * If CrewAI or the LLM is unavailable, it falls back to running the
    deterministic tools directly (the "engine path") and synthesises a summary
    without the LLM. Either way the structured results are identical, because
    the crew's tools and the engine path call the same core implementations.

Public API:
    analyze_paper(path)         -> models.report.Report
    analyze_text(text, name)    -> models.report.Report
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from agents.base import extract_references, load_text, load_env
from agents.crew_tools import (
    CORE_TOOLS,
    build_crewai_tools,
    get_cached_results,
    set_paper_context,
)
from models.reference import Reference
from models.report import AgentResult, Report

# AI-score band for status decisions.
_LIKELY_AI = 65


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def analyze_paper(path: str, use_crew: bool = True) -> Report:
    """Analyze a paper file (.pdf/.md/.txt) and return a full Report."""
    load_env()
    text = load_text(path)
    return analyze_text(text, file_name=os.path.basename(path), use_crew=use_crew)


def analyze_text(text: str, file_name: str = "input.txt", use_crew: bool = True) -> Report:
    """Analyze already-extracted paper text and return a full Report."""
    load_env()
    references = _safe_extract_references(text)
    set_paper_context(text, references)

    crew_summary: Optional[str] = None
    if use_crew:
        crew_summary = _run_crew()

    results = get_cached_results()
    if not results:
        # Engine path: run every tool implementation directly.
        for impl in CORE_TOOLS.values():
            try:
                impl()
            except Exception as exc:  # noqa: BLE001
                print(f"[orchestrator] tool failed: {exc}")
        results = get_cached_results()

    return _build_report(text, file_name, references, results, crew_summary)


# --------------------------------------------------------------------------- #
# CrewAI crew
# --------------------------------------------------------------------------- #
def _run_crew() -> Optional[str]:
    """
    Build and run the CrewAI crew. Returns the editor's synthesis text, or
    ``None`` if CrewAI / the LLM is unavailable or the run fails (the caller
    then falls back to the deterministic engine path).
    """
    # The crew can run on any LiteLLM-supported provider. Gemini by default, but
    # Alibaba DashScope/Qwen (the deployment target) works by setting
    # PAPERGUARD_CREW_MODEL=dashscope/qwen-plus (+ DASHSCOPE_API_KEY), optionally
    # with PAPERGUARD_CREW_API_BASE for the OpenAI-compatible endpoint.
    if not (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("PAPERGUARD_CREW_API_BASE")
    ):
        return None
    try:
        from crewai import Agent, Crew, LLM, Process, Task
    except Exception:
        return None

    tools = build_crewai_tools()
    if not tools:
        return None

    try:
        model_name = os.getenv("PAPERGUARD_CREW_MODEL", "gemini/gemini-2.5-flash")
        # Deterministic synthesis by default (same env var as the verdict calls in
        # services.gemini) so a paper's report is reproducible run-to-run.
        try:
            crew_temp = float(os.getenv("PAPERGUARD_LLM_TEMPERATURE", "0.0"))
        except ValueError:
            crew_temp = 0.0
        llm_kwargs: Dict[str, Any] = {"model": model_name, "temperature": crew_temp}
        crew_api_base = os.getenv("PAPERGUARD_CREW_API_BASE")
        if crew_api_base:
            llm_kwargs["base_url"] = crew_api_base
        llm = LLM(**llm_kwargs)

        citation_agent = Agent(
            role="Citation Integrity Specialist",
            goal="Determine whether every reference exists and truly supports the claim it is cited for.",
            backstory="A meticulous research librarian who has caught countless fabricated and misused citations.",
            tools=[tools["citation"]], llm=llm, verbose=False, allow_delegation=False,
        )
        ai_agent = Agent(
            role="AI-Text Forensics Analyst",
            goal="Report how much of the paper is AI-generated using the model detector and flag style patchwork.",
            backstory="An analyst who interprets the fine-tuned detector's per-paragraph scores and stylometric signals.",
            tools=[tools["ai_detection"]], llm=llm, verbose=False, allow_delegation=False,
        )
        plagiarism_agent = Agent(
            role="Plagiarism Investigator",
            goal="Find text overlapping open web and scholarly sources, honestly bounded by available databases.",
            backstory="A careful investigator who distinguishes genuine overlap from properly attributed quotation.",
            tools=[tools["plagiarism"]], llm=llm, verbose=False, allow_delegation=False,
        )
        quality_agent = Agent(
            role="Academic Writing Reviewer",
            goal="Assess structure, readability, and scholarly prose quality of the paper.",
            backstory="A journal copy-editor with an eye for structure and academic tone.",
            tools=[tools["quality"]], llm=llm, verbose=False, allow_delegation=False,
        )
        editor = Agent(
            role="Research Integrity Editor",
            goal="Synthesise all specialist findings into a single honest verdict and resolve conflicts between them.",
            backstory="A senior editor who weighs evidence across concerns and never overstates certainty.",
            llm=llm, verbose=False, allow_delegation=False,
        )

        def _mk_task(desc: str, out: str, agent: "Agent") -> "Task":
            return Task(description=desc, expected_output=out, agent=agent)

        t_citation = _mk_task(
            "Use your tool to verify the paper's citations. Report the citation-health "
            "score, any fabricated (NOT_FOUND) references, and any citations whose "
            "claims are unsupported.",
            "A short paragraph summarising citation integrity with specific flagged references.",
            citation_agent,
        )
        t_ai = _mk_task(
            "Use your tool to run model-based AI detection. Report the overall AI "
            "score and classification, which paragraphs are flagged as likely-AI, and "
            "any stylometric-patchwork paragraphs (possible AI text pasted into human "
            "writing). Note the model's known blind spot: slang/style-masked AI.",
            "A short paragraph on AI-generation likelihood with flagged/patchwork paragraphs.",
            ai_agent,
        )
        t_plag = _mk_task(
            "Use your tool to scan for plagiarism against open sources. Report the "
            "similarity score and flagged paragraphs, and state the honest limitation "
            "that proprietary databases are not covered.",
            "A short paragraph on text-overlap findings with the limitation stated.",
            plagiarism_agent,
        )
        t_quality = _mk_task(
            "Use your tool to review writing quality. Report the overall quality score, "
            "structural completeness, and the most important prose issues.",
            "A short paragraph on writing quality with concrete issues.",
            quality_agent,
        )
        t_synth = _mk_task(
            "Review all four specialist reports. Produce a concise, honest research-"
            "integrity assessment for this paper. Explicitly resolve conflicts, e.g.: "
            "if plagiarism overlap is actually a properly cited quotation, or if a high "
            "AI score coincides with technical writing the quality reviewer deemed "
            "domain-appropriate (lower the AI concern accordingly). Do not overstate "
            "certainty; AI detection and plagiarism are indicators, not verdicts.",
            "A 2-4 sentence executive summary plus a short bulleted list of the key "
            "issues and any resolved conflicts.",
            editor,
        )
        t_synth.context = [t_citation, t_ai, t_plag, t_quality]

        crew = Crew(
            agents=[citation_agent, ai_agent, plagiarism_agent, quality_agent, editor],
            tasks=[t_citation, t_ai, t_plag, t_quality, t_synth],
            process=Process.sequential,
            verbose=False,
        )
        result = crew.kickoff()
        return str(getattr(result, "raw", result)).strip() or None
    except Exception as exc:  # noqa: BLE001 - any crew failure => engine fallback
        print(f"[orchestrator] CrewAI run failed, falling back to engine: {exc}")
        return None


# --------------------------------------------------------------------------- #
# Report assembly
# --------------------------------------------------------------------------- #
def _build_report(
    text: str,
    file_name: str,
    references: List[Reference],
    results: Dict[str, Any],
    crew_summary: Optional[str],
) -> Report:
    agent_results: List[AgentResult] = []

    # AI detection (model-based) -> AgentResult
    ai = results.get("AIDetection")
    if ai is not None:
        overall = ai.get("overall_ai_score")
        status = "warning" if (overall is not None and overall >= _LIKELY_AI) else "passed"
        findings = [
            f"Overall AI-content score: {overall}% ({ai.get('classification')})."
            if overall is not None else "AI score could not be computed (detector unavailable).",
        ]
        flagged = ai.get("flagged_paragraphs") or []
        if flagged:
            findings.append(f"Paragraph(s) flagged as likely-AI: {', '.join(map(str, flagged))}.")
        stylo = ai.get("stylometry") or {}
        if stylo.get("outlier_count"):
            idxs = ", ".join(str(o.get("paragraph_index")) for o in stylo.get("outliers", []))
            findings.append(
                f"Stylometric patchwork: paragraph(s) {idxs} deviate in style "
                f"(possible mixed authorship / pasted AI)."
            )
        agent_results.append(AgentResult(
            agent_name="AIDetection", status=status, findings=findings, metadata=ai,
        ))

    # The other agents already stored full AgentResult dumps.
    for name in ("CitationAgent", "QualityAgent", "PlagiarismAgent"):
        payload = results.get(name)
        if payload is not None:
            agent_results.append(AgentResult(**payload))

    conflict_notes = _cross_agent_conflicts(results)

    summary = crew_summary or _fallback_summary(results, conflict_notes)

    return Report(
        paper_title=_guess_title(text),
        file_name=file_name,
        summary=summary,
        agent_results=agent_results,
        extracted_references=references,
    )


def _cross_agent_conflicts(results: Dict[str, Any]) -> List[str]:
    """Deterministic cross-agent conflict/consistency notes (from the plan)."""
    notes: List[str] = []
    ai = results.get("AIDetection") or {}
    quality = (results.get("QualityAgent") or {}).get("metadata", {})
    plag = (results.get("PlagiarismAgent") or {}).get("metadata", {})
    citation = (results.get("CitationAgent") or {}).get("metadata", {})

    ai_score = ai.get("overall_ai_score")
    q_struct = (quality.get("structure") or {}).get("completeness_score")
    if ai_score is not None and ai_score >= _LIKELY_AI and q_struct is not None and q_struct >= 0.8:
        notes.append(
            "AI detection is high but the paper is structurally complete and domain-"
            "appropriate; treat the AI signal as an indicator, not a verdict."
        )

    plag_flags = plag.get("flagged_paragraph_count")
    cite_health = citation.get("citation_health_score")
    if plag_flags and cite_health is not None and cite_health >= 80:
        notes.append(
            "Plagiarism overlap was flagged while citations are largely healthy - "
            "some overlap may be properly attributed quotation; review in context."
        )

    if citation.get("not_found_count"):
        notes.append(
            f"{citation['not_found_count']} reference(s) appear fabricated (not found "
            "in CrossRef/Semantic Scholar) - a strong integrity concern."
        )
    return notes


def _fallback_summary(results: Dict[str, Any], conflict_notes: List[str]) -> str:
    ai = results.get("AIDetection") or {}
    citation = (results.get("CitationAgent") or {}).get("metadata", {})
    plag = (results.get("PlagiarismAgent") or {}).get("metadata", {})
    quality = (results.get("QualityAgent") or {}).get("metadata", {})

    parts = ["PaperGuard analysis (engine mode):"]
    if ai.get("overall_ai_score") is not None:
        parts.append(f"AI-content {ai['overall_ai_score']}% ({ai.get('classification')}).")
    if citation.get("citation_health_score") is not None:
        parts.append(f"Citation health {citation['citation_health_score']}%.")
    if plag.get("plagiarism_score") is not None:
        parts.append(f"Plagiarism similarity {plag['plagiarism_score']}%.")
    if quality.get("overall_quality_score") is not None:
        parts.append(f"Writing quality {quality['overall_quality_score']}/10.")
    summary = " ".join(parts)
    if conflict_notes:
        summary += " Conflict notes: " + " ".join(conflict_notes)
    return summary


def _guess_title(text: str) -> Optional[str]:
    for line in (text or "").splitlines():
        s = line.strip().lstrip("#").strip()
        if len(s) > 10 and not s.lower().startswith(("abstract", "introduction")):
            return s[:200]
    return None


def _safe_extract_references(text: str) -> List[Reference]:
    try:
        return extract_references(text)
    except Exception:
        return []
