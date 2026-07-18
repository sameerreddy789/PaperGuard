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

# AI-score band for status decisions. Kept in sync with
# agents.detector_agent._LIKELY_AI (desklib v1.01's benchmark-derived ~90
# operating point, not the model's naive 50% default).
_LIKELY_AI = float(os.getenv("PAPERGUARD_DETECTOR_AI_THRESHOLD", "90"))


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
        model_name = os.getenv("PAPERGUARD_CREW_MODEL", "gemini/gemini-3.1-flash-lite")
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
            "score, any fabricated (NOT_FOUND) references, any citations whose claims "
            "are unsupported, any RETRACTED references (a serious integrity concern), "
            "and any DOI metadata mismatches (possible tampering or mis-transcription).",
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
        quality_meta = (results.get("QualityAgent") or {}).get("metadata", {}) or {}
        _annotate_heatmap_with_tone(ai, quality_meta)

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
        near_outliers = stylo.get("near_outliers") or []
        if near_outliers and not stylo.get("outlier_count"):
            idxs = ", ".join(str(o.get("paragraph_index")) for o in near_outliers)
            findings.append(
                f"Possible (lower-confidence) style shift: paragraph(s) {idxs} deviate "
                f"somewhat from the document's overall style, though not enough to meet "
                f"the primary patchwork threshold."
            )
        conflict_count = sum(
            1 for h in (ai.get("heatmap") or []) if h.get("ai_score_conflicts_with_tone")
        )
        if conflict_count:
            findings.append(
                f"{conflict_count} paragraph(s) scored 'Likely AI' but were independently "
                f"rated casual/informal in tone by the writing-quality reviewer -- a known "
                f"detector blind spot (see conflict notes)."
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
    headline = _headline_metrics(results)

    # Hard facts (conflict_notes) are ALWAYS attached as a structured field,
    # regardless of whether the LLM crew ran. Previously they only reached the
    # user if the LLM chose to mention them inside its prose summary -- a
    # fabricated-reference count is too important to depend on LLM phrasing.
    summary = crew_summary or _fallback_summary(results, conflict_notes)

    return Report(
        paper_title=_guess_title(text),
        file_name=file_name,
        summary=summary,
        agent_results=agent_results,
        extracted_references=references,
        conflict_notes=conflict_notes,
        headline_metrics=headline,
    )


def _annotate_heatmap_with_tone(ai: Dict[str, Any], quality_meta: Dict[str, Any]) -> None:
    """
    Cross-reference each AI-detection heatmap paragraph against the Quality
    Agent's section-level tone verdict (mutates ``ai["heatmap"]`` in place).

    Rationale (found via a synthetic mixed-authorship test paper, see
    PROJECT_REPORT.md): the detector can read casual, first-person, informal
    prose as "Likely AI" when it is wrapped in academic scaffolding (citations,
    structured claims) -- a known, reduced-but-not-eliminated blind spot per the
    frozen benchmark (75%, not 100%, disguised-AI recall). Rather than silently
    trust the raw AI score in that situation, tag the conflict so the UI/summary
    can surface it instead of stating a bare "Likely AI" verdict.

    This does NOT change any score or classification -- it only adds a flag
    (``ai_score_conflicts_with_tone``) for downstream reporting.
    """
    section_tone = {
        rep.get("section"): rep.get("tone")
        for rep in (quality_meta.get("section_reports") or [])
        if rep.get("section")
    }
    if not section_tone:
        return
    for h in ai.get("heatmap") or []:
        tone = section_tone.get(h.get("section"))
        is_casual = tone in {"casual", "mixed"}
        is_likely_ai = h.get("classification") == "Likely AI"
        h["section_tone"] = tone
        h["ai_score_conflicts_with_tone"] = bool(is_casual and is_likely_ai)


def _band(value: Optional[float], high: float, low: float, higher_is_worse: bool = True) -> str:
    """Classify a 0-100 metric into good/warning/bad for dashboard coloring."""
    if value is None:
        return "unknown"
    if higher_is_worse:
        if value >= high:
            return "bad"
        if value >= low:
            return "warning"
        return "good"
    # higher_is_worse=False (e.g. citation health, quality): higher is better.
    if value <= low:
        return "bad"
    if value <= high:
        return "warning"
    return "good"


def _headline_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic dashboard headline numbers, computed once here so the UI
    (or any other consumer, e.g. the PDF export) never has to re-derive band
    thresholds itself. All four are computed unconditionally regardless of
    which one the UI visually leads with -- ``citation_health_percent`` is
    given top billing in app.py's dashboard (PaperGuard's differentiator; see
    PROJECT_REPORT.md Sections 7-8), with ``ai_percent``/``similarity_percent``
    as supporting signals rather than the Turnitin-style headline this project
    originally led with.
    """
    ai = results.get("AIDetection") or {}
    plag = (results.get("PlagiarismAgent") or {}).get("metadata", {}) or {}
    citation = (results.get("CitationAgent") or {}).get("metadata", {}) or {}
    quality = (results.get("QualityAgent") or {}).get("metadata", {}) or {}

    ai_score = ai.get("overall_ai_score")
    similarity = plag.get("plagiarism_score")
    cite_health = citation.get("citation_health_score")
    quality_score = quality.get("overall_quality_score")

    return {
        "ai_percent": ai_score,
        "ai_band": _band(ai_score, high=_LIKELY_AI, low=35, higher_is_worse=True),
        "similarity_percent": similarity,
        "similarity_band": _band(similarity, high=50, low=20, higher_is_worse=True),
        "citation_health_percent": cite_health,
        "citation_health_band": _band(cite_health, high=80, low=50, higher_is_worse=False),
        "quality_score": quality_score,
        "not_found_citation_count": citation.get("not_found_count", 0),
        "retracted_citation_count": citation.get("retracted_count", 0),
        "patchwork_paragraph_count": (ai.get("stylometry") or {}).get("outlier_count", 0),
    }


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

    # Widened rule: very-high AI score + multiple sections independently rated
    # casual/mixed tone + no stylometric patchwork outliers is the specific
    # pattern of the detector's known disguised-AI blind spot (a human author
    # writing in an informal voice, wrapped in academic scaffolding) rather
    # than genuine uniform AI generation. Surface this explicitly rather than
    # letting a bare "Likely AI" stand unqualified.
    casual_sections = [
        rep.get("section") for rep in (quality.get("section_reports") or [])
        if rep.get("tone") in {"casual", "mixed"}
    ]
    stylo = ai.get("stylometry") or {}
    if (
        ai_score is not None and ai_score >= _LIKELY_AI
        and len(casual_sections) >= 2
        and not stylo.get("outlier_count")
    ):
        notes.append(
            "LOW-CONFIDENCE AI VERDICT: the overall AI score is very high, but "
            f"{len(casual_sections)} section(s) ({', '.join(casual_sections)}) were "
            "independently rated casual/informal in tone by the writing-quality "
            "reviewer, and no stylometric patchwork was detected between sections. "
            "This pattern matches a known detector blind spot (informal human prose "
            "wrapped in academic citation/structure reads as AI-like) rather than "
            "confirmed uniform AI generation -- treat the AI score with reduced "
            "confidence here and weigh the tone/structure signals alongside it."
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
            "in CrossRef/OpenAlex) - a strong integrity concern."
        )
    if citation.get("retracted_count"):
        notes.append(
            f"{citation['retracted_count']} cited reference(s) have been RETRACTED "
            "per CrossRef - the strongest citation integrity concern possible."
        )
    if citation.get("doi_mismatch_count"):
        notes.append(
            f"{citation['doi_mismatch_count']} reference(s) have metadata that does "
            "not match their resolved CrossRef record (possible tampering or "
            "mis-transcription)."
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
