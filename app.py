"""
PaperGuard -- Streamlit UI.

Upload an academic paper (PDF/MD/TXT) and get a multi-agent integrity report:
per-paragraph AI-detection heatmap (with safety-net override reasoning),
citation verification, plagiarism scan, and writing-quality review.

Run:  streamlit run app.py
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

st.set_page_config(page_title="PaperGuard", layout="wide")

# --------------------------------------------------------------------------- #
# Styling (light-mode, high-contrast; heatmap colour bands)
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
      .pg-title { font-size: 2.1rem; font-weight: 800; color: #0F172A; margin-bottom: 0; }
      .pg-sub   { color: #475569; font-size: 1.02rem; margin-top: 2px; }
      .pg-para  { padding: 10px 14px; margin: 7px 0; border-radius: 8px;
                  border-left: 5px solid; }
      .pg-head  { display:flex; justify-content:space-between; align-items:center; }
      .pg-sec   { font-weight: 700; font-size: 0.9rem; }
      .pg-score { font-weight: 800; font-size: 0.95rem; }
      .pg-body  { color: #0F172A; margin-top: 6px; font-size: 0.9rem; line-height: 1.4; }
      .pg-note  { margin-top: 8px; padding: 7px 10px; border-radius: 6px;
                  background: #EEF2FF; color: #3730A3; font-size: 0.82rem; font-weight: 600; }
      .pg-legend span { display:inline-block; padding:3px 10px; border-radius: 12px;
                  font-size:0.78rem; font-weight:700; margin-right:8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

_HEAT = {
    "high":    {"bg": "#FEE2E2", "border": "#EF4444", "text": "#991B1B"},
    "medium":  {"bg": "#FEF3C7", "border": "#F59E0B", "text": "#92400E"},
    "low":     {"bg": "#DCFCE7", "border": "#22C55E", "text": "#166534"},
    "unknown": {"bg": "#F1F5F9", "border": "#94A3B8", "text": "#475569"},
}

# Band -> color for the Integrity Dashboard headline numbers.
_BAND_COLOR = {
    "bad": "#EF4444", "warning": "#F59E0B", "good": "#22C55E", "unknown": "#94A3B8",
}


# --------------------------------------------------------------------------- #
# Analysis (cached by file content + settings)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _analyze(file_bytes: bytes, suffix: str, use_crew: bool, model: str, _cache_key: str) -> Dict[str, Any]:
    """Run the orchestrator on uploaded bytes and return report.model_dump()."""
    if model:
        os.environ["PAPERGUARD_DETECTOR_MODEL"] = model
    from agents.orchestrator import analyze_paper

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(file_bytes)
        tmp.close()
        report = analyze_paper(tmp.name, use_crew=use_crew)
        data = report.model_dump()
        data["_source_text"] = report_source_text(tmp.name)
        return data
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def report_source_text(path: str) -> str:
    """Re-extract the paper text (for full-paragraph display in the heatmap)."""
    try:
        from agents.base import load_text
        return load_text(path)
    except Exception:
        return ""


def _full_paragraphs(text: str) -> Dict[int, str]:
    """Map paragraph_index -> full paragraph text, matching the detector selection."""
    try:
        from agents.ai_detection import _select_paragraphs
        paras = _select_paragraphs(text, None)
        return {i: p for i, (_sec, p) in enumerate(paras, start=1)}
    except Exception:
        return {}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _agent(report: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    for ar in report.get("agent_results", []):
        if ar.get("agent_name") == name:
            return ar
    return None


def _meta(report: Dict[str, Any], name: str) -> Dict[str, Any]:
    ar = _agent(report, name)
    return (ar or {}).get("metadata") or {}


def _fmt(v: Any, suffix: str = "") -> str:
    return f"{v}{suffix}" if v is not None else "N/A"


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #
def render_dashboard(report: Dict[str, Any]) -> None:
    """
    Integrity Dashboard: Citation Health gets top billing as PaperGuard's
    primary differentiator (existence + retraction + DOI-consistency + claim
    support -- no major commercial tool checks the latter two), with AI% and
    Similarity% shown as supporting signals rather than the headline (see
    PROJECT_REPORT.md Sections 7-8 for the positioning rationale: AI/plagiarism
    detection is a crowded category we're structurally behind in, citation
    verification is the genuinely differentiated capability). Uses
    ``report['headline_metrics']`` (computed deterministically by the
    orchestrator, task-1 fix) rather than re-deriving thresholds here, so the
    UI and any other consumer (PDF export, API) always agree on the same
    numbers.
    """
    h = report.get("headline_metrics") or {}
    cite = _meta(report, "CitationAgent")
    plag = _meta(report, "PlagiarismAgent")

    st.markdown(
        """
        <style>
          .pg-headline { border-radius: 12px; padding: 18px 20px; text-align: center; }
          .pg-headline .val { font-size: 2.4rem; font-weight: 800; line-height: 1; }
          .pg-headline .lbl { font-size: 0.95rem; font-weight: 700; margin-top: 4px; opacity: 0.85; }
          .pg-headline .sub { font-size: 0.78rem; margin-top: 4px; opacity: 0.75; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    def _headline_card(col, label: str, value: Optional[float], band: str, sub: str) -> None:
        color = _BAND_COLOR.get(band, _BAND_COLOR["unknown"])
        with col:
            st.markdown(
                f'<div class="pg-headline" style="background:{color}14;border:2px solid {color};">'
                f'<div class="val" style="color:{color};">{_fmt(value, "%")}</div>'
                f'<div class="lbl" style="color:{color};">{_escape(label)}</div>'
                f'<div class="sub">{_escape(sub)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # --- Citation Health leads: PaperGuard's differentiator, not AI%/Similarity%. --- #
    hc1, hc2 = st.columns(2)
    _headline_card(
        hc1, "Citation Health", h.get("citation_health_percent"), h.get("citation_health_band", "unknown"),
        f"{h.get('not_found_citation_count', 0)} not found · "
        f"{h.get('retracted_citation_count', 0)} retracted -- existence, retraction, "
        f"DOI-consistency, and claim support.",
    )
    _headline_card(
        hc2, "AI-Generated Content", h.get("ai_percent"), h.get("ai_band", "unknown"),
        "desklib deberta-v3-large detector - indicator, not a verdict.",
    )

    st.write("")

    # --- Secondary metrics (AI/plagiarism supporting signals + quality). --- #
    c1, c2, c3 = st.columns(3)
    c1.metric("Similarity / Plagiarism", _fmt(h.get("similarity_percent"), "%"),
              help="Open-access scholarly sources + known-text fingerprints only.")
    c2.metric("Writing Quality", _fmt(h.get("quality_score"), "/10"))
    c3.metric("Patchwork paragraphs", h.get("patchwork_paragraph_count", 0),
              help="Paragraphs whose stylometric fingerprint deviates from the rest of the paper.")

    # --- Hard facts (always shown, independent of the LLM crew - task 1 fix). --- #
    notes = report.get("conflict_notes") or []
    if notes:
        with st.container(border=True):
            st.markdown("**Hard facts** *(deterministic - always shown regardless of LLM availability)*")
            for n in notes:
                st.markdown(f"- {n}")


def render_heatmap(report: Dict[str, Any]) -> None:
    ai = _meta(report, "AIDetection")
    heatmap: List[Dict[str, Any]] = ai.get("heatmap") or []
    comp = ai.get("components") or {}

    st.markdown(
        '<div class="pg-legend">'
        '<span style="background:#FEE2E2;color:#991B1B;">Likely AI (&ge;90%)</span>'
        '<span style="background:#FEF3C7;color:#92400E;">Uncertain</span>'
        '<span style="background:#DCFCE7;color:#166534;">Likely Human (&le;35%)</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Detector: {'on' if comp.get('detector_available') else 'off'} · "
        f"model: {comp.get('detector_model', 'n/a')} (sigmoid classifier output)"
    )
    if not comp.get("detector_available"):
        st.warning("Detector model unavailable; AI scores could not be computed.", icon=None)

    # Stylometric patchwork ("Frankenstein") signal.
    stylo = ai.get("stylometry") or {}
    if stylo.get("available"):
        cohesion = stylo.get("style_cohesion")
        outliers = stylo.get("outliers") or []
        if outliers:
            idxs = ", ".join(str(o.get("paragraph_index")) for o in outliers)
            st.error(
                f"Possible mixed authorship (patchwork): paragraph(s) {idxs} deviate "
                f"stylometrically from the rest of the paper (style cohesion {cohesion}). "
                "This can indicate AI text pasted into human writing. Indicative, not definitive.",
                icon=None,
            )
        else:
            st.caption(f"Stylometric cohesion: {cohesion} (no anomalous-style paragraphs detected).")

    if not heatmap:
        st.info("No paragraph-level AI results available.")
        return

    fulltext = _full_paragraphs(report.get("_source_text", ""))
    patch_idx = {o.get("paragraph_index") for o in (stylo.get("outliers") or [])}

    for entry in heatmap:
        level = entry.get("heat_level", "unknown")
        c = _HEAT.get(level, _HEAT["unknown"])
        score = entry.get("final_ai_score")
        idx = entry.get("paragraph_index")
        body = fulltext.get(idx) or entry.get("text_preview") or ""
        note = ""
        if idx in patch_idx:
            note = (
                '<div class="pg-note" style="background:#FEF2F2;color:#991B1B;">'
                'Stylometric outlier &mdash; possible different author (patchwork).</div>'
            )

        st.markdown(
            f'<div class="pg-para" style="border-color:{c["border"]}; background:{c["bg"]};">'
            f'<div class="pg-head">'
            f'<span class="pg-sec" style="color:{c["text"]};">Paragraph {idx} &middot; {entry.get("section","")}</span>'
            f'<span class="pg-score" style="color:{c["text"]};">{_fmt(score, "% AI")}</span>'
            f'</div>'
            f'<div class="pg-body">{_escape(body)}</div>'
            f'{note}</div>',
            unsafe_allow_html=True,
        )


def _normalize_for_match(text: str) -> str:
    """Loose normalization for cross-agent paragraph matching (see note below)."""
    return " ".join((text or "").split()).lower()


def _build_plagiarism_index(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Map normalized paragraph text -> plagiarism match entry.

    The AI-detection heatmap and the plagiarism agent select overlapping but
    NOT identical paragraph subsets (different min-length filters and, for
    plagiarism, a cap of 10 paragraphs) - so their ``paragraph_index`` values
    do not line up. We join on normalized paragraph TEXT instead, using the
    ``paragraph_text`` field the plagiarism agent now includes precisely for
    this purpose (see agents/plagiarism_agent.py).
    """
    plag = _meta(report, "PlagiarismAgent")
    index: Dict[str, Dict[str, Any]] = {}
    for m in plag.get("matches") or []:
        text = m.get("paragraph_text")
        if text:
            index[_normalize_for_match(text)] = m
    return index


def render_overlay(report: Dict[str, Any]) -> None:
    """
    Combined per-paragraph overlay: AI heat + plagiarism flag + stylometric
    patchwork, all in one pass over the same paragraph list (the AI-detection
    heatmap, which covers the full paper) so a reviewer sees every signal for
    a paragraph at a glance instead of cross-referencing three tabs.
    """
    ai = _meta(report, "AIDetection")
    heatmap: List[Dict[str, Any]] = ai.get("heatmap") or []
    stylo = ai.get("stylometry") or {}
    patch_idx = {o.get("paragraph_index") for o in (stylo.get("outliers") or [])}
    plag_index = _build_plagiarism_index(report)

    if not heatmap:
        st.info("No paragraph-level results available.")
        return

    fulltext = _full_paragraphs(report.get("_source_text", ""))
    st.caption(
        "One row per paragraph: AI-likelihood color, plagiarism match (if any), "
        "and stylometric patchwork flag (if any) - combined so nothing requires "
        "cross-referencing separate tabs."
    )

    for entry in heatmap:
        idx = entry.get("paragraph_index")
        level = entry.get("heat_level", "unknown")
        c = _HEAT.get(level, _HEAT["unknown"])
        score = entry.get("final_ai_score")
        body = fulltext.get(idx) or entry.get("text_preview") or ""

        plag_match = plag_index.get(_normalize_for_match(body))
        is_patchwork = idx in patch_idx

        badges = []
        if plag_match and plag_match.get("flagged"):
            badges.append(
                f'<span style="background:#FEE2E2;color:#991B1B;padding:2px 8px;'
                f'border-radius:10px;font-size:0.75rem;font-weight:700;margin-left:6px;">'
                f'Plagiarism {plag_match.get("best_similarity")}%</span>'
            )
        elif plag_match and plag_match.get("downgraded_attributed_quote"):
            badges.append(
                f'<span style="background:#EEF2FF;color:#3730A3;padding:2px 8px;'
                f'border-radius:10px;font-size:0.75rem;font-weight:700;margin-left:6px;">'
                f'Quoted &amp; cited (not counted)</span>'
            )
        if is_patchwork:
            badges.append(
                '<span style="background:#FEF2F2;color:#991B1B;padding:2px 8px;'
                'border-radius:10px;font-size:0.75rem;font-weight:700;margin-left:6px;">'
                'Patchwork</span>'
            )
        badge_html = "".join(badges)

        st.markdown(
            f'<div class="pg-para" style="border-color:{c["border"]}; background:{c["bg"]};">'
            f'<div class="pg-head">'
            f'<span class="pg-sec" style="color:{c["text"]};">Paragraph {idx} &middot; '
            f'{entry.get("section","")}{badge_html}</span>'
            f'<span class="pg-score" style="color:{c["text"]};">{_fmt(score, "% AI")}</span>'
            f'</div>'
            f'<div class="pg-body">{_escape(body)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_citations(report: Dict[str, Any]) -> None:
    meta = _meta(report, "CitationAgent")
    tiers = meta.get("tier_counts") or {}
    a, b, c, d = st.columns(4)
    a.metric("Verified", tiers.get("VERIFIED", 0))
    b.metric("Partial", tiers.get("PARTIALLY_VERIFIED", 0))
    c.metric("Existence only", tiers.get("EXISTENCE_ONLY", 0))
    d.metric("Not found", tiers.get("NOT_FOUND", 0))

    if meta.get("retracted_count") or meta.get("doi_mismatch_count"):
        e, f = st.columns(2)
        if meta.get("retracted_count"):
            e.error(f"{meta['retracted_count']} RETRACTED reference(s) cited.")
        if meta.get("doi_mismatch_count"):
            f.warning(f"{meta['doi_mismatch_count']} reference(s) with metadata mismatches.")

    if not meta.get("llm_claim_verification_enabled"):
        st.caption("Claim-support verification was off (no valid LLM key); tiers reflect existence checks.")

    rows = meta.get("results") or []
    if rows:
        table = [{
            "#": r.get("index"),
            "Title": (r.get("title") or "")[:80],
            "Year": r.get("year"),
            "Exists": "Yes" if r.get("exists") else "No",
            "Tier": r.get("tier"),
            "Claim": r.get("claim_verdict") or "-",
            "Retracted": "YES" if (r.get("retraction") or {}).get("retracted") else "-",
            "DOI issues": "; ".join((r.get("doi_consistency") or {}).get("mismatches") or []) or "-",
        } for r in rows]
        st.dataframe(table, use_container_width=True, hide_index=True)
    else:
        st.info("No references were extracted.")


def render_findings(report: Dict[str, Any], name: str, empty: str) -> None:
    ar = _agent(report, name)
    if not ar:
        st.info(empty)
        return
    st.caption(f"Status: {ar.get('status', 'unknown')}")
    for f in ar.get("findings", []):
        st.markdown(f"- {f}")


def render_references(report: Dict[str, Any]) -> None:
    refs = report.get("extracted_references", [])
    st.caption(f"{len(refs)} reference(s) extracted")
    for i, r in enumerate(refs, start=1):
        title = r.get("title") or r.get("raw_text") or "(untitled)"
        authors = ", ".join(r.get("authors") or []) or "unknown authors"
        year = r.get("year") or "n.d."
        doi = f" · doi:{r['doi']}" if r.get("doi") else ""
        st.markdown(f"**[{i}]** {title}  \n<span style='color:#475569;font-size:0.85rem;'>{authors} ({year}){doi}</span>",
                    unsafe_allow_html=True)


def _escape(text: str) -> str:
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# --------------------------------------------------------------------------- #
# PDF export
# --------------------------------------------------------------------------- #
def build_pdf(report: Dict[str, Any]) -> bytes:
    """Render the report to a PDF (reportlab) and return the bytes."""
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem,
    )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, topMargin=1.6 * cm, bottomMargin=1.6 * cm,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm, title="PaperGuard Report",
    )
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("pg_h1", parent=ss["Title"], fontSize=20, textColor=colors.HexColor("#0F172A"))
    h2 = ParagraphStyle("pg_h2", parent=ss["Heading2"], fontSize=13, textColor=colors.HexColor("#1E293B"),
                        spaceBefore=12, spaceAfter=4)
    body = ParagraphStyle("pg_body", parent=ss["BodyText"], fontSize=9.5, leading=14, alignment=TA_LEFT)
    small = ParagraphStyle("pg_small", parent=ss["BodyText"], fontSize=8, textColor=colors.HexColor("#64748B"))

    ai = _meta(report, "AIDetection")
    plag = _meta(report, "PlagiarismAgent")
    qual = _meta(report, "QualityAgent")
    h = report.get("headline_metrics") or {}

    story: List[Any] = []
    story.append(Paragraph("PaperGuard &mdash; Integrity Report", h1))
    story.append(Paragraph(_escape(report.get("paper_title") or report.get("file_name", "")), body))
    story.append(Spacer(1, 8))

    # Metrics table -- reuses report['headline_metrics'] (same deterministic
    # values the Streamlit dashboard shows) so the PDF and UI never disagree.
    # Citation Health leads (PaperGuard's differentiator), matching the
    # dashboard's card order.
    metrics = [
        ["Citation Health", _fmt(h.get("citation_health_percent"), "%"), "AI Content", _fmt(h.get("ai_percent"), "%")],
        ["Plagiarism", _fmt(h.get("similarity_percent"), "%"), "Writing Quality", _fmt(h.get("quality_score"), "/10")],
    ]
    tbl = Table(metrics, colWidths=[3.6 * cm, 4.0 * cm, 3.6 * cm, 4.0 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#0F172A")),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(tbl)

    story.append(Paragraph("Executive Summary", h2))
    story.append(Paragraph(_escape(report.get("summary", "(none)")), body))

    # Hard facts (deterministic, always populated regardless of LLM - task 1).
    notes = report.get("conflict_notes") or []
    if notes:
        story.append(Paragraph("Hard Facts", h2))
        items = [ListItem(Paragraph(_escape(n), body)) for n in notes]
        story.append(ListFlowable(items, bulletType="bullet", start="•"))

    # Stylometry
    stylo = ai.get("stylometry") or {}
    if stylo.get("available") and stylo.get("outliers"):
        idxs = ", ".join(str(o.get("paragraph_index")) for o in stylo["outliers"])
        story.append(Paragraph("Stylometric Patchwork Signal", h2))
        story.append(Paragraph(
            _escape(f"Paragraph(s) {idxs} deviate stylometrically from the rest of the paper "
                    f"(style cohesion {stylo.get('style_cohesion')}); possible mixed authorship."),
            body,
        ))

    # Per-agent findings (Citations first -- PaperGuard's differentiator).
    for name, label in [
        ("CitationAgent", "Citations"),
        ("AIDetection", "AI Detection"),
        ("PlagiarismAgent", "Plagiarism"),
        ("QualityAgent", "Writing Quality"),
    ]:
        ar = _agent(report, name)
        if not ar:
            continue
        story.append(Paragraph(f"{label} &mdash; status: {_escape(ar.get('status',''))}", h2))
        items = [ListItem(Paragraph(_escape(f), body)) for f in ar.get("findings", [])[:12]]
        if items:
            story.append(ListFlowable(items, bulletType="bullet", start="•"))

    # References
    refs = report.get("extracted_references", [])
    story.append(Paragraph(f"References ({len(refs)})", h2))
    for i, r in enumerate(refs, start=1):
        title = r.get("title") or r.get("raw_text") or "(untitled)"
        authors = ", ".join(r.get("authors") or []) or "unknown authors"
        year = r.get("year") or "n.d."
        story.append(Paragraph(_escape(f"[{i}] {title} — {authors} ({year})"), small))

    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "Disclaimer: PaperGuard is a pre-submission self-check, not a definitive verdict. "
        "AI-detection and plagiarism results are probabilistic indicators; plagiarism coverage "
        "is limited to open web and open-access scholarly sources.", small,
    ))

    doc.build(story)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Annotated PDF export (Turnitin-style highlights on the ORIGINAL PDF)
# --------------------------------------------------------------------------- #
def build_annotated_pdf(report: Dict[str, Any], original_pdf_bytes: bytes):
    """Highlight likely-AI / plagiarism / patchwork spans on the uploaded PDF."""
    from core.pdf_parser import highlight_pdf, spans_from_report
    spans = spans_from_report(report)
    return highlight_pdf(original_pdf_bytes, spans)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
# Note: the old local v2.0 fallback path (training/mega_dataset_model_v2) was
# removed in the model swap to desklib/ai-text-detector-v1.01 -- it used a
# different (incompatible) model architecture/class than the current detector.
_default_model = "desklib/ai-text-detector-v1.01"

with st.sidebar:
    st.header("Settings")
    gemini_key = st.text_input("Gemini API key", type="password",
                               help="Enables the Linguistic agent + CrewAI crew. Without it, detection runs on the model alone.")
    model_path = st.text_input("Detector model", value=_default_model,
                               help="Local path or HuggingFace repo id.")
    use_crew = st.toggle("Use CrewAI multi-agent crew", value=True,
                         help="Off = deterministic engine path (faster, no LLM synthesis).")
    st.divider()
    st.caption("PaperGuard - multi-agent academic integrity verification.")

if gemini_key:
    os.environ["GEMINI_API_KEY"] = gemini_key.strip()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
st.markdown('<p class="pg-title">PaperGuard</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="pg-sub">Citation-integrity verification &mdash; existence, retraction, and '
    'claim-support checks &mdash; plus AI-content, plagiarism, and writing-quality signals.</p>',
    unsafe_allow_html=True,
)

uploaded = st.file_uploader("Upload a paper", type=["pdf", "md", "txt"])
run = st.button("Analyze paper", type="primary", disabled=uploaded is None)

if run and uploaded is not None:
    file_bytes = uploaded.getvalue()
    suffix = Path(uploaded.name).suffix or ".txt"
    cache_key = hashlib.sha256(
        file_bytes + f"|{use_crew}|{model_path}|{bool(gemini_key)}".encode()
    ).hexdigest()
    with st.spinner("Running the agent society... this can take 1-3 minutes on first run."):
        try:
            st.session_state["report"] = _analyze(file_bytes, suffix, use_crew, model_path, cache_key)
            st.session_state["report_name"] = uploaded.name
            # Keep the ORIGINAL bytes (not the temp file, which _analyze already
            # deleted) so the annotated-PDF export can highlight spans directly
            # on the paper the user actually uploaded.
            st.session_state["report_pdf_bytes"] = file_bytes if suffix.lower() == ".pdf" else None
        except Exception as exc:  # noqa: BLE001
            st.error(f"Analysis failed: {exc}")

report = st.session_state.get("report")
if report:
    st.divider()
    title = report.get("paper_title") or st.session_state.get("report_name", "")
    st.subheader(title)
    render_dashboard(report)

    with st.container(border=True):
        st.markdown("**Executive summary**")
        st.write(report.get("summary", "(none)"))

    tabs = st.tabs(["Citations", "Overlay", "AI Heatmap", "Plagiarism", "Writing Quality", "References"])
    with tabs[0]:
        render_citations(report)
    with tabs[1]:
        render_overlay(report)
    with tabs[2]:
        render_heatmap(report)
    with tabs[3]:
        render_findings(report, "PlagiarismAgent", "No plagiarism results.")
    with tabs[4]:
        render_findings(report, "QualityAgent", "No writing-quality results.")
    with tabs[5]:
        render_references(report)

    stem = Path(st.session_state.get("report_name", "report")).stem
    original_pdf = st.session_state.get("report_pdf_bytes")
    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        try:
            st.download_button(
                "Download report (PDF)",
                data=build_pdf(report),
                file_name=f"paperguard_report_{stem}.pdf",
                mime="application/pdf",
            )
        except Exception as exc:  # noqa: BLE001
            st.caption(f"PDF export unavailable: {exc}")
    with dl2:
        st.download_button(
            "Download full report (JSON)",
            data=json.dumps({k: v for k, v in report.items() if k != "_source_text"}, indent=2, ensure_ascii=False),
            file_name=f"paperguard_report_{stem}.json",
            mime="application/json",
        )
    with dl3:
        if original_pdf:
            try:
                annotated, hl_stats = build_annotated_pdf(report, original_pdf)
                st.download_button(
                    "Download annotated PDF (highlighted)",
                    data=annotated,
                    file_name=f"paperguard_annotated_{stem}.pdf",
                    mime="application/pdf",
                    help=f"Highlighted {hl_stats['highlighted']}/{hl_stats['requested']} flagged spans "
                         f"directly on the original PDF ({hl_stats['not_found']} could not be located, "
                         "usually due to PDF text reflow/hyphenation).",
                )
            except Exception as exc:  # noqa: BLE001
                st.caption(f"Annotated PDF unavailable: {exc}")
        else:
            st.caption("Annotated PDF only available for PDF uploads.")

    st.divider()
    st.caption(
        "Disclaimer: PaperGuard is a pre-submission self-check, not a definitive verdict. "
        "AI detection and plagiarism results are probabilistic indicators. Plagiarism coverage "
        "is limited to open web and open-access scholarly sources (not Turnitin's private database)."
    )
else:
    st.info("Upload a paper (PDF, Markdown, or text) and click **Analyze paper** to begin.")
