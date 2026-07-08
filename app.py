"""
PaperGuard -- Streamlit UI.

Upload an academic paper (PDF/MD/TXT) and get a multi-agent integrity report:
per-paragraph AI-detection heatmap (with safety-net override reasoning),
citation verification, plagiarism scan, and writing-quality review.

Run:  streamlit run app.py
"""

from __future__ import annotations

import hashlib
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
    """Map paragraph_index -> full paragraph text, matching the safety-net selection."""
    try:
        from agents.safety_net import _select_paragraphs
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
def render_metrics(report: Dict[str, Any]) -> None:
    ai = _meta(report, "AIDetectionSafetyNet")
    cite = _meta(report, "CitationAgent")
    plag = _meta(report, "PlagiarismAgent")
    qual = _meta(report, "QualityAgent")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AI Content", _fmt(ai.get("overall_ai_score"), "%"),
              ai.get("classification") or "")
    c2.metric("Citation Health", _fmt(cite.get("citation_health_score"), "%"),
              f"{cite.get('not_found_count', 0)} not found")
    c3.metric("Plagiarism", _fmt(plag.get("plagiarism_score"), "%"),
              f"{plag.get('flagged_paragraph_count', 0)} flagged")
    c4.metric("Writing Quality", _fmt(qual.get("overall_quality_score"), "/10"))


def render_heatmap(report: Dict[str, Any]) -> None:
    ai = _meta(report, "AIDetectionSafetyNet")
    heatmap: List[Dict[str, Any]] = ai.get("heatmap") or []
    comp = ai.get("components") or {}

    st.markdown(
        '<div class="pg-legend">'
        '<span style="background:#FEE2E2;color:#991B1B;">Likely AI (&ge;65%)</span>'
        '<span style="background:#FEF3C7;color:#92400E;">Uncertain</span>'
        '<span style="background:#DCFCE7;color:#166534;">Likely Human (&le;35%)</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Detector: {'on' if comp.get('detector_available') else 'off'} · "
        f"Linguistic (LLM): {'on' if comp.get('linguistic_available') else 'off'} · "
        f"Safety net: {'ACTIVE' if ai.get('safety_net_active') else 'inactive'} · "
        f"Overrides applied: {ai.get('overrides_applied', 0)}"
    )
    if not comp.get("linguistic_available"):
        st.warning(
            "Linguistic agent is off (no valid GEMINI_API_KEY). AI detection is "
            "running on the calibrated model alone, without the LLM safety net.",
            icon=None,
        )

    if not heatmap:
        st.info("No paragraph-level AI results available.")
        return

    fulltext = _full_paragraphs(report.get("_source_text", ""))

    for entry in heatmap:
        level = entry.get("heat_level", "unknown")
        c = _HEAT.get(level, _HEAT["unknown"])
        score = entry.get("final_ai_score")
        idx = entry.get("paragraph_index")
        body = fulltext.get(idx) or entry.get("text_preview") or ""
        det = entry.get("detector_score")
        lin = entry.get("linguistic_score")

        note = ""
        if entry.get("conflict_detected"):
            ov = entry.get("override_type") or "compromise"
            note = (
                f'<div class="pg-note">Safety-net override [{ov}] &mdash; '
                f'model {det}% vs context {lin}%. {entry.get("reasoning", "")}</div>'
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


def render_citations(report: Dict[str, Any]) -> None:
    meta = _meta(report, "CitationAgent")
    tiers = meta.get("tier_counts") or {}
    a, b, c, d = st.columns(4)
    a.metric("Verified", tiers.get("VERIFIED", 0))
    b.metric("Partial", tiers.get("PARTIALLY_VERIFIED", 0))
    c.metric("Existence only", tiers.get("EXISTENCE_ONLY", 0))
    d.metric("Not found", tiers.get("NOT_FOUND", 0))

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
# Sidebar
# --------------------------------------------------------------------------- #
_LOCAL_MODEL = r"training\mega_dataset_model_v2"
_default_model = _LOCAL_MODEL if Path(_LOCAL_MODEL).exists() else "vediumsameer/paperguard-ai-detector"

with st.sidebar:
    st.header("Settings")
    gemini_key = st.text_input("Gemini API key", type="password",
                               help="Enables the Linguistic agent + CrewAI crew. Without it, detection runs on the model alone.")
    serper_key = st.text_input("Serper API key (optional)", type="password",
                               help="Enables web plagiarism search.")
    model_path = st.text_input("Detector model", value=_default_model,
                               help="Local path or HuggingFace repo id.")
    use_crew = st.toggle("Use CrewAI multi-agent crew", value=True,
                         help="Off = deterministic engine path (faster, no LLM synthesis).")
    st.divider()
    st.caption("PaperGuard - multi-agent academic integrity verification.")

if gemini_key:
    os.environ["GEMINI_API_KEY"] = gemini_key.strip()
if serper_key:
    os.environ["SERPER_API_KEY"] = serper_key.strip()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
st.markdown('<p class="pg-title">PaperGuard</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="pg-sub">Multi-agent academic integrity verification &mdash; citations, '
    'AI-content detection with a self-correcting safety net, plagiarism, and writing quality.</p>',
    unsafe_allow_html=True,
)

uploaded = st.file_uploader("Upload a paper", type=["pdf", "md", "txt"])
run = st.button("Analyze paper", type="primary", disabled=uploaded is None)

if run and uploaded is not None:
    file_bytes = uploaded.getvalue()
    suffix = Path(uploaded.name).suffix or ".txt"
    cache_key = hashlib.sha256(
        file_bytes + f"|{use_crew}|{model_path}|{bool(gemini_key)}|{bool(serper_key)}".encode()
    ).hexdigest()
    with st.spinner("Running the agent society... this can take 1-3 minutes on first run."):
        try:
            st.session_state["report"] = _analyze(file_bytes, suffix, use_crew, model_path, cache_key)
            st.session_state["report_name"] = uploaded.name
        except Exception as exc:  # noqa: BLE001
            st.error(f"Analysis failed: {exc}")

report = st.session_state.get("report")
if report:
    st.divider()
    title = report.get("paper_title") or st.session_state.get("report_name", "")
    st.subheader(title)
    render_metrics(report)

    with st.container(border=True):
        st.markdown("**Executive summary**")
        st.write(report.get("summary", "(none)"))

    tabs = st.tabs(["AI Heatmap", "Citations", "Plagiarism", "Writing Quality", "References"])
    with tabs[0]:
        render_heatmap(report)
    with tabs[1]:
        render_citations(report)
    with tabs[2]:
        render_findings(report, "PlagiarismAgent", "No plagiarism results.")
    with tabs[3]:
        render_findings(report, "QualityAgent", "No writing-quality results.")
    with tabs[4]:
        render_references(report)

    st.download_button(
        "Download full report (JSON)",
        data=json.dumps({k: v for k, v in report.items() if k != "_source_text"}, indent=2, ensure_ascii=False),
        file_name=f"paperguard_report_{Path(st.session_state.get('report_name','report')).stem}.json",
        mime="application/json",
    )

    st.divider()
    st.caption(
        "Disclaimer: PaperGuard is a pre-submission self-check, not a definitive verdict. "
        "AI detection and plagiarism results are probabilistic indicators. Plagiarism coverage "
        "is limited to open web and open-access scholarly sources (not Turnitin's private database)."
    )
else:
    st.info("Upload a paper (PDF, Markdown, or text) and click **Analyze paper** to begin.")
