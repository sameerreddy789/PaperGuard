"""
Provider-agnostic LLM selector for PaperGuard's sub-agent calls (citation claim
verification, plagiarism similarity, quality prose review, reference parsing).

This is the single place that decides "Gemini or Qwen" for those calls. The
CrewAI crew-level LLM (agents/orchestrator.py) is already provider-agnostic via
LiteLLM/CrewAI's own ``LLM`` class and ``PAPERGUARD_CREW_MODEL`` -- this module
closes the matching gap for the *sub-agent* calls, which previously only had a
Gemini backend (``services.gemini``).

Selection (``PAPERGUARD_LLM_PROVIDER``):
    "gemini"    -> always use services.gemini
    "dashscope" -> always use services.dashscope_llm (Qwen)
    unset/"auto" (default) -> prefer DASHSCOPE_API_KEY if set and GEMINI_API_KEY
                              is not (so a pure-Alibaba deployment "just works"
                              by setting only DASHSCOPE_API_KEY); otherwise
                              Gemini if its key is set; otherwise whichever
                              module is importable (both degrade to None calls
                              gracefully if no key is present).

Public interface matches both backends exactly: ``call_llm`` and
``call_llm_json`` module-level functions, so ``get_llm()`` callers can treat
whatever this returns as an interchangeable "the LLM module" of the active
provider -- no caller changes are needed anywhere in the agent code.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional


def _provider() -> str:
    return os.getenv("PAPERGUARD_LLM_PROVIDER", "auto").strip().lower()


def _select_module():
    """Return the active backend module (services.gemini or services.dashscope_llm)."""
    provider = _provider()

    if provider == "gemini":
        from services import gemini
        return gemini
    if provider == "dashscope":
        from services import dashscope_llm
        return dashscope_llm

    # auto: prefer whichever key is actually configured. If both or neither
    # are set, Gemini stays the default (matches the project's existing
    # behavior/docs) unless only DashScope is configured.
    has_gemini = bool(os.getenv("GEMINI_API_KEY"))
    has_dashscope = bool(os.getenv("DASHSCOPE_API_KEY"))
    if has_dashscope and not has_gemini:
        from services import dashscope_llm
        return dashscope_llm
    from services import gemini
    return gemini


def active_provider_name() -> str:
    """Human-readable name of the currently-active backend, for logging/UI."""
    mod = _select_module()
    return "dashscope" if mod.__name__.endswith("dashscope_llm") else "gemini"


def call_llm(
    prompt: str,
    system_instruction: Optional[str] = None,
    response_format: Optional[str] = None,
) -> Optional[str]:
    """Delegate to the active provider's ``call_llm`` (same signature/contract)."""
    try:
        mod = _select_module()
    except Exception:
        return None
    return mod.call_llm(prompt, system_instruction=system_instruction, response_format=response_format)


def call_llm_json(prompt: str, system_instruction: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Delegate to the active provider's ``call_llm_json`` (same signature/contract)."""
    try:
        mod = _select_module()
    except Exception:
        return None
    return mod.call_llm_json(prompt, system_instruction=system_instruction)
