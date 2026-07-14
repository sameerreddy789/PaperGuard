"""
Qwen (Alibaba DashScope) LLM wrapper -- OpenAI-compatible endpoint.

Same public interface as ``services.gemini``: ``call_llm`` and
``call_llm_json``, so every existing caller (``agents.base.get_llm()`` /
``services.llm``) works with either backend without changes.

DashScope exposes Qwen models via an OpenAI-compatible Chat Completions API
(https://dashscope.aliyuncs.com/compatible-mode/v1 for mainland China,
https://dashscope-intl.aliyuncs.com/compatible-mode/v1 international), so this
uses the standard ``openai`` Python client pointed at that base URL rather than
a bespoke HTTP client -- it's the officially documented integration path and
means we inherit the client's retry/error handling for free.

Env vars:
    DASHSCOPE_API_KEY              required
    PAPERGUARD_DASHSCOPE_MODEL      default "qwen-plus"
    PAPERGUARD_DASHSCOPE_API_BASE   default the international compatible-mode
                                     endpoint (override for the mainland one)
    PAPERGUARD_LLM_TEMPERATURE       shared with services.gemini, default 0.0
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional

_DEFAULT_API_BASE = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

_client = None
_client_key: Optional[str] = None


def _get_client():
    """Return a cached OpenAI-compatible client, (re)creating it if the key changed."""
    global _client, _client_key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return None
    if _client is None or _client_key != api_key:
        try:
            from openai import OpenAI

            base_url = os.getenv("PAPERGUARD_DASHSCOPE_API_BASE", _DEFAULT_API_BASE)
            _client = OpenAI(api_key=api_key, base_url=base_url)
            _client_key = api_key
        except Exception as e:  # noqa: BLE001
            print(f"Failed to initialize DashScope (Qwen) client: {e}")
            return None
    return _client


def _model_name() -> str:
    return os.getenv("PAPERGUARD_DASHSCOPE_MODEL", "qwen-plus")


def call_llm(
    prompt: str,
    system_instruction: Optional[str] = None,
    response_format: Optional[str] = None,
) -> Optional[str]:
    """
    Call Qwen (via DashScope's OpenAI-compatible endpoint) and return the
    response text, or ``None`` on failure/no key. Mirrors
    ``services.gemini.call_llm`` exactly so callers are provider-agnostic.
    """
    client = _get_client()
    if client is None:
        return None

    try:
        temperature = float(os.getenv("PAPERGUARD_LLM_TEMPERATURE", "0.0"))
    except ValueError:
        temperature = 0.0

    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})

    kwargs: Dict[str, Any] = {
        "model": _model_name(),
        "messages": messages,
        "temperature": temperature,
    }
    if response_format == "json":
        # Qwen (OpenAI-compatible mode) supports the standard JSON-mode flag.
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    except Exception as e:  # noqa: BLE001
        # Some Qwen models/regions may reject response_format; retry once
        # without it rather than failing outright (call_llm_json still strips
        # markdown fences, so plain-text JSON-ish output is still usable).
        if response_format == "json" and "response_format" in kwargs:
            try:
                kwargs.pop("response_format")
                response = client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
            except Exception as e2:  # noqa: BLE001
                print(f"DashScope (Qwen) API error: {e2}")
                return None
        print(f"DashScope (Qwen) API error: {e}")
        return None


def call_llm_json(prompt: str, system_instruction: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Call Qwen and parse the response as JSON (identical contract to services.gemini)."""
    response_text = call_llm(prompt=prompt, system_instruction=system_instruction, response_format="json")
    if not response_text:
        return None
    try:
        cleaned = response_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return json.loads(cleaned.strip())
    except json.JSONDecodeError as e:
        print(f"Failed to parse Qwen JSON response: {e}\nResponse was: {response_text}")
        return None


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    print("Testing DashScope (Qwen) API...")
    res = call_llm("Explain what PaperGuard is in one sentence.", system_instruction="You are a helpful AI.")
    print(f"Result: {res}")
