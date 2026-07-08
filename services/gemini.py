"""
Gemini LLM wrapper (google-genai SDK).

Public interface is unchanged: ``call_llm`` and ``call_llm_json``. Uses the
current ``google-genai`` client (the older ``google-generativeai`` package is
end-of-life). The model name is read from ``PAPERGUARD_GEMINI_MODEL`` (default
``gemini-2.5-flash``) so it stays consistent across the codebase.

The client is created lazily and re-created if the API key changes at runtime
(the Streamlit UI can set the key mid-session).
"""

import os
import json
from typing import Dict, Any, Optional

from google import genai
from google.genai import types

_client = None
_client_key: Optional[str] = None


def _get_client():
    """Return a cached genai.Client, (re)creating it if the API key changed."""
    global _client, _client_key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    if _client is None or _client_key != api_key:
        try:
            _client = genai.Client(api_key=api_key)
            _client_key = api_key
        except Exception as e:
            print(f"Failed to initialize Gemini client: {e}")
            return None
    return _client


def _model_name() -> str:
    return os.getenv("PAPERGUARD_GEMINI_MODEL", "gemini-2.5-flash")


def call_llm(
    prompt: str,
    system_instruction: Optional[str] = None,
    response_format: Optional[str] = None,
) -> Optional[str]:
    """
    Call Gemini and return the response text (or None on failure/no key).

    Args:
        prompt: The main user prompt.
        system_instruction: Optional system instructions to guide behaviour.
        response_format: If "json", requests a JSON response mime type.
    """
    client = _get_client()
    if client is None:
        return None

    config_kwargs: Dict[str, Any] = {"temperature": 0.1}
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction
    if response_format == "json":
        config_kwargs["response_mime_type"] = "application/json"

    try:
        response = client.models.generate_content(
            model=_model_name(),
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        return response.text
    except Exception as e:
        print(f"Gemini API error: {e}")
        return None


def call_llm_json(prompt: str, system_instruction: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Call Gemini and parse the response as JSON."""
    response_text = call_llm(
        prompt=prompt,
        system_instruction=system_instruction,
        response_format="json",
    )
    if not response_text:
        return None

    try:
        cleaned_text = response_text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        return json.loads(cleaned_text.strip())
    except json.JSONDecodeError as e:
        print(f"Failed to parse LLM JSON response: {e}\nResponse was: {response_text}")
        return None


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    print("Testing Gemini API...")
    res = call_llm("Explain what PaperGuard is in one sentence.", system_instruction="You are a helpful AI.")
    print(f"Result: {res}")
