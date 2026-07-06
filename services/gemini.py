import os
import json
from typing import Dict, Any, Optional, List
import google.generativeai as genai

# Try to initialize the API, but don't fail immediately if key is missing 
# (allows importing the module and checking later)
_initialized = False

def _init_api():
    global _initialized
    if not _initialized:
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            _initialized = True
        else:
            print("Warning: GEMINI_API_KEY not found in environment.")

def call_llm(prompt: str, system_instruction: Optional[str] = None, response_format: Optional[str] = None) -> Optional[str]:
    """
    Call Gemini 3.1 Flash Lite API.
    
    Args:
        prompt: The main user prompt.
        system_instruction: Optional system instructions to guide behavior.
        response_format: If "json", asks the model to return structured JSON.
    """
    _init_api()
    if not _initialized:
        return None
        
    # According to plan, we use gemini-3.1-flash-lite (or fallback to latest flash)
    model_name = "gemini-2.5-flash" # Note: using 2.5-flash as 3.1-flash-lite may not be universally available in all SDKs yet, adapt as needed.
    
    # Configure generation
    generation_config = genai.types.GenerationConfig(
        temperature=0.1,  # Low temperature for deterministic output
    )
    
    # If JSON is requested, we can use response_mime_type in newer SDKs, 
    # but prompting usually works well too.
    if response_format == "json":
        generation_config.response_mime_type = "application/json"
        
    try:
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction
        )
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        return response.text
    except Exception as e:
        print(f"Gemini API error: {e}")
        return None

def call_llm_json(prompt: str, system_instruction: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Call Gemini and parse the response as JSON.
    """
    response_text = call_llm(
        prompt=prompt,
        system_instruction=system_instruction,
        response_format="json"
    )
    
    if not response_text:
        return None
        
    try:
        # Strip potential markdown formatting if the API returned it despite MIME type
        cleaned_text = response_text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
            
        return json.loads(cleaned_text.strip())
    except json.JSONDecodeError as e:
        print(f"Failed to parse LLM JSON response: {e}\nResponse was: {response_text}")
        return None

if __name__ == "__main__":
    import sys
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
        
    print("Testing Gemini API...")
    res = call_llm("Explain what PaperGuard is in one sentence.", system_instruction="You are a helpful AI.")
    print(f"Result: {res}")
