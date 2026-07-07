import json
from typing import Dict, Any

class ConflictResolver:
    """
    Track 3 Hero Architecture: The Conflict Resolution Protocol.
    When the raw mathematical PyTorch model fails (Logit Saturation / Mode Collapse),
    the Linguistic Agent steps in to override the false negatives/positives.
    """
    
    def __init__(self, conflict_threshold: float = 30.0):
        # If the difference in AI probability between agents is > 30%, it triggers a conflict.
        self.conflict_threshold = conflict_threshold

    def resolve(self, detector_payload: Dict[str, Any], linguistic_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes the raw mathematical output from the PyTorch model and the contextual analysis
        from the LLM, and forces a consensual override if a conflict is detected.
        """
        pytorch_ai_prob = detector_payload.get("ai_probability", 0.0)
        llm_ai_prob = linguistic_payload.get("ai_probability", 0.0)
        llm_reasoning = linguistic_payload.get("reasoning", "No context provided.")
        
        difference = abs(pytorch_ai_prob - llm_ai_prob)
        
        final_verdict = {
            "conflict_detected": False,
            "original_pytorch_score": pytorch_ai_prob,
            "original_llm_score": llm_ai_prob,
            "final_consensus_score": 0.0,
            "resolution_reasoning": "Agents are in agreement. No override necessary."
        }

        # --- THE LINGUISTIC OVERRIDE LOOP (Track 3 Core Feature) ---
        if difference >= self.conflict_threshold:
            final_verdict["conflict_detected"] = True
            
            # Scenario A: The Logit Saturation (The 100% Human Bug)
            if pytorch_ai_prob < 10.0 and llm_ai_prob > 60.0:
                final_verdict["final_consensus_score"] = llm_ai_prob
                final_verdict["resolution_reasoning"] = (
                    "OVERRIDE ENGAGED: PyTorch model exhibited severe logit saturation (mode collapse), "
                    "flagging synthetic text as human. The Linguistic Agent successfully intercepted the "
                    f"structural anomalies. LLM Reasoning: {llm_reasoning}"
                )
                
            # Scenario B: The ESL False Positive (The Human Penalty)
            elif pytorch_ai_prob > 80.0 and llm_ai_prob < 30.0:
                final_verdict["final_consensus_score"] = llm_ai_prob
                final_verdict["resolution_reasoning"] = (
                    "OVERRIDE ENGAGED: PyTorch model falsely flagged non-native (ESL) human phrasing "
                    "as AI-generated due to rigid syntax. Linguistic Agent verified human semantic intent. "
                    f"LLM Reasoning: {llm_reasoning}"
                )
                
            # Scenario C: Unresolved / Balanced Compromise
            else:
                # If neither extreme is met, the Orchestrator calculates a weighted average.
                # We weight the Linguistic Agent slightly higher (60/40) for safety.
                final_consensus = (pytorch_ai_prob * 0.4) + (llm_ai_prob * 0.6)
                final_verdict["final_consensus_score"] = final_consensus
                final_verdict["resolution_reasoning"] = (
                    "COMPROMISE REACHED: Agents strongly disagreed, but neither triggered extreme bounds. "
                    "A 60/40 weighted consensus was applied in favor of the contextual LLM."
                )
        else:
            # If they agree (difference < 30%), we just take the PyTorch math as primary.
            final_verdict["final_consensus_score"] = pytorch_ai_prob
            
        return final_verdict


# === PROMPT LOGIC FOR THE LINGUISTIC AGENT ===
LINGUISTIC_AGENT_PROMPT = """
You are the Linguistic Analysis Agent in a Multi-Agent Society designed to detect academic fraud.
Another agent (a PyTorch Sequence Classifier) has analyzed this text block mathematically, but it has severe blind spots regarding non-native ESL writers, heavily formatted text, and stylistic masking (like sarcasm).

Your task is to analyze the semantic intent, tone, and structural flow of the text to prevent false positives and false negatives. 

Look specifically for:
1. "Patchwriting" or "Frankensteining" (Abrupt shifts between robotic and casual tones).
2. Advanced AI masking (ChatGPT commanded to use slang, lowercase letters, or sarcasm).
3. "The ESL Penalty" (Rigid, textbook-perfect academic transitions written by a non-native human).

Return a JSON payload containing:
{
    "ai_probability": <float between 0.0 and 100.0>,
    "reasoning": "<1 sentence justifying your score based on structural and semantic markers>"
}
"""
