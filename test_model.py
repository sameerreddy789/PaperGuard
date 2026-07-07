import os
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Test Data
human_text = """
The study of quantum mechanics has long been a foundational pillar of modern physics. 
In this paper, we explore the implications of entanglement entropy in bipartite systems. 
Our methodology relies on standard perturbative techniques, specifically focusing on the 
first-order corrections to the ground state energy. The results indicate a strong correlation 
between the spatial separation of the subsystems and the decay rate of the entropy.
"""

ai_text = """
Quantum mechanics is a really fascinating topic that scientists study a lot. It is super 
important for physics. In this essay, I will talk about how things get entangled and what 
that means for energy. We used some math to look at the ground state and found out that 
when things are farther apart, the energy changes. This is a very interesting discovery 
that will change how we see the universe. In conclusion, quantum physics is great.
"""

def test_local_model():
    print("\n" + "="*50)
    print("Testing LOCAL Model (D:\\Hackathons\\1\\training\\ai_detector_final)")
    print("="*50)
    
    model_path = r"D:\Hackathons\1\training\ai_detector_final"
    
    if not os.path.exists(model_path):
        print("❌ Local model not found!")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    for name, text in [("Human-written Academic Text", human_text), ("AI-generated Generic Text", ai_text)]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        # model has labels: 0 -> ai, 1 -> human (based on our training mapping)
        ai_prob = probs[0][0].item()
        human_prob = probs[0][1].item()
        
        print(f"\n--- {name} ---")
        print(f"AI Probability:    {ai_prob*100:.2f}%")
        print(f"Human Probability: {human_prob*100:.2f}%")

def test_huggingface_api():
    print("\n" + "="*50)
    print("Testing HUGGING FACE API (vediumsameer/paperguard-ai-detector)")
    print("="*50)
    
    API_URL = "https://api-inference.huggingface.co/models/vediumsameer/paperguard-ai-detector"
    # Token is read from the HF_TOKEN environment variable (never hardcode secrets).
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("\n❌ HF_TOKEN environment variable not set. Set it before running:")
        print("   Windows (cmd):  set HF_TOKEN=your_token_here")
        print("   PowerShell:     $env:HF_TOKEN=\"your_token_here\"")
        return
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    for name, text in [("Human-written Academic Text", human_text), ("AI-generated Generic Text", ai_text)]:
        response = requests.post(API_URL, headers=headers, json={"inputs": text})
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n--- {name} ---")
            for score_dict in result[0]:
                print(f"{score_dict['label']} Probability: {score_dict['score']*100:.2f}%")
        elif "is currently loading" in response.text:
            print(f"\n⚠️ The API model is still booting up (cold start). Try again in 30 seconds.")
            print(response.json())
        else:
            print(f"\n❌ API Error: {response.status_code}")
            print(response.text)

if __name__ == "__main__":
    test_local_model()
    test_huggingface_api()
