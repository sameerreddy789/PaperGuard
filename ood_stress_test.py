import torch
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the EXACT final v2.0 checkpoint we just finished training!
CHECKPOINT_PATH = "./training/mega_dataset_model_v2"

def load_model():
    print(f"Loading final v2.0 checkpoint: {CHECKPOINT_PATH}...")
    # Load tokenizer from local path just to be safe
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_PATH)
    return tokenizer, model

def predict_text(tokenizer, model, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # In our dataset, 0 = AI, 1 = HUMAN
    ai_prob = predictions[0][0].item() * 100
    human_prob = predictions[0][1].item() * 100
    return human_prob, ai_prob

# === OOD STRESS TEST GAUNTLET ===
test_cases = [
    {
        "name": "The ESL / Non-Native Human Essay",
        "description": "It intentionally uses rigid, textbook-perfect transitions and slightly unnatural phrasing common to non-native writers.",
        "text": "Furthermore, it is very important to discuss about how technologies change the modern schools. In my country, many students do not have high-speed internet in their houses, consequently they struggle to finish their homework assignments on time. Moreover, teachers try their best to teach with old books because buying computers is too much expensive for public schools. In conclusion, we must understand that technology can help education only if all students can access it equally, otherwise the gap between rich and poor people will become wider.",
        "expected": "HUMAN"
    },
    {
        "name": "The 'Sarcastic Student' AI Prompt",
        "description": "ChatGPT was explicitly commanded to use casual slang, abbreviations, and a frustrated tone to mask its default structured voice.",
        "text": "so basicly macbeth is completely losing his mind after he listens to those random witches. like honestly, why would u kill the king just cuz some spooky lady in a swamp said u would become the boss? absolute clown behavior imo. lady macbeth is even worse tho, she keeps gaslighting him into going through with it and then acts all shocked when she cant wash the literal imaginary blood off her hands. ngl the whole play is just a massive warning sign about what happens when u let ambition completely ruin ur vibe.",
        "expected": "AI"
    },
    {
        "name": "The Claude 3.5 Opus 'Heavy Academic'",
        "description": "This block mimics dense, ultra-sophisticated AI prose that avoids cliché words like 'delve' or 'testament.'",
        "text": "The architectural integrity of early Byzantine fortifications reflects a meticulous synthesis of strategic geographical placement and empirical engineering principles. Rather than relying solely on raw structural mass, regional architects implemented a system of alternating brick courses and stone facings, specifically designed to dissipate the kinetic energy of seismic disruptions and primitive siege mechanics. This defensive paradigm underscores a deeper sociopolitical imperative, wherein the preservation of urban granaries directly dictated the long-term survival vectors of peripheral imperial outposts.",
        "expected": "AI"
    },
    {
        "name": "The 'Frankenstein' Patchwork Essay",
        "description": "This text alternates between pure AI text generation and raw human writing.",
        "text": "[PART 1 - HUMAN WRITING]\ni went to the museum last week to see the ancient roman exhibit for my history project. honestly it was kind of small and crowded but seeing the actual rusty swords and old coins was pretty cool. you can tell those soldiers lived a super rough life just by looking at how heavy their chest armor was.\n\n[PART 2 - COPY-PASTED AI WRITING]\nFurthermore, it is essential to analyze the metallurgical composition of these artifacts. The Roman military-industrial apparatus relied heavily on standardized carbon-bonding techniques to mass-produce resilient weaponry, thereby ensuring tactical superiority over disorganized tribal factions. Consequently, these preservation efforts serve as a profound testament to classical engineering prowess.",
        "expected": "MIXED"
    }
]

if __name__ == "__main__":
    print("🚀 BOOTING UP CLEAN OOD STRESS TEST GAUNTLET 🚀")
    tokenizer, model = load_model()
    print("✅ Model loaded successfully! Commencing Stress Test...\n")
    
    for idx, case in enumerate(test_cases, 1):
        print(f"--- TEST {idx}: {case['name']} ---")
        print(f"Description: {case['description']}")
        print(f"Expected: {case['expected']}")
        
        start_time = time.time()
        human_prob, ai_prob = predict_text(tokenizer, model, case['text'])
        elapsed = time.time() - start_time
        
        print(f"Prediction Time: {elapsed:.3f} seconds")
        print(f"➡️  AI Probability:    {ai_prob:.2f}%")
        print(f"➡️  Human Probability: {human_prob:.2f}%\n")
        
        # Simple visual Pass/Fail
        if case['expected'] == "MIXED":
            print("🟡 RESULT: MIXED (Check Heatmapping)\n")
        elif (case['expected'] == "AI" and ai_prob > 50) or (case['expected'] == "HUMAN" and human_prob > 50):
            print("🟢 RESULT: PASS (Model nailed it!)\n")
        else:
            print("🔴 RESULT: FAIL (Model was tricked!)\n")
        
        time.sleep(1) # Little pause for dramatic effect in the console
