import os
import torch
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Set Hugging Face token from environment (required for gated datasets)
hf_token = os.getenv("HF_TOKEN")

print("============================================================")
print("PaperGuard v2.0 MEGA-DATASET TRAINING (Opus + MGTPD Hybrid)")
print("============================================================")

model_name = "vediumsameer/paperguard-ai-detector"  # Starting from our v1.5 base!
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

datasets_to_merge = []

# 1. Claude Opus 4.8 Distill (Captures the newest Anthropic models)
print("📦 Loading Claude Opus 4.8 Distill (5k samples)...")
try:
    ds_opus = load_dataset("11-47/claude_opus_4.8_distill_5k", split="train")
    # Opus dataset has 'instruction' and 'response'. All responses are AI generated.
    def map_opus(example):
        return {"text": str(example["response"]), "label": 0} # 0 = AI
    ds_opus = ds_opus.map(map_opus, remove_columns=ds_opus.column_names)
    datasets_to_merge.append(ds_opus)
except Exception as e:
    print(f"⚠️ Failed to load Opus dataset: {e}")

# 2. Ateeqq AI vs Human (Reduces student false positives)
print("📦 Loading Ateeqq Academic (6k samples)...")
try:
    ds_ateeqq = load_dataset("Ateeqq/AI-and-Human-Generated-Text", split="train")
    def map_ateeqq(example):
        # Ateeqq label: 0=Human, 1=AI. We swap to match our standard: 0=AI, 1=Human
        return {"text": str(example["abstract"]), "label": 1 if example["label"] == 0 else 0}
    ds_ateeqq = ds_ateeqq.map(map_ateeqq, remove_columns=ds_ateeqq.column_names)
    datasets_to_merge.append(ds_ateeqq)
except Exception as e:
    print(f"⚠️ Failed to load Ateeqq dataset: {e}")

# 3. AI Text Detection Pile (1.39 Million samples, Open Alternative)
print("📦 Loading AI Text Detection Pile (1.39M samples)...")
try:
    # We load only the first 250k rows directly to prevent memory deadlock
    ds_pile = load_dataset("artem9k/ai-text-detection-pile", split="train[:250000]")
    ds_pile = ds_pile.shuffle(seed=42)
    
    def map_pile(example):
        # The AI Text Detection Pile uses 'text' and 'source' or 'label'.
        # We enforce our standard: 0=AI, 1=Human
        text = str(example.get("text", example.get("Text", "")))
        label = example.get("label", example.get("source", 0))
        
        if isinstance(label, str):
            label_lower = label.lower().strip()
            final_label = 1 if label_lower in ["human", "0", "0.0"] else 0
        else:
            final_label = 1 if int(label) == 0 else 0
            
        return {"text": text, "label": final_label}
        
    # We don't remove_columns aggressively here if we aren't 100% sure of the schema, 
    # but map handles it safely. We will remove columns by name dynamically.
    columns_to_remove = ds_pile.column_names
    ds_pile = ds_pile.map(map_pile, remove_columns=columns_to_remove)
    datasets_to_merge.append(ds_pile)
except Exception as e:
    print(f"❌ CRITICAL ERROR loading AI Text Detection Pile: {e}")
    exit(1)

# Merge everything
print("🔄 Merging mega-dataset...")
final_dataset = concatenate_datasets(datasets_to_merge)
final_dataset = final_dataset.shuffle(seed=42)

print(f"✅ Final Mega-Dataset Size: {len(final_dataset)} rows")

# Split train/test (90/10)
split_ds = final_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_ds["train"]
test_dataset = split_ds["test"]

print("⚙️ Tokenizing (this will take a few minutes)...")
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

print("🚀 Initializing v2.0 Model...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=2, 
    ignore_mismatched_sizes=True
)

training_args = TrainingArguments(
    output_dir="./training/mega_dataset_output",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=2, # 2 epochs of 250k is ~13 hours on RTX 3050
    weight_decay=0.01,
    logging_steps=100,
    fp16=True,  # Crucial for RTX 3050 speed
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
)

print("🔥 STARTING 13-HOUR MEGA TRAINING RUN 🔥")
trainer.train()

print("✅ Training Complete!")
eval_results = trainer.evaluate()
print(f"📊 Final Evaluation: {eval_results}")

print("💾 Saving v2.0 Model locally...")
trainer.save_model("./training/mega_dataset_model_v2")
tokenizer.save_pretrained("./training/mega_dataset_model_v2")

print("☁️ Uploading v2.0 to Hugging Face...")
from huggingface_hub import HfApi
api = HfApi()
try:
    api.upload_folder(
        folder_path="./training/mega_dataset_model_v2",
        repo_id="vediumsameer/paperguard-v2-mega",
        commit_message=f"Upload v2.0 Mega-Dataset model (Opus + Hybrid). Eval: {eval_results}"
    )
    print("✅ Successfully pushed to Hugging Face!")
except Exception as e:
    print(f"⚠️ Failed to push to Hugging Face: {e}")
    print("You can push manually later.")

print("============================================================")
print("🎉 v2.0 PIPELINE FINISHED.")
print("============================================================")
