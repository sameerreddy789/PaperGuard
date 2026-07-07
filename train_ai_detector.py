"""
PaperGuard AI Detection Model - Fine-tuning Script
Fine-tunes dima806/ai-generated-essay-detection-distilbert on 
silentone0725/ai-human-text-detection-v1 dataset.

Run: python train_ai_detector.py
"""

# ============================================================
# FORCE ALL CACHES TO D DRIVE (must be before any imports)
# ============================================================
import os
os.environ["HF_HOME"] = r"D:\Hackathons\1\training\hf_cache"
os.environ["HF_DATASETS_CACHE"] = r"D:\Hackathons\1\training\hf_cache\datasets"
os.environ["TRANSFORMERS_CACHE"] = r"D:\Hackathons\1\training\hf_cache\models"
os.environ["TORCH_HOME"] = r"D:\Hackathons\1\training\hf_cache\torch"
os.environ["PIP_CACHE_DIR"] = r"D:\Hackathons\1\training\pip_cache"

import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ============================================================
# CONFIG
# ============================================================
BASE_MODEL = "dima806/ai-generated-essay-detection-distilbert"
DATASET_NAME = "silentone0725/ai-human-text-detection-v1"
OUTPUT_DIR = "./training/ai_detector_output"
FINAL_MODEL_DIR = "./training/ai_detector_final"
MAX_LENGTH = 512  # DistilBERT max token length
BATCH_SIZE = 8    # Safe for 6GB VRAM with fp16
EPOCHS = 3
LEARNING_RATE = 2e-5

# ============================================================
# CHECK GPU
# ============================================================
print("=" * 60)
print("PaperGuard AI Detector - Training Script")
print("=" * 60)

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"✅ GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
else:
    print("⚠️  No GPU detected. Training on CPU (will be very slow).")

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# LOAD DATASET
# ============================================================
print("\n📦 Loading dataset...")
dataset = load_dataset(DATASET_NAME)
print(f"   Train:      {len(dataset['train']):,} samples")
print(f"   Validation: {len(dataset['validation']):,} samples")
print(f"   Test:       {len(dataset['test']):,} samples")

# Check label distribution
train_labels = dataset["train"]["label"]
unique_labels = sorted(set(train_labels))
print(f"   Labels:     {unique_labels}")
for label in unique_labels:
    count = train_labels.count(label)
    print(f"     - '{label}': {count:,} ({count/len(train_labels)*100:.1f}%)")

# ============================================================
# BUILD LABEL MAPPING
# ============================================================
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}
num_labels = len(unique_labels)
print(f"\n🏷️  Label mapping: {label2id}")

# ============================================================
# LOAD TOKENIZER & MODEL
# ============================================================
print(f"\n🤖 Loading base model: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,  # In case label count differs from original
)
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# TOKENIZE DATASET
# ============================================================
print("\n[INFO] Tokenizing dataset...")

def tokenize_fn(examples):
    # Ensure all inputs are strings (handles potential None values)
    texts = [str(text) if text is not None else "" for text in examples["text"]]
    tokens = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    # Convert string labels to integers
    tokens["labels"] = [label2id[l] for l in examples["label"]]
    return tokens

# Filter out None values just in case
dataset = dataset.filter(lambda x: x["text"] is not None and x["label"] is not None)

tokenized_dataset = dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=["text", "label"],
    desc="Tokenizing",
)
tokenized_dataset.set_format("torch")
print("   Done.")

# ============================================================
# METRICS
# ============================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

# ============================================================
# TRAINING ARGS
# ============================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,  # Eval can use bigger batches
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),  # Half precision to save VRAM
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,  # Only keep 2 best checkpoints to save disk
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=100,
    logging_dir=f"{OUTPUT_DIR}/logs",
    report_to="none",  # No wandb/tensorboard
    dataloader_num_workers=0,  # Windows compatibility
    warmup_ratio=0.1,
)

# ============================================================
# TRAINER
# ============================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# ============================================================
# TRAIN
# ============================================================
print("\n" + "=" * 60)
print("🚀 STARTING TRAINING")
print(f"   Epochs:        {EPOCHS}")
print(f"   Batch size:    {BATCH_SIZE}")
print(f"   Learning rate: {LEARNING_RATE}")
print(f"   FP16:          {torch.cuda.is_available()}")
print(f"   Device:        {device}")
print("=" * 60 + "\n")

train_result = trainer.train()

print("\n✅ Training complete!")
print(f"   Training loss:  {train_result.training_loss:.4f}")

# ============================================================
# EVALUATE ON TEST SET
# ============================================================
print("\n📊 Evaluating on test set...")
test_results = trainer.evaluate(tokenized_dataset["test"])
print(f"   Test Accuracy: {test_results['eval_accuracy']:.4f}")
print(f"   Test F1:       {test_results['eval_f1']:.4f}")

# Detailed classification report
print("\n📋 Detailed Classification Report:")
test_preds = trainer.predict(tokenized_dataset["test"])
pred_labels = np.argmax(test_preds.predictions, axis=-1)
true_labels = test_preds.label_ids
print(classification_report(true_labels, pred_labels, target_names=[id2label[i] for i in range(num_labels)]))

# ============================================================
# SAVE FINAL MODEL
# ============================================================
print(f"\n💾 Saving final model to {FINAL_MODEL_DIR}...")
trainer.save_model(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)

# Save label mapping for inference
import json
with open(os.path.join(FINAL_MODEL_DIR, "label_mapping.json"), "w") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

print("\n" + "=" * 60)
print("🎉 ALL DONE!")
print(f"   Model saved to: {FINAL_MODEL_DIR}")
print(f"   Test Accuracy:  {test_results['eval_accuracy']:.4f}")
print(f"   Test F1:        {test_results['eval_f1']:.4f}")
print("=" * 60)
print("\nNext steps:")
print("  1. Upload to Hugging Face:  huggingface-cli upload sameerreddy789/paperguard-ai-detector ./training/ai_detector_final")
print("  2. Or use locally in PaperGuard's AI detection agent")
