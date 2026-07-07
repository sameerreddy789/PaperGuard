"""
PaperGuard Multi-Dataset Training Script
Aggregates 4 different datasets covering all modern LLMs.
Trains locally, pushes to HF, and AUTO-DELETES local caches to save disk space.
"""

# ============================================================
# FORCE CACHES TO D DRIVE
# ============================================================
import os
import shutil

CACHE_DIR = r"D:\Hackathons\1\training\hf_cache"
FINAL_MODEL_DIR = r"D:\Hackathons\1\training\multi_dataset_model_final"

os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE_DIR, "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_DIR, "models")
os.environ["TORCH_HOME"] = os.path.join(CACHE_DIR, "torch")

import torch
import numpy as np
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score
from huggingface_hub import HfApi

# ============================================================
# CONFIG
# ============================================================
BASE_MODEL = "vediumsameer/paperguard-ai-detector" # Starting from where we left off
OUTPUT_DIR = r"D:\Hackathons\1\training\multi_dataset_output"
MAX_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5

print("\n" + "="*60)
print("PaperGuard Multi-Dataset Aggregation & Training")
print("="*60)

# ============================================================
# DATASET AGGREGATION & NORMALIZATION
# ============================================================
# Target schema: {"text": str, "label": int (0: AI, 1: Human)}

print("\n📦 Downloading and normalizing datasets (This might take a while)...")

datasets_to_merge = []

# 1. Defactify Text Dataset (GPT-4o, LLaMA-3, Mistral)
print("   -> Loading Defactify Dataset...")
ds1 = load_dataset("Rajarshi-Roy-research/Defactify_Text_Dataset", split="train")
# Label: 0 = Human, 1 = AI. We need to swap to match our convention (0=AI, 1=Human)
def map_ds1(example):
    return {"text": str(example["Text"]), "label": 1 if example["Label_A"] == 0 else 0}
ds1 = ds1.map(map_ds1, remove_columns=ds1.column_names)
datasets_to_merge.append(ds1)
# (HC3 removed due to Hugging Face deprecating Python loader scripts)

# 3. AI-and-Human-Generated-Text (GPT-3/4 Academic)
print("   -> Loading Academic Text Dataset...")
ds3 = load_dataset("Ateeqq/AI-and-Human-Generated-Text", split="train")
# Labels: 0 = Human, 1 = AI -> Swap to (0=AI, 1=Human)
def map_ds3(example):
    return {"text": str(example["abstract"]), "label": 1 if example["label"] == 0 else 0}
ds3 = ds3.map(map_ds3, remove_columns=ds3.column_names)
datasets_to_merge.append(ds3)

# ============================================================
# CONCATENATE & SHUFFLE
# ============================================================
print("\n🔗 Merging datasets...")
master_dataset = concatenate_datasets(datasets_to_merge).shuffle(seed=42)

# Filter out None or empty
master_dataset = master_dataset.filter(lambda x: x["text"] is not None and len(x["text"].strip()) > 10)

print(f"   Total Aggregated Samples: {len(master_dataset):,}")

# Split into Train/Val/Test
print("\n🔪 Splitting into Train/Val/Test...")
train_test = master_dataset.train_test_split(test_size=0.1)
val_test = train_test["test"].train_test_split(test_size=0.5)

dataset = DatasetDict({
    "train": train_test["train"],
    "validation": val_test["train"],
    "test": val_test["test"]
})

print(f"   Train: {len(dataset['train']):,}")
print(f"   Val:   {len(dataset['validation']):,}")
print(f"   Test:  {len(dataset['test']):,}")

# ============================================================
# LOAD MODEL & TOKENIZER
# ============================================================
print(f"\n🤖 Loading Base Model: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL)

# ============================================================
# TOKENIZATION
# ============================================================
print("\n[INFO] Tokenizing massive dataset...")
def tokenize_fn(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"], desc="Tokenizing")
tokenized_dataset.set_format("torch")

# ============================================================
# TRAINING
# ============================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds, average="weighted")}

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=1, # Keep disk usage strictly low
    load_best_model_at_end=True,
    logging_steps=200,
    report_to="none",
    dataloader_num_workers=0
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

print("\n" + "="*60)
print("🚀 STARTING CONTINUAL LEARNING OVERNIGHT RUN")
print("="*60 + "\n")

trainer.train()

# ============================================================
# EVALUATE
# ============================================================
print("\n📊 Evaluating on test set...")
test_results = trainer.evaluate(tokenized_dataset["test"])
print(f"   Test Accuracy: {test_results['eval_accuracy']:.4f}")

print(f"\n💾 Saving final model to {FINAL_MODEL_DIR}...")
trainer.save_model(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)

# ============================================================
# AUTOMATIC UPLOAD & CLEANUP
# ============================================================
print("\n☁️  Uploading automatically to Hugging Face...")
api = HfApi()
try:
    api.upload_folder(
        folder_path=FINAL_MODEL_DIR,
        repo_id=BASE_MODEL,
        repo_type="model",
        commit_message="Multi-dataset continual learning update"
    )
    print("   Upload successful!")
    
    print("\n🧹 Initiating Disk Cleanup...")
    # Delete model directory
    if os.path.exists(FINAL_MODEL_DIR):
        shutil.rmtree(FINAL_MODEL_DIR)
        print(f"   Deleted {FINAL_MODEL_DIR}")
    
    # Delete training checkpoints output dir
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        print(f"   Deleted {OUTPUT_DIR}")
        
    # Delete huggingface massive dataset cache
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        print(f"   Deleted {CACHE_DIR}")
        
    print("\n✅ Auto-Cleanup Complete. Disk space fully recovered!")

except Exception as e:
    print(f"\n❌ Upload failed: {e}")
    print("   Skipping auto-cleanup so you don't lose the model. Manually upload and delete later.")

print("\n" + "="*60)
print("🎉 FULL PIPELINE FINISHED.")
print("="*60)
