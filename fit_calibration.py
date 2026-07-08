"""
Re-fit the AI-detector's logit-margin calibration on a labelled dataset.

The Detector scores AI-likelihood from the logit MARGIN (human_logit - ai_logit)
via:   ai_prob = sigmoid((MIDPOINT - margin) / SCALE)
because the model's raw softmax is saturated. The default MIDPOINT/SCALE are
heuristic; this script fits them properly from labelled data so the calibration
is dataset-grounded (research-grade).

It is intentionally standalone: it depends only on torch + transformers + numpy
(and optionally `datasets`), NOT on the PaperGuard `agents` package, so it runs
in the training venv (which has torch/transformers/datasets) without the app's
pydantic stack.

Usage
-----
  # Prove the fitting math with synthetic margins (no model/data needed):
  python fit_calibration.py --self-test

  # Fit from a HuggingFace dataset:
  python fit_calibration.py --model training/mega_dataset_model_v2 \
      --hf-dataset Ateeqq/AI-and-Human-Generated-Text --split train \
      --text-col text --label-col label --ai-label 1 --samples 400

  # Fit from a local CSV (columns: text,label):
  python fit_calibration.py --model training/mega_dataset_model_v2 \
      --csv data/labelled.csv --text-col text --label-col label --ai-label 1

The script prints the fitted values to export:
  set PAPERGUARD_DETECTOR_CALIB_MIDPOINT=<...>
  set PAPERGUARD_DETECTOR_CALIB_SCALE=<...>
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from typing import List, Optional, Tuple

import numpy as np

# Current defaults (keep in sync with agents/detector_agent.py).
_DEFAULT_MIDPOINT = 12.0
_DEFAULT_SCALE = 2.5
_MAX_LENGTH = 512


# --------------------------------------------------------------------------- #
# Logistic fit (1-D, numpy gradient descent -> midpoint/scale)
# --------------------------------------------------------------------------- #
def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def fit_midpoint_scale(
    margins: np.ndarray, labels: np.ndarray, iters: int = 5000, lr: float = 0.05
) -> Tuple[float, float]:
    """
    Fit  P(AI) = sigmoid(w0 + w1 * margin)  by gradient descent, then convert to
    the (midpoint, scale) parameterisation used by the detector:

        sigmoid((midpoint - margin)/scale) = sigmoid(midpoint/scale - margin/scale)
        =>  w1 = -1/scale ,  w0 = midpoint/scale
        =>  scale = -1/w1 ,  midpoint = -w0/w1
    """
    # Standardise margins for stable optimisation, then de-standardise the coefs.
    mu, sigma = float(margins.mean()), float(margins.std() or 1.0)
    x = (margins - mu) / sigma
    y = labels.astype(float)

    a, b = 0.0, 0.0  # P(AI) = sigmoid(a + b*x)   (x is standardised margin)
    n = len(x)
    for _ in range(iters):
        p = _sigmoid(a + b * x)
        ga = float((p - y).mean())
        gb = float(((p - y) * x).mean())
        a -= lr * ga
        b -= lr * gb

    # De-standardise: a + b*((m-mu)/sigma) = (a - b*mu/sigma) + (b/sigma)*m
    w0 = a - b * mu / sigma
    w1 = b / sigma
    if abs(w1) < 1e-9:
        return _DEFAULT_MIDPOINT, _DEFAULT_SCALE
    scale = -1.0 / w1
    midpoint = -w0 / w1
    # scale must be positive (higher margin => lower P(AI)); guard degenerate fits.
    if scale <= 0:
        scale = abs(scale) or _DEFAULT_SCALE
    return round(midpoint, 3), round(scale, 3)


def _accuracy(margins: np.ndarray, labels: np.ndarray, midpoint: float, scale: float) -> float:
    probs = _sigmoid((midpoint - margins) / scale)
    preds = (probs >= 0.5).astype(int)
    return float((preds == labels).mean())


# --------------------------------------------------------------------------- #
# Model margins
# --------------------------------------------------------------------------- #
def _load_model(model_name: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    ai_idx = 0
    for label, idx in (getattr(model.config, "label2id", None) or {}).items():
        if str(label).strip().lower() == "ai":
            ai_idx = int(idx)
    return tok, model, ai_idx


def _margins_for(texts: List[str], model_name: str) -> np.ndarray:
    import torch

    tok, model, ai_idx = _load_model(model_name)
    human_idx = 1 - ai_idx
    out = []
    for i, t in enumerate(texts):
        inputs = tok(t, return_tensors="pt", truncation=True, max_length=_MAX_LENGTH)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        out.append(float(logits[human_idx].item() - logits[ai_idx].item()))
        if (i + 1) % 50 == 0:
            print(f"  scored {i + 1}/{len(texts)}", file=sys.stderr)
    return np.array(out, dtype=float)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _load_csv(path: str, text_col: str, label_col: str, ai_label: str, limit: int):
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row[text_col])
            labels.append(1 if str(row[label_col]).strip() == str(ai_label) else 0)
            if len(texts) >= limit:
                break
    return texts, labels


def _load_hf(name: str, split: str, text_col: str, label_col: str, ai_label: str, limit: int):
    from datasets import load_dataset

    ds = load_dataset(name, split=split)
    ds = ds.shuffle(seed=42).select(range(min(limit, len(ds))))
    texts = [r[text_col] for r in ds]
    labels = [1 if str(r[label_col]).strip() == str(ai_label) else 0 for r in ds]
    return texts, labels


# --------------------------------------------------------------------------- #
def _self_test() -> None:
    """Fit on synthetic margins (AI ~7, human ~17) to prove the math."""
    rng = random.Random(0)
    margins, labels = [], []
    for _ in range(400):
        margins.append(rng.gauss(7.0, 1.2)); labels.append(1)   # AI: low margin
        margins.append(rng.gauss(17.0, 1.2)); labels.append(0)  # human: high margin
    m, y = np.array(margins), np.array(labels)
    midpoint, scale = fit_midpoint_scale(m, y)
    print("[self-test] synthetic AI~7 / human~17")
    print(f"  fitted MIDPOINT={midpoint}  SCALE={scale}")
    print(f"  accuracy default(12.0/2.5): {_accuracy(m, y, 12.0, 2.5):.3f}")
    print(f"  accuracy fitted           : {_accuracy(m, y, midpoint, scale):.3f}")
    print("  (midpoint should land ~12 between the two clusters; scale small)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit detector logit-margin calibration.")
    ap.add_argument("--self-test", action="store_true", help="Fit on synthetic data; no model/dataset needed.")
    ap.add_argument("--model", default="training/mega_dataset_model_v2", help="Local path or HF repo id.")
    ap.add_argument("--csv", default=None, help="Local CSV with text/label columns.")
    ap.add_argument("--hf-dataset", default=None, help="HuggingFace dataset name.")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--ai-label", default="1", help="Value of label-col that means 'AI'.")
    ap.add_argument("--samples", type=int, default=400)
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return

    if args.csv:
        texts, labels = _load_csv(args.csv, args.text_col, args.label_col, args.ai_label, args.samples)
    elif args.hf_dataset:
        texts, labels = _load_hf(args.hf_dataset, args.split, args.text_col, args.label_col, args.ai_label, args.samples)
    else:
        ap.error("Provide --csv or --hf-dataset (or use --self-test).")

    if not texts:
        ap.error("No samples loaded.")

    print(f"Loaded {len(texts)} samples ({sum(labels)} AI / {len(labels) - sum(labels)} human).", file=sys.stderr)
    margins = _margins_for(texts, args.model)
    y = np.array(labels)

    midpoint, scale = fit_midpoint_scale(margins, y)
    print("\n=== Calibration fit ===")
    print(f"margin stats: AI mean={margins[y==1].mean():.2f}  human mean={margins[y==0].mean():.2f}")
    print(f"accuracy default({_DEFAULT_MIDPOINT}/{_DEFAULT_SCALE}): {_accuracy(margins, y, _DEFAULT_MIDPOINT, _DEFAULT_SCALE):.3f}")
    print(f"accuracy fitted({midpoint}/{scale})       : {_accuracy(margins, y, midpoint, scale):.3f}")
    print("\nExport these to use the fitted calibration:")
    print(f"  set PAPERGUARD_DETECTOR_CALIB_MIDPOINT={midpoint}")
    print(f"  set PAPERGUARD_DETECTOR_CALIB_SCALE={scale}")


if __name__ == "__main__":
    main()
