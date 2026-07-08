"""
Blind benchmark for the PaperGuard detector against fresh multi-model text.

Feed it a JSON file of labeled samples collected from ChatGPT / Claude / Grok /
Gemini / etc. (plus real human text), and it reports how well the (calibrated)
detector separates AI from human -- with a per-model leaderboard and, crucially,
recall on "disguised" AI (human-styled) vs "default" AI.

Input format (default: benchmark_samples.json) -- a JSON list:
[
  {"model": "gpt-5",   "label": "ai",    "group": "disguised", "text": "..."},
  {"model": "gpt-5",   "label": "ai",    "group": "default",   "text": "..."},
  {"model": "human",   "label": "human", "group": "real",       "text": "..."},
  ...
]
  - label: "ai" or "human"  (ground truth; required)
  - model: source name (optional; for the leaderboard)
  - group: "default" | "disguised" | "real" | anything (optional)

Run AFTER training finishes (loading the model competes for VRAM). Point at the
new weights:
    set PAPERGUARD_DETECTOR_MODEL=training/paperguard_v2_1
    python benchmark_detector.py benchmark_samples.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from typing import Any, Dict, List

# Decision threshold on the calibrated AI probability (0-100).
_THRESHOLD = 50.0


def _load(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list of {label, text, ...} objects.")
    return data


def _rate(correct: int, total: int) -> str:
    return f"{(100.0 * correct / total):.1f}% ({correct}/{total})" if total else "n/a"


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark the detector on labeled multi-model text.")
    ap.add_argument("input", nargs="?", default="benchmark_samples.json")
    ap.add_argument("--threshold", type=float, default=_THRESHOLD,
                    help="AI-probability cutoff for the binary AI/human decision.")
    args = ap.parse_args()

    samples = _load(args.input)
    from agents.detector_agent import DetectorAgent
    det = DetectorAgent()
    if not det.score_text("probe").get("available"):
        print("ERROR: detector model unavailable (set PAPERGUARD_DETECTOR_MODEL).", file=sys.stderr)
        sys.exit(1)

    rows = []
    for s in samples:
        label = str(s.get("label", "")).strip().lower()
        if label not in {"ai", "human"}:
            continue
        scored = det.score_text(s.get("text", ""))
        ai_p = scored.get("ai_probability")
        pred = "ai" if (ai_p is not None and ai_p >= args.threshold) else "human"
        rows.append({
            "model": s.get("model", "?"),
            "group": s.get("group", ""),
            "label": label,
            "ai_prob": ai_p,
            "pred": pred,
            "correct": pred == label,
            "text": (s.get("text", "") or "")[:70],
        })

    if not rows:
        print("No valid samples (need label 'ai'/'human').")
        return

    # ---- Per-sample ---- #
    print("\n=== Per sample ===")
    for r in rows:
        mark = "OK " if r["correct"] else "XX "
        ap_s = f"{r['ai_prob']:.1f}%" if r["ai_prob"] is not None else "n/a"
        print(f"  {mark} [{r['label']:>5} -> {r['pred']:>5} | AI {ap_s:>6}] "
              f"{r['model']}/{r['group']}: {r['text']}...")

    # ---- Aggregate ---- #
    total = len(rows)
    correct = sum(r["correct"] for r in rows)
    ai_rows = [r for r in rows if r["label"] == "ai"]
    hu_rows = [r for r in rows if r["label"] == "human"]
    ai_recall = sum(r["correct"] for r in ai_rows)
    fp = sum(1 for r in hu_rows if r["pred"] == "ai")

    print("\n=== Overall ===")
    print(f"  accuracy      : {_rate(correct, total)}")
    print(f"  AI recall     : {_rate(ai_recall, len(ai_rows))}   (AI caught)")
    print(f"  human FPR     : {_rate(fp, len(hu_rows))}   (humans wrongly flagged)")

    # ---- By group (default vs disguised vs real) ---- #
    by_group = defaultdict(lambda: [0, 0])
    for r in rows:
        g = r["group"] or "(none)"
        by_group[g][0] += r["correct"]
        by_group[g][1] += 1
    print("\n=== By group ===")
    for g, (c, t) in sorted(by_group.items()):
        print(f"  {g:>12}: {_rate(c, t)}")

    # ---- Threshold-independent quality + threshold sweep (needs BOTH classes) ---- #
    scored_rows = [r for r in rows if r["ai_prob"] is not None]
    have_both = any(r["label"] == "ai" for r in scored_rows) and any(r["label"] == "human" for r in scored_rows)
    if have_both:
        y = [1 if r["label"] == "ai" else 0 for r in scored_rows]
        s = [r["ai_prob"] for r in scored_rows]
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y, s)
            print(f"\n=== AUC (threshold-independent separation) ===\n  AUC = {auc:.3f}")
            print("  (High AUC + bad accuracy => just recalibrate the threshold, no retrain needed.)")
        except Exception as e:  # noqa: BLE001
            print(f"\n[AUC unavailable: {e}]")

        print("\n=== Threshold sweep (AI-prob cutoff -> recall / FPR) ===")
        print(f"  {'cutoff':>6}  {'AI recall':>10}  {'human FPR':>10}  {'Youden J':>9}")
        ai_s = [r["ai_prob"] for r in scored_rows if r["label"] == "ai"]
        hu_s = [r["ai_prob"] for r in scored_rows if r["label"] == "human"]
        for t in (20, 30, 40, 50, 60, 65, 70, 80):
            rec = sum(1 for v in ai_s if v >= t) / len(ai_s) if ai_s else 0
            fpr = sum(1 for v in hu_s if v >= t) / len(hu_s) if hu_s else 0
            print(f"  {t:>6}  {rec*100:>9.1f}%  {fpr*100:>9.1f}%  {(rec-fpr):>9.2f}")
        print("  (Best cutoff ~ max Youden J = recall - FPR.)")

        # ---- Dev/test split: pick threshold on dev, REPORT on held-out test ---- #
        def _bucket(r):
            h = int(hashlib.sha1((r["text"] or str(id(r))).encode("utf-8", "ignore")).hexdigest(), 16)
            return "dev" if h % 2 == 0 else "test"
        dev = [r for r in scored_rows if _bucket(r) == "dev"]
        test = [r for r in scored_rows if _bucket(r) == "test"]

        def _rf(rowset, t):
            a = [r["ai_prob"] for r in rowset if r["label"] == "ai"]
            h = [r["ai_prob"] for r in rowset if r["label"] == "human"]
            rec = sum(1 for v in a if v >= t) / len(a) if a else 0
            fpr = sum(1 for v in h if v >= t) / len(h) if h else 0
            return rec, fpr

        # best Youden threshold chosen ONLY on dev
        best_t, best_j = 50, -1
        for t in range(5, 96, 5):
            rec, fpr = _rf(dev, t)
            if rec - fpr > best_j:
                best_j, best_t = rec - fpr, t
        dev_rec, dev_fpr = _rf(dev, best_t)
        test_rec, test_fpr = _rf(test, best_t)
        print("\n=== Dev/test split (threshold chosen on DEV, reported on TEST) ===")
        print(f"  chosen cutoff (max Youden on dev): {best_t}")
        print(f"  DEV : recall {dev_rec*100:.1f}%  FPR {dev_fpr*100:.1f}%  (n={len(dev)})")
        print(f"  TEST: recall {test_rec*100:.1f}%  FPR {test_fpr*100:.1f}%  (n={len(test)})  <- the honest number")
        print("  (Small n: treat as indicative; wide confidence intervals.)")
    else:
        print("\n[AUC/threshold sweep skipped: need BOTH ai and human samples.]")

    # ---- Leaderboard: which model's AI evades best (lowest AI recall) ---- #
    by_model = defaultdict(lambda: [0, 0])
    for r in ai_rows:
        by_model[r["model"]][0] += r["correct"]
        by_model[r["model"]][1] += 1
    print("\n=== AI-recall by model (lower = better evasion by that model) ===")
    for m, (c, t) in sorted(by_model.items(), key=lambda kv: (kv[1][0] / kv[1][1]) if kv[1][1] else 1):
        print(f"  {m:>18}: {_rate(c, t)}")


if __name__ == "__main__":
    main()
