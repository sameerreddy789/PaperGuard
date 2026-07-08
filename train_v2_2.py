"""
PaperGuard detector v2.2 training pipeline  (Owner: @sameerreddy)

Goal: fix the v2.0 blind spot (slang / style-masked AI) and improve
generalization by training on a curated, HARD, diverse pool instead of more
easy data. Success is measured on a HELD-OUT set of unseen models + adversarial
attacks (accuracy / F1 / AUC / FPR) -- NOT eval_loss on the training distribution.

Label convention (matches the deployed model): 0 = AI, 1 = Human.

-----------------------------------------------------------------------------
v2.2 POST-MORTEM FIX (why v2.1 failed the frozen external benchmark)
-----------------------------------------------------------------------------
v2.1 scored AUC ~0.39 (below random) and 70-100% human FPR on the external
benchmark despite 93.9% held-out RAID recall. Root cause: the held-out eval was
**AI-only** (only AI rows were routed to eval_pool), so:
  * roc_auc_score() saw one class every epoch -> eval_auc = nan
  * false_positive_rate was never measured (no human in eval)
  * metric_for_best_model="auc" was always nan -> best-checkpoint selection dead
The model could (and did) satisfy the AI-only objective by over-flagging modern
human prose; nothing penalized it because FPR was invisible for the whole run.

This version fixes that with two changes:
  1. BALANCED, MULTI-REGISTER HELD-OUT EVAL. eval_pool now contains held-out
     HUMAN (RAID human + diverse registers: arxiv / news / informal / literary)
     alongside the held-out AI, so eval_auc and eval_false_positive_rate are
     REAL numbers and early-stopping actually works. Register rows are pulled
     DISJOINT from training (skip-offset) and de-duped against the frozen
     benchmark file to prevent test leakage.
  2. LOUD nan-AUC ALARM. A TrainerCallback aborts training if eval_auc comes
     back nan / the eval is one-class -- so this failure is caught at epoch 1,
     not after 5 GPU-hours + external audit. A pre-train assertion also refuses
     to start if eval_pool isn't two-class.
Plus diverse human negatives are added to the TRAINING pool (not just eval) so
the model can't win by flagging formality/register.
-----------------------------------------------------------------------------

Data sources (schemas verified):
  * liamdugan/raid            -> multi-model + 12 adversarial attacks + human.
                                 fields: model ("human"=human else AI), generation, attack, domain
  * Ateeqq/AI-and-Human...    -> academic abstracts. fields: abstract, label (1=AI,0=Human)
  * artem9k/ai-text-detection-pile -> fields: source ("human"=human else AI), text
  * (optional) a reasoning/CoT dataset (best-effort, auto-detected text field)

Held-out generalization test (no leakage into training):
  * RAID rows whose model is in --heldout-models  -> "unseen model" eval
  * RAID rows whose attack == --heldout-attack     -> "adversarial" eval

Typical runs
------------
  # Smoke test the curation (tiny, streaming, no training) -- also prints whether
  # the held-out eval came out two-class (the v2.1 guard):
  python train_v2_2.py --smoke

  # Real run (continue from v2.0), ~300k pool, 2 epochs:
  python train_v2_2.py --pool-size 300000 --epochs 2

  # Faithful to the "1 epoch ~= 150k" tip:
  python train_v2_2.py --pool-size 150000 --epochs 3

  # Push v2.2 to HF after training (needs HF_TOKEN):
  python train_v2_2.py --pool-size 300000 --epochs 2 --push

Run this in the training venv (has torch/transformers/datasets/sklearn + GPU).

Golden rules for a 6 GB laptop GPU (e.g. RTX 3050) -- ~4.5-5 h total:
  1. ROUTE THE CACHE. HF silently downloads tens of GB to C:. Use --cache-dir on
     a spacious drive (or set HF_DATASETS_CACHE / HF_HOME). Default: ./hf_cache.
  2. BASE MODEL = DistilBERT. Continue from v2.0 (default) or distilbert-base-*.
     Do NOT try RoBERTa-large / Llama on 6 GB.
  3. NO MULTITASKING while training (no 4K video / game / heavy dev server) -
     the GPU needs 100% for PyTorch.

Batch-size math (6 GB VRAM):
  * --batch-size 8  -> ~4.5-5.5 GB VRAM (sweet spot). 16 will OOM instantly.
  * eval batch is auto-set to 2x (no gradients stored during eval).
  * OOM fallback (background apps eating VRAM): --batch-size 4 --grad-accum 2
    -> holds 4 in memory but trains like an effective batch of 8.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter
from typing import Any, Callable, Dict, List, Optional

LABEL_AI = 0
LABEL_HUMAN = 1
_SEED = 42
_MIN_CHARS = 120          # skip very short fragments
_MAX_CHARS = 6000         # cap absurdly long rows before tokenization

# RAID models held out entirely from training (unseen-model generalization test).
DEFAULT_HELDOUT_MODELS = ["gpt4", "mistral-chat"]
# One adversarial attack held out entirely (adversarial generalization test).
DEFAULT_HELDOUT_ATTACK = "paraphrase"

# Diverse HUMAN registers, mirroring the external frozen benchmark's registers so
# that training-time held-out FPR actually predicts benchmark FPR. Used for BOTH
# the held-out eval and (a disjoint slice of) the training pool. Best-effort: any
# source whose id/config/field is wrong or unreachable is skipped with a warning
# (the RAID-human slice guarantees the eval is two-class even if all of these
# fail). Format: (dataset_id, config, split, text_field, register_name).
HUMAN_REGISTERS = [
    ("gfissore/arxiv-abstracts-2021", None, "train", "abstract", "arxiv"),
    ("abisee/cnn_dailymail",          "3.0.0", "train", "article", "news"),
    ("Yelp/yelp_review_full",         None, "train", "text",     "informal"),
    ("sedthh/gutenberg_english",      None, "train", "TEXT",     "humanities"),
]
# Path to the frozen benchmark; its texts are EXCLUDED from train+eval (no leak).
BENCHMARK_PATH = "benchmark_samples.json"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _load_env_file(path: str = ".env") -> None:
    """Zero-dependency .env loader (so HF_TOKEN etc. are available)."""
    try:
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if v.strip():
                        os.environ.setdefault(k.strip(), v.strip())
    except Exception:
        pass


def _clean(text: Any) -> Optional[str]:
    if not isinstance(text, str):
        return None
    t = " ".join(text.split())
    if len(t) < _MIN_CHARS:
        return None
    return t[:_MAX_CHARS]


def _hash(text: str) -> str:
    return hashlib.sha1(text[:200].encode("utf-8", "ignore")).hexdigest()


class Pool:
    """Accumulates unique (text,label) samples with per-bucket quotas."""

    def __init__(self):
        self.rows: List[Dict[str, Any]] = []
        self.seen: set = set()

    def add(self, text: str, label: int, bucket: str) -> bool:
        text = _clean(text)
        if text is None:
            return False
        h = _hash(text)
        if h in self.seen:
            return False
        self.seen.add(h)
        self.rows.append({"text": text, "label": label, "bucket": bucket})
        return True

    def count(self, bucket: str) -> int:
        return sum(1 for r in self.rows if r["bucket"] == bucket)


# --------------------------------------------------------------------------- #
# Loaders (streaming for the big sets)
# --------------------------------------------------------------------------- #
def load_raid(train_pool: Pool, eval_pool: Pool, n_ai: int, n_human: int,
              heldout_models: List[str], heldout_attack: str,
              n_heldout: int, max_iter: int, n_heldout_human: int = 2000,
              exclude_hashes: Optional[set] = None) -> None:
    """Single streaming pass over RAID, routing rows into train vs held-out eval.

    Held-out eval gets BOTH classes: AI (unseen models + held-out attack) and a
    HUMAN slice (``eval_human_raid``) so eval AUC/FPR are real (the v2.1 bug was
    an AI-only eval). The first ``n_heldout_human`` RAID-human rows go to eval;
    the rest go to training.
    """
    from datasets import load_dataset
    exclude_hashes = exclude_hashes or set()
    print(f"[RAID] streaming (target train: {n_ai} AI / {n_human} human, "
          f"heldout models {heldout_models}, heldout attack '{heldout_attack}', "
          f"heldout human {n_heldout_human})")
    ds = load_dataset("liamdugan/raid", split="train", streaming=True)

    ai_t = hu_t = hm_e = adv_e = he_e = 0
    for i, row in enumerate(ds):
        if i >= max_iter:
            break
        model = (row.get("model") or "").strip().lower()
        attack = (row.get("attack") or "none").strip().lower()
        text = row.get("generation")
        c = _clean(text)
        if c is not None and _hash(c) in exclude_hashes:
            continue  # never train/eval on a frozen-benchmark text

        if model == "human":
            # Reserve the first rows for the held-out eval; rest go to training.
            if he_e < n_heldout_human and eval_pool.add(text, LABEL_HUMAN, "eval_human_raid"):
                he_e += 1
            elif hu_t < n_human and train_pool.add(text, LABEL_HUMAN, "raid_human"):
                hu_t += 1
            continue

        # AI row. Route to held-out eval if it matches a held-out facet.
        if model in heldout_models:
            if hm_e < n_heldout and eval_pool.add(text, LABEL_AI, "eval_unseen_model"):
                hm_e += 1
            continue
        if attack == heldout_attack:
            if adv_e < n_heldout and eval_pool.add(text, LABEL_AI, "eval_adversarial"):
                adv_e += 1
            continue

        # Otherwise a training AI row (includes non-held-out adversarial attacks).
        if ai_t < n_ai and train_pool.add(text, LABEL_AI, "raid_ai"):
            ai_t += 1

        # RAID's value is AI diversity + the held-out facets; human is scarce here
        # and is filled mainly from the pile, so don't block on the human target.
        if (ai_t >= n_ai and hm_e >= n_heldout and adv_e >= n_heldout
                and he_e >= n_heldout_human):
            break
        if i % 100000 == 0 and i:
            print(f"  ...scanned {i} rows (ai_t={ai_t} hu_t={hu_t} "
                  f"unseen={hm_e} adv={adv_e} eval_human={he_e})")
    print(f"[RAID] done: train AI={ai_t} human={hu_t} | "
          f"eval unseen={hm_e} adversarial={adv_e} human={he_e}")


def load_pile(train_pool: Pool, n_ai: int, n_human: int, max_iter: int) -> None:
    from datasets import load_dataset
    print(f"[pile] streaming (target {n_ai} AI / {n_human} human)")
    ds = load_dataset("artem9k/ai-text-detection-pile", split="train", streaming=True)
    ds = ds.shuffle(seed=_SEED, buffer_size=10000)
    ai = hu = 0
    for i, row in enumerate(ds):
        if i >= max_iter or (ai >= n_ai and hu >= n_human):
            break
        src = str(row.get("source", "")).strip().lower()
        text = row.get("text")
        if src == "human":
            if hu < n_human and train_pool.add(text, LABEL_HUMAN, "pile_human"):
                hu += 1
        else:
            if ai < n_ai and train_pool.add(text, LABEL_AI, "pile_ai"):
                ai += 1
    print(f"[pile] done: AI={ai} human={hu}")


def load_ateeqq(train_pool: Pool, cap: int) -> None:
    from datasets import load_dataset
    print(f"[ateeqq] loading academic abstracts (cap {cap})")
    try:
        ds = load_dataset("Ateeqq/AI-and-Human-Generated-Text", split="train")
    except Exception as e:  # noqa: BLE001
        print(f"[ateeqq] skipped: {e}")
        return
    ai = hu = 0
    for row in ds:
        if ai + hu >= cap:
            break
        # Ateeqq label 1=AI, 0=Human -> our 0=AI, 1=Human
        our = LABEL_AI if row.get("label") == 1 else LABEL_HUMAN
        if train_pool.add(row.get("abstract"), our, "ateeqq"):
            ai += our == LABEL_AI
            hu += our == LABEL_HUMAN
    print(f"[ateeqq] done: AI={ai} human={hu}")


def load_cot(train_pool: Pool, repo: str, cap: int) -> None:
    """Best-effort reasoning/CoT slice (all AI). Auto-detects the text field."""
    if not repo or cap <= 0:
        return
    from datasets import load_dataset
    print(f"[cot] best-effort loading {repo} (cap {cap}, labelled AI)")
    try:
        ds = load_dataset(repo, split="train", streaming=True)
    except Exception as e:  # noqa: BLE001
        print(f"[cot] skipped ({repo}): {e}")
        return
    text_keys = ["text", "response", "output", "solution", "answer", "completion", "content"]
    added = 0
    for i, row in enumerate(ds):
        if added >= cap or i >= cap * 20:
            break
        text = None
        for k in text_keys:
            if isinstance(row.get(k), str) and len(row[k]) > _MIN_CHARS:
                text = row[k]
                break
        if text is None:  # try conversation-style
            conv = row.get("conversations") or row.get("messages")
            if isinstance(conv, list):
                parts = [m.get("value") or m.get("content") for m in conv
                         if isinstance(m, dict) and (m.get("from") in ("gpt", "assistant")
                                                     or m.get("role") == "assistant")]
                text = "\n".join(p for p in parts if isinstance(p, str)) or None
        if text and train_pool.add(text, LABEL_AI, "cot"):
            added += 1
    print(f"[cot] done: AI={added}")


def load_benchmark_hashes(path: str = BENCHMARK_PATH) -> set:
    """Hashes of frozen-benchmark texts, so we never leak the test set into
    train/eval. Uses the SAME _clean+_hash the pools use, so matches are exact."""
    hashes: set = set()
    try:
        with open(path, encoding="utf-8") as f:
            for row in json.load(f):
                c = _clean(row.get("text"))
                if c is not None:
                    hashes.add(_hash(c))
    except Exception as e:  # noqa: BLE001
        print(f"[leak-guard] benchmark file not read ({e}); proceeding without it.")
        return hashes
    print(f"[leak-guard] excluding {len(hashes)} frozen-benchmark texts from train+eval")
    return hashes


def load_human_registers(pool: Pool, per_source: int, bucket: str,
                         exclude_hashes: set, skip_rows: int = 0) -> None:
    """Pull HUMAN rows from diverse registers (arxiv/news/informal/literary) into
    ``pool``. Best-effort per source (skip on any failure). ``skip_rows`` offsets
    the stream so the eval slice (skip_rows=0) and the train slice (skip_rows=
    per_source_eval) are DISJOINT rows from the same corpus -> no train/eval leak.
    """
    if per_source <= 0:
        return
    from datasets import load_dataset
    for ds_id, config, split, field, reg in HUMAN_REGISTERS:
        got = seen = 0
        try:
            ds = load_dataset(ds_id, config, split=split, streaming=True)
            if skip_rows:
                ds = ds.skip(skip_rows)
            for row in ds:
                if got >= per_source or seen >= per_source * 50:
                    break
                seen += 1
                c = _clean(row.get(field))
                if c is None or _hash(c) in exclude_hashes:
                    continue
                if pool.add(c, LABEL_HUMAN, f"{bucket}_{reg}"):
                    got += 1
        except Exception as e:  # noqa: BLE001
            print(f"[registers] skipped {ds_id} ({reg}): {type(e).__name__}: {e}")
            continue
        print(f"[registers] {reg}: +{got} human -> {bucket}")


# --------------------------------------------------------------------------- #
# Balancing
# --------------------------------------------------------------------------- #
def balance_eval(eval_pool: Pool) -> List[Dict[str, Any]]:
    """Balance the held-out eval to equal AI/human so AUC/FPR aren't dominated by
    class imbalance. Human keeps its register mix; AI keeps its facet mix."""
    ai = [r for r in eval_pool.rows if r["label"] == LABEL_AI]
    hu = [r for r in eval_pool.rows if r["label"] == LABEL_HUMAN]
    n = min(len(ai), len(hu))
    rng = random.Random(_SEED)
    rng.shuffle(ai)
    rng.shuffle(hu)
    out = ai[:n] + hu[:n]
    rng.shuffle(out)
    print(f"[eval-balance] AI={len(ai)} human={len(hu)} -> {len(out)} balanced")
    print(f"[eval-balance] register mix: {Counter(r['bucket'] for r in out)}")
    return out
def balance(train_pool: Pool) -> List[Dict[str, Any]]:
    ai = [r for r in train_pool.rows if r["label"] == LABEL_AI]
    hu = [r for r in train_pool.rows if r["label"] == LABEL_HUMAN]
    n = min(len(ai), len(hu))
    rng = random.Random(_SEED)
    rng.shuffle(ai)
    rng.shuffle(hu)
    balanced = ai[:n] + hu[:n]
    rng.shuffle(balanced)
    print(f"[balance] AI avail={len(ai)} human avail={len(hu)} -> {len(balanced)} balanced")
    print(f"[balance] source mix: {Counter(r['bucket'] for r in balanced)}")
    return balanced


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def build_compute_metrics():
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = np.asarray(logits)
        labels = np.asarray(labels)
        preds = logits.argmax(axis=-1)
        # AI is label 0; "AI score" = softmax prob of class 0.
        ex = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = ex / ex.sum(axis=-1, keepdims=True)
        ai_score = probs[:, LABEL_AI]
        is_ai_true = (labels == LABEL_AI).astype(int)

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro")
        try:
            auc = roc_auc_score(is_ai_true, ai_score)
        except ValueError:
            auc = float("nan")
        # FPR = human (label 1) misclassified as AI (pred 0)
        human_mask = labels == LABEL_HUMAN
        fpr = float((preds[human_mask] == LABEL_AI).mean()) if human_mask.any() else float("nan")
        # Recall on AI (catching machine text)
        ai_mask = labels == LABEL_AI
        ai_recall = float((preds[ai_mask] == LABEL_AI).mean()) if ai_mask.any() else float("nan")
        return {"accuracy": acc, "f1_macro": f1, "auc": auc,
                "false_positive_rate": fpr, "ai_recall": ai_recall}

    return compute_metrics


class OneClassEvalError(RuntimeError):
    """Raised when the held-out eval isn't two-class (the v2.1 failure mode)."""


def assert_two_class_eval(eval_rows: List[Dict[str, Any]]) -> None:
    """Refuse to start training on a degenerate eval set. This is the guard that
    would have stopped v2.1 before 5 GPU-hours were spent training blind."""
    labels = Counter(r["label"] for r in eval_rows)
    n_ai, n_hu = labels.get(LABEL_AI, 0), labels.get(LABEL_HUMAN, 0)
    if n_ai == 0 or n_hu == 0:
        raise OneClassEvalError(
            f"Held-out eval is ONE-CLASS (AI={n_ai}, human={n_hu}). AUC/FPR would "
            "be nan and metric_for_best_model='auc' could not select a checkpoint "
            "-- this is exactly the v2.1 failure. Aborting. Ensure RAID-human "
            "and/or HUMAN_REGISTERS actually loaded into eval_pool."
        )
    print(f"[eval-guard] OK: held-out eval is two-class (AI={n_ai}, human={n_hu}).")


def make_auc_sanity_callback():
    """A TrainerCallback that ABORTS if eval_auc comes back nan (one-class /
    degenerate eval) and prints the real FPR/AUC each epoch. Lazy import so the
    module stays importable without transformers (e.g. for --smoke)."""
    from transformers import TrainerCallback

    class AucSanityCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if not metrics:
                return control
            auc = metrics.get("eval_auc")
            fpr = metrics.get("eval_false_positive_rate")
            rec = metrics.get("eval_ai_recall")
            if auc is None or (isinstance(auc, float) and math.isnan(auc)):
                print("\n" + "!" * 72)
                print("[ALARM] eval_auc is nan -> held-out eval is one-class / "
                      "degenerate. FPR is UNMEASURED and best-checkpoint selection "
                      "is broken. Stopping training NOW (this is the v2.1 trap).")
                print("!" * 72 + "\n")
                control.should_training_stop = True
                return control
            fpr_s = f"{fpr:.3f}" if isinstance(fpr, float) and not math.isnan(fpr) else fpr
            print(f"[eval] held-out AUC={auc:.3f} FPR={fpr_s} ai_recall={rec}")
            if isinstance(fpr, float) and not math.isnan(fpr) and fpr > 0.10:
                print(f"[warn] held-out human FPR {fpr:.1%} is high -- the model is "
                      "over-flagging human text (v2.1 symptom). Watch the external "
                      "benchmark before trusting recall gains.")
            return control

    return AucSanityCallback()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Train PaperGuard detector v2.2.")
    ap.add_argument("--base-model", default="vediumsameer/paperguard-ai-detector",
                    help="Continue from v2.0 (default) or use distilbert-base-cased.")
    ap.add_argument("--pool-size", type=int, default=300000,
                    help="Target balanced training pool (per-epoch size). ~500000 recommended if time allows.")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Train batch. 8 = sweet spot for 6GB. OOM? use 4 with --grad-accum 2.")
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="gradient_accumulation_steps; OOM fallback: --batch-size 4 --grad-accum 2 (effective 8).")
    ap.add_argument("--cache-dir", default="hf_cache",
                    help="HF datasets/model cache. Point at a spacious drive (e.g. D:\\hf_cache) so C: isn't filled.")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--heldout-models", nargs="*", default=DEFAULT_HELDOUT_MODELS)
    ap.add_argument("--heldout-attack", default=DEFAULT_HELDOUT_ATTACK)
    ap.add_argument("--heldout-size", type=int, default=3000, help="per held-out AI facet")
    ap.add_argument("--heldout-human", type=int, default=2000,
                    help="RAID-human rows reserved for the held-out eval (v2.2: eval must be two-class).")
    ap.add_argument("--eval-human-per-source", type=int, default=800,
                    help="held-out human rows per diverse register (arxiv/news/informal/literary).")
    ap.add_argument("--train-reg-per-source", type=int, default=8000,
                    help="TRAIN human rows per diverse register (disjoint from eval), so the "
                         "model can't win by flagging formality/register.")
    ap.add_argument("--ateeqq-cap", type=int, default=20000)
    ap.add_argument("--cot-dataset", default="open-thoughts/OpenThoughts-114k")
    ap.add_argument("--cot-cap", type=int, default=20000)
    ap.add_argument("--pile-frac", type=float, default=0.3,
                    help="fraction of the pool filled from the (easy) pile for stability")
    ap.add_argument("--output-dir", default="./training/v2_2_output")
    ap.add_argument("--save-dir", default="./training/paperguard_v2_2")
    ap.add_argument("--push", action="store_true", help="Push to HF after training (needs HF_TOKEN).")
    ap.add_argument("--repo-id", default="vediumsameer/paperguard-ai-detector")
    ap.add_argument("--smoke", action="store_true", help="Tiny curation-only dry run (no training).")
    args = ap.parse_args()
    _load_env_file()  # pick up HF_TOKEN etc. from .env

    # Route HF caches to a spacious drive BEFORE datasets/transformers touch disk.
    cache = os.path.abspath(args.cache_dir)
    os.makedirs(cache, exist_ok=True)
    os.environ.setdefault("HF_DATASETS_CACHE", cache)
    os.environ.setdefault("HF_HOME", cache)
    print(f"[cache] HF cache -> {cache}")

    if args.smoke:
        args.pool_size = 400
        args.heldout_size = 40
        args.heldout_human = 40
        args.eval_human_per_source = 20
        args.train_reg_per_source = 40
        args.ateeqq_cap = 100
        args.cot_cap = 40

    half = args.pool_size // 2
    # Human is the SCARCE class -> the pile is the primary human source.
    # RAID is the primary AI source (adversarial + multi-model diversity); it has
    # only ~tens of thousands of human docs, so we take those opportunistically.
    raid_ai = half
    raid_human_cap = 30000
    pile_human = half
    pile_ai = max(int(half * 0.15), 5000)   # small stability slice of easy AI
    raid_max_iter = 20000 if args.smoke else 1_500_000
    pile_max_iter = 20000 if args.smoke else 3_000_000

    # Never train/eval on the frozen benchmark texts (keeps the test set clean).
    exclude_hashes = load_benchmark_hashes()

    train_pool, eval_pool = Pool(), Pool()
    # RAID: training AI + held-out AI facets + a held-out HUMAN slice for the eval.
    load_raid(train_pool, eval_pool, raid_ai, raid_human_cap,
              [m.lower() for m in args.heldout_models], args.heldout_attack.lower(),
              args.heldout_size, raid_max_iter,
              n_heldout_human=args.heldout_human, exclude_hashes=exclude_hashes)
    # Diverse-register HUMAN. Eval slice first (skip_rows=0); TRAIN slice offset so
    # the two are disjoint rows from the same corpora (no train/eval leakage).
    load_human_registers(eval_pool, args.eval_human_per_source, "eval_human",
                         exclude_hashes, skip_rows=0)
    load_human_registers(train_pool, args.train_reg_per_source, "train_reg",
                         exclude_hashes, skip_rows=args.eval_human_per_source)
    load_ateeqq(train_pool, args.ateeqq_cap)
    load_cot(train_pool, args.cot_dataset, args.cot_cap)
    load_pile(train_pool, pile_ai, pile_human, pile_max_iter)

    train_rows = balance(train_pool)
    eval_rows = balance_eval(eval_pool)   # two-class, register-diverse held-out eval
    print(f"\n[pool] final train={len(train_rows)}  held-out eval={len(eval_rows)} "
          f"({Counter(r['bucket'] for r in eval_rows)})")

    if args.smoke:
        n_ai = sum(r["label"] == LABEL_AI for r in eval_rows)
        n_hu = sum(r["label"] == LABEL_HUMAN for r in eval_rows)
        print(f"\n[smoke] curation OK. eval two-class? AI={n_ai} human={n_hu} "
              f"({'YES' if n_ai and n_hu else 'NO -- would abort real run'}).")
        print("[smoke] Skipping training. Re-run without --smoke to train.")
        return

    # HARD GUARD: refuse to train blind on a one-class eval (the v2.1 failure).
    assert_two_class_eval(eval_rows)

    # ---- Train ---- #
    import torch  # noqa: F401
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def tok(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True,
                         max_length=args.max_length)

    train_ds = Dataset.from_list([{"text": r["text"], "label": r["label"]} for r in train_rows])
    eval_ds = Dataset.from_list([{"text": r["text"], "label": r["label"]} for r in eval_rows])
    train_ds = train_ds.map(tok, batched=True)
    eval_ds = eval_ds.map(tok, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=2, ignore_mismatched_sizes=True)
    model.config.id2label = {0: "ai", 1: "human"}
    model.config.label2id = {"ai": 0, "human": 1}

    targs = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(args.batch_size * 2, 16),
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=100,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="auc",
        report_to="none",
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      eval_dataset=eval_ds, compute_metrics=build_compute_metrics(),
                      callbacks=[make_auc_sanity_callback()])

    print("\n[train] starting...")
    trainer.train()
    metrics = trainer.evaluate()
    print(f"\n[eval] HELD-OUT (two-class: unseen-model + adversarial AI vs "
          f"multi-register human): {metrics}")
    print("[eval] NOTE: held-out FPR/AUC are necessary but NOT sufficient -- the "
          "ship/block decision is made on the FROZEN EXTERNAL benchmark "
          "(benchmark_detector.py + benchmark_samples.json), per benchmark_results.md.")

    trainer.save_model(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    print(f"[save] model saved to {args.save_dir}")
    print("[next] re-fit calibration:  python fit_calibration.py --model "
          f"{args.save_dir} --hf-dataset liamdugan/raid ...  (see fit_calibration.py)")

    if args.push:
        _push(args.save_dir, args.repo_id, metrics)


def _push(save_dir: str, repo_id: str, metrics: Dict[str, Any]) -> None:
    import os
    if not os.getenv("HF_TOKEN"):
        print("[push] HF_TOKEN not set; skipping push.")
        return
    # Safety gate: never overwrite the deployed v2.0 with a model that looks bad
    # even on the (lenient) held-out set. The frozen external benchmark is the
    # real gate; this just stops an obviously-broken push (the v2.1 near-miss).
    auc = metrics.get("eval_auc")
    fpr = metrics.get("eval_false_positive_rate")
    bad = ((isinstance(auc, float) and (math.isnan(auc) or auc < 0.90))
           or (isinstance(fpr, float) and (math.isnan(fpr) or fpr > 0.10)))
    if bad and os.getenv("PAPERGUARD_FORCE_PUSH") != "1":
        print(f"[push] REFUSING to push: held-out AUC={auc}, FPR={fpr} look bad "
              "(need AUC>=0.90 and FPR<=0.10, AND a passing frozen-benchmark run). "
              "This guards the deployed v2.0. Override with PAPERGUARD_FORCE_PUSH=1 "
              "only after benchmark_detector.py passes.")
        return
    from huggingface_hub import HfApi
    card = f"""---
license: mit
language: en
pipeline_tag: text-classification
tags: [ai-text-detection, academic-integrity, paperguard]
---

# PaperGuard AI Detector (v2.2)

Fine-tuned DistilBERT for AI-generated text detection in academic writing.
v2.2 was trained on a curated **hard** pool (RAID multi-model + adversarial,
academic abstracts, reasoning/CoT, and a stability slice) with a **balanced,
multi-register held-out eval** (the v2.1 eval was AI-only, which hid a
catastrophic false-positive regression) to improve generalization and
robustness to style-masked AI.

Labels: `0 = ai`, `1 = human`. The raw softmax is intentionally not used
directly; PaperGuard scores off the calibrated logit margin.

## Held-out evaluation (unseen models + adversarial attacks)
- accuracy: {metrics.get('eval_accuracy')}
- f1_macro: {metrics.get('eval_f1_macro')}
- auc: {metrics.get('eval_auc')}
- false_positive_rate: {metrics.get('eval_false_positive_rate')}
- ai_recall: {metrics.get('eval_ai_recall')}
"""
    with open(f"{save_dir}/README.md", "w", encoding="utf-8") as f:
        f.write(card)
    api = HfApi()
    api.upload_folder(folder_path=save_dir, repo_id=repo_id,
                      commit_message=f"Upload v2.2 (held-out AUC={metrics.get('eval_auc')})")
    print(f"[push] uploaded v2.2 to {repo_id}")


if __name__ == "__main__":
    main()
