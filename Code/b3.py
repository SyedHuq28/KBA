#!/usr/bin/env python3
"""
b3_retriever_router.py

Replaces threshold-based B3 with a trained logistic regression router
on retriever score features. Same spirit as B3 (uses only retriever scores,
no model internals) but learns the decision boundary from data instead of
hardcoding thresholds.

Features per instance (from retriever_scores list):
  1. top1     — highest retriever score
  2. top2     — second highest score (0 if <2 docs)
  3. gap      — top1 - top2
  4. entropy  — distribution entropy over softmax scores
  5. mean     — mean of all scores
  6. std      — std of all scores

Output:
  results/b3_retriever_router.json
"""

import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, accuracy_score

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

MODES     = ["answer", "refuse", "conflict"]
LABEL_MAP = {"answer": 0, "refuse": 1, "conflict": 2}

# ── Feature extraction ─────────────────────────────────────────────────────────

def softmax(scores):
    s = np.array(scores, dtype=np.float64)
    if s.size == 0:
        return s
    s = s - s.max()
    e = np.exp(s)
    return e / e.sum()

def extract_features(retriever_scores):
    s = np.array(retriever_scores, dtype=np.float32) if retriever_scores else np.array([0.5])
    n = len(s)

    top1 = float(np.sort(s)[::-1][0]) if n >= 1 else 0.0
    top2 = float(np.sort(s)[::-1][1]) if n >= 2 else 0.0
    gap  = top1 - top2

    p   = softmax(s)
    ent = float(-np.sum(p * np.log(p + 1e-9)))
    mean = float(s.mean())
    std  = float(s.std()) if n > 1 else 0.0

    return [top1, top2, gap, ent, mean, std]

FEATURE_NAMES = ["top1", "top2", "gap", "entropy", "mean", "std"]

# ── Load instances ─────────────────────────────────────────────────────────────

def load_split(split):
    path = Path(f"instances_{split}.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run 0_build_instances.py first.")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

print("Loading instances...")
train_rows = load_split("train")
val_rows   = load_split("val")
test_rows  = load_split("test")

def rows_to_Xy(rows):
    X = np.array(
        [extract_features(r.get("retriever_scores", [])) for r in rows],
        dtype=np.float32
    )
    y = np.array([LABEL_MAP[r["true_mode"]] for r in rows], dtype=np.int64)
    return X, y

X_tr, y_tr = rows_to_Xy(train_rows)
X_va, y_va = rows_to_Xy(val_rows)
X_te, y_te = rows_to_Xy(test_rows)

print(f"Train: {len(y_tr)}  Val: {len(y_va)}  Test: {len(y_te)}")
print(f"Features: {FEATURE_NAMES}")

# ── Train ──────────────────────────────────────────────────────────────────────

scaler = StandardScaler().fit(X_tr)
X_tr_s = scaler.transform(X_tr)
X_va_s = scaler.transform(X_va)
X_te_s = scaler.transform(X_te)

clf = LogisticRegression(
    max_iter=2000, C=1.0, multi_class="multinomial",
    solver="lbfgs", random_state=42
)
clf.fit(X_tr_s, y_tr)

# ── Evaluate ───────────────────────────────────────────────────────────────────

def evaluate(X_s, y, name):
    pred     = clf.predict(X_s)
    acc      = float(accuracy_score(y, pred))
    macro_f1 = float(f1_score(y, pred, average="macro", zero_division=0))
    per_f1   = f1_score(y, pred, average=None, labels=[0, 1, 2], zero_division=0).tolist()
    print(f"\n── {name} ──────────────────────────────────────")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Macro F1:  {macro_f1:.4f}")
    print(classification_report(y, pred, target_names=MODES, zero_division=0))
    return {
        "accuracy":    acc,
        "macro_f1":    macro_f1,
        "f1_answer":   per_f1[0],
        "f1_refuse":   per_f1[1],
        "f1_conflict": per_f1[2],
    }

val_metrics  = evaluate(X_va_s, y_va, "VAL")
test_metrics = evaluate(X_te_s, y_te, "TEST")

# ── Learned coefficients (interpretable) ──────────────────────────────────────

print("\n── Learned coefficients per class ──────────────────")
for cls_idx, cls_name in enumerate(MODES):
    coef   = clf.coef_[cls_idx]
    ranked = sorted(zip(FEATURE_NAMES, coef), key=lambda x: abs(x[1]), reverse=True)
    print(f"  {cls_name}: " + "  ".join(f"{n}={v:+.3f}" for n, v in ranked))

# ── Save ───────────────────────────────────────────────────────────────────────

output = {
    "description": (
        "B3 replacement: logistic regression trained on retriever score features. "
        "No model internals used — only retriever_scores from instances_*.jsonl."
    ),
    "features": FEATURE_NAMES,
    "n_train":  int(len(y_tr)),
    "n_val":    int(len(y_va)),
    "n_test":   int(len(y_te)),
    "val":      val_metrics,
    "test":     test_metrics,
    "coefficients": {
        cls_name: {
            feat: float(clf.coef_[i][j])
            for j, feat in enumerate(FEATURE_NAMES)
        }
        for i, cls_name in enumerate(MODES)
    },
    "intercepts": {
        cls_name: float(clf.intercept_[i])
        for i, cls_name in enumerate(MODES)
    },
}

out_path = RESULTS_DIR / "b3_retriever_router.json"
out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
print(f"\nSaved → {out_path}")
