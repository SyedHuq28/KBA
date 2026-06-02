#!/usr/bin/env python3
"""
reeval.py  —  Re-evaluate saved results JSONL files with correct classification logic.

Usage:
    python reeval.py --input results.jsonl --output results_fixed.jsonl
    python reeval.py --input results.jsonl --output results_fixed.jsonl --has_answer_key

Gold label mapping (from your samples):
    0 → answer
    1 → refuse
    2 → conflict
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------
GOLD_INT_TO_STR = {0: "answer", 1: "refuse", 2: "conflict"}
GOLD_STR_TO_INT = {"answer": 0, "refuse": 1, "conflict": 2}

# ---------------------------------------------------------------------------
# Refuse / Conflict signal patterns (broad, case-insensitive)
# These cover SelfRAG special tokens and natural language variants
# ---------------------------------------------------------------------------

REFUSE_PATTERNS = [
    # Exact / near-exact strings
    r"not enough information",
    r"insufficient information",
    r"cannot (be )?determined",
    r"cannot answer",
    r"no (relevant |sufficient )?information",
    r"the (documents?|passage|text|context) (do(es)? not|don'?t) (contain|provide|mention|include|have)",
    r"(based on|according to) the (provided |given )?(documents?|passage|text|context)[^.]*not",
    r"(no|without) (relevant )?information (is |was )?(provided|available|found|given)",
    r"(the )?(provided |given )?(documents?|passage|text|context) (lack|lacks|do not contain)",
    # SelfRAG special tokens
    r"\[No Retrieval\]",
    r"\[Irrelevant\]",
    # Common refuse phrases
    r"i (don'?t|do not|cannot|can'?t) (know|have enough|find)",
    r"there (is|are) no (information|data|evidence)",
    r"the answer (is not|cannot be) (found|determined)",
    r"unable to (answer|determine|find)",
    r"information (is )?not (available|provided|given|found)",
]

CONFLICT_PATTERNS = [
    # Exact / near-exact strings
    r"conflicting information",
    r"contradictory information",
    r"conflict(s)? (between|in|among)",
    r"contradict(s|ion|ing)?",
    r"(documents?|sources?|passages?) (conflict|contradict|disagree|differ)",
    r"(one|some) (document|source|passage).*(while|whereas|but|however).*(another|other)",
    r"inconsisten(t|cy|cies)",
    r"(both|two) (documents?|sources?|passages?) (say|state|claim|indicate|suggest)",
    r"discrepan(t|cy|cies)",
    r"documents contain conflicting",
    r"conflicting (information|evidence|data|claims?|statements?)",
]

_REFUSE_RE  = re.compile("|".join(REFUSE_PATTERNS),  re.IGNORECASE)
_CONFLICT_RE = re.compile("|".join(CONFLICT_PATTERNS), re.IGNORECASE)


def classify_output(text: str) -> str:
    """
    Classify model output into one of: 'answer', 'refuse', 'conflict'.
    Priority: conflict > refuse > answer  (most specific wins).
    """
    if _CONFLICT_RE.search(text):
        return "conflict"
    if _REFUSE_RE.search(text):
        return "refuse"
    return "answer"


def normalise_gold(gold) -> str:
    """Accept int (0/1/2) or string ('answer'/'refuse'/'conflict')."""
    if isinstance(gold, int):
        return GOLD_INT_TO_STR.get(gold, "answer")
    if isinstance(gold, str):
        return gold.lower()
    return "answer"


def is_answer_correct(output: str, question: str) -> bool:
    """
    For the 'answer' mode: the model output is correct if it is NOT a refuse/conflict
    signal AND actually contains some substantive text (>5 chars after stripping
    SelfRAG tokens and special markers).
    
    NOTE: For a proper QA accuracy check you'd need the gold answer string.
    Since your JSONL doesn't store it, we use a heuristic: non-empty, non-refuse,
    non-conflict, non-degenerate output counts as a correct answer attempt.
    """
    # Strip SelfRAG tokens like [Utility:5], [Continue to Use Evidence], etc.
    cleaned = re.sub(r"\[.*?\]", "", output).strip()
    # Strip repeated assistant tokens (degenerate outputs)
    cleaned = re.sub(r"(<\|assistant\|>\s*)+", "", cleaned).strip()
    # Must be substantive
    return len(cleaned) > 5


# ---------------------------------------------------------------------------
# Per-instance correctness logic
# ---------------------------------------------------------------------------

def compute_correct(gold_str: str, pred_str: str, output: str, question: str) -> int:
    """
    Correctness rules:
      - gold == 'refuse'  → correct iff pred == 'refuse'   (model emitted refuse signal)
      - gold == 'conflict'→ correct iff pred == 'conflict'  (model emitted conflict signal)
      - gold == 'answer'  → correct iff pred == 'answer'
                            AND output is substantive (not degenerate / not refuse/conflict)
    """
    if gold_str == "refuse":
        return int(pred_str == "refuse")
    if gold_str == "conflict":
        return int(pred_str == "conflict")
    # gold == 'answer'
    if pred_str != "answer":
        return 0
    return int(is_answer_correct(output, question))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: List[dict]) -> dict:
    total   = len(results)
    correct = sum(r["correct"] for r in results)
    accuracy = correct / total if total else 0.0

    # Per-class TP/FP/FN
    classes = ["answer", "refuse", "conflict"]
    counts  = {c: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for c in classes}
    for r in results:
        g = r["gold_str"]
        p = r["pred"]
        for c in classes:
            if g == c and p == c:
                counts[c]["tp"] += 1
            elif g != c and p == c:
                counts[c]["fp"] += 1
            elif g == c and p != c:
                counts[c]["fn"] += 1
            else:
                counts[c]["tn"] += 1

    metrics = {"accuracy": accuracy, "chance_accuracy": 1.0 / len(classes)}

    f1s = []
    for c in classes:
        tp = counts[c]["tp"]
        fp = counts[c]["fp"]
        fn = counts[c]["fn"]
        tn = counts[c]["tn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)

        # FAR = False Answer Rate = FP / (FP + TN)  for this class
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        metrics[f"{c}_precision"] = prec
        metrics[f"{c}_recall"]    = rec
        metrics[f"{c}_f1"]        = f1
        metrics[f"{c}_far"]       = far  # False Answer Rate
        metrics[f"{c}_tp"]        = tp
        metrics[f"{c}_fp"]        = fp
        metrics[f"{c}_fn"]        = fn
        metrics[f"{c}_tn"]        = tn

    metrics["macro_f1"] = sum(f1s) / len(f1s)

    # Overall FAR: fraction of non-answer golds predicted as answer
    non_answer_golds  = [r for r in results if r["gold_str"] != "answer"]
    false_answers     = [r for r in non_answer_golds if r["pred"] == "answer"]
    metrics["overall_far"] = (
        len(false_answers) / len(non_answer_golds) if non_answer_golds else 0.0
    )

    # Answer accuracy (only answer-gold rows)
    answer_rows = [r for r in results if r["gold_str"] == "answer"]
    metrics["answer_accuracy"] = (
        sum(r["correct"] for r in answer_rows) / len(answer_rows)
        if answer_rows else 0.0
    )

    # Refuse accuracy
    refuse_rows = [r for r in results if r["gold_str"] == "refuse"]
    metrics["refuse_accuracy"] = (
        sum(r["correct"] for r in refuse_rows) / len(refuse_rows)
        if refuse_rows else 0.0
    )

    # Conflict accuracy
    conflict_rows = [r for r in results if r["gold_str"] == "conflict"]
    metrics["conflict_accuracy"] = (
        sum(r["correct"] for r in conflict_rows) / len(conflict_rows)
        if conflict_rows else 0.0
    )

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def reeval_file(input_path: str, output_path: str) -> dict:
    results = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)

            output   = r.get("output", "")
            question = r.get("question", "")
            gold_raw = r.get("gold", 0)

            gold_str = normalise_gold(gold_raw)
            pred_str = classify_output(output)

            correct  = compute_correct(gold_str, pred_str, output, question)

            r["gold_str"] = gold_str
            r["pred"]     = pred_str
            r["correct"]  = correct
            results.append(r)

    # Write fixed JSONL
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Compute and write metrics
    metrics = compute_metrics(results)
    metrics_path = out.with_suffix(".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    p = argparse.ArgumentParser(description="Re-evaluate saved RAG results JSONL")
    p.add_argument("--input",  required=True,  help="Input results JSONL")
    p.add_argument("--output", required=True,  help="Output fixed JSONL")
    args = p.parse_args()

    print(f"Re-evaluating: {args.input}")
    metrics = reeval_file(args.input, args.output)

    print(f"\n{'='*50}")
    print(f"  Fixed results → {args.output}")
    print(f"{'='*50}")
    print(f"  Accuracy        : {metrics['accuracy']:.4f}  (chance={metrics['chance_accuracy']:.4f})")
    print(f"  Macro F1        : {metrics['macro_f1']:.4f}")
    print(f"  Overall FAR     : {metrics['overall_far']:.4f}")
    print()
    for c in ["answer", "refuse", "conflict"]:
        print(f"  [{c}]")
        print(f"    Accuracy  : {metrics[f'{c}_accuracy' if f'{c}_accuracy' in metrics else 'accuracy']:.4f}")
        print(f"    Precision : {metrics[f'{c}_precision']:.4f}")
        print(f"    Recall    : {metrics[f'{c}_recall']:.4f}")
        print(f"    F1        : {metrics[f'{c}_f1']:.4f}")
        print(f"    FAR       : {metrics[f'{c}_far']:.4f}")
        print(f"    TP/FP/FN/TN: {metrics[f'{c}_tp']}/{metrics[f'{c}_fp']}/{metrics[f'{c}_fn']}/{metrics[f'{c}_tn']}")
        print()


if __name__ == "__main__":
    main()
