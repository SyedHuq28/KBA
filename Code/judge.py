#!/usr/bin/env python3
import argparse, json, re, glob, csv
from pathlib import Path
from collections import Counter
import numpy as np

# Canonical strings used in your project
REFUSE_STRING = "Not enough information."
CONFLICT_STRING = "Documents contain conflicting information."
MODES = ["answer", "refuse", "conflict"]

# -----------------------------
# Evaluation: first-line triage
# -----------------------------
def first_nonempty_line(text: str) -> str:
    for line in (text or "").splitlines():
        s = line.strip()
        if s:
            return s
    return ""

REFUSE_RE = re.compile(
    r"\b("
    r"not enough (information|evidence)|insufficient (information|evidence)|"
    r"cannot (answer|determine|verify)|can't (answer|determine|verify)|"
    r"unable to (answer|determine|verify)|"
    r"documents? do not (contain|include)|"
    r"cannot be answered from the provided"
    r")\b",
    re.IGNORECASE,
)
CONFLICT_RE = re.compile(
    r"\b(documents?|sources?|passages?|provided documents?)\b.*\b("
    r"conflict|conflicting|contradict|contradiction|inconsistent|disagree|cannot reconcile|at odds"
    r")\b",
    re.IGNORECASE,
)

def classify_mode_firstline(text: str) -> str:
    line = first_nonempty_line(text)
    low = line.lower()

    # canonical strings (strongest)
    if low.startswith(CONFLICT_STRING.lower()):
        return "conflict"
    if low.startswith(REFUSE_STRING.lower()):
        return "refuse"

    # first-line patterns
    if CONFLICT_RE.search(line):
        return "conflict"
    if REFUSE_RE.search(line):
        return "refuse"
    return "answer"

def macro_f1(y_true, y_pred, labels=MODES) -> float:
    f1s=[]
    for lab in labels:
        tp=sum((t==lab and p==lab) for t,p in zip(y_true,y_pred))
        fp=sum((t!=lab and p==lab) for t,p in zip(y_true,y_pred))
        fn=sum((t==lab and p!=lab) for t,p in zip(y_true,y_pred))
        prec=tp/(tp+fp+1e-9)
        rec =tp/(tp+fn+1e-9)
        f1=2*prec*rec/(prec+rec+1e-9)
        f1s.append(f1)
    return float(np.mean(f1s))

def acc(y_true, y_pred) -> float:
    return float(np.mean([t==p for t,p in zip(y_true,y_pred)]))

def far_refuse_conflict(y_true, y_pred) -> float:
    denom = sum(t in ("refuse","conflict") for t in y_true)
    num = sum((t in ("refuse","conflict") and p=="answer") for t,p in zip(y_true,y_pred))
    return float(num / max(1, denom))

def answer_substring(rows, get_text_fn) -> float:
    vals=[]
    for r in rows:
        if r.get("true_mode") != "answer":
            continue
        gold = (r.get("gold_answer") or "").strip().lower()
        pred = (get_text_fn(r) or "").strip().lower()
        if not gold:
            continue
        vals.append(1.0 if gold in pred else 0.0)
    return float(np.mean(vals)) if vals else 0.0

# -----------------------------
# I/O helpers
# -----------------------------
def load_json(path: Path):
    return json.load(open(path, "r", encoding="utf-8"))

def save_csv(path: Path, headers, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)

def md_table(headers, rows):
    # simple markdown table
    out=[]
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"]*len(headers)) + " |")
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)

# -----------------------------
# Table builders
# -----------------------------
def table_prompt_dependence(results_dir: Path):
    path = results_dir / "prompt_dependence.json"
    if not path.exists():
        return None, None
    d = load_json(path)
    rows=[]
    for tmpl in ["P0","P1","P2","P3","P4","P5"]:
        if tmpl not in d or "val_acc" not in d[tmpl]:
            continue
        rows.append([tmpl, f"{d[tmpl]['val_acc']:.4f}", f"{d[tmpl]['val_f1']:.4f}", d[tmpl].get("n","")])
    headers=["Template","Val Acc","Val Macro-F1","N"]
    return headers, rows

def table_steering_val(rescored_val_all: Path):
    d = load_json(rescored_val_all)
    # keys like ablation_val_a2.json -> parsed alpha
    def alpha_from_key(k):
        m=re.search(r"_a([0-9.]+)\.json", k)
        return float(m.group(1)) if m else None

    rows=[]
    for k,obj in d.items():
        a = alpha_from_key(k)
        if a is None:
            continue
        s3=obj["S3"]; s4=obj["S4"]
        rows.append([
            a,
            f"{s3['macro_f1']:.4f}", f"{s3['FAR_refuse+conflict']:.4f}", f"{s3['answer_substring']:.4f}",
            f"{s4['macro_f1']:.4f}", f"{s4['FAR_refuse+conflict']:.4f}", f"{s4['answer_substring']:.4f}",
        ])
    rows=sorted(rows, key=lambda x: x[0])
    headers=["alpha(rms)","S3_f1","S3_FAR","S3_ansSub","S4_f1","S4_FAR","S4_ansSub"]
    return headers, rows

def table_systems_test(rescored_test_path: Path):
    d = load_json(rescored_test_path)
    rows=[]
    for cond in ["S2","S3","S4"]:
        if cond not in d:
            continue
        x=d[cond]
        rows.append([cond, f"{x['acc']:.4f}", f"{x['macro_f1']:.4f}", f"{x['FAR_refuse+conflict']:.4f}", f"{x['answer_substring']:.4f}",
                     str(x.get("pred_counts",{}))])
    headers=["System","Acc","Macro-F1","FAR(ref+conf)","Ans substring","Pred counts"]
    return headers, rows

def table_causal_patching(causal_path: Path):
    d = load_json(causal_path)
    res = d.get("results", {})
    rows=[]
    for k,v in res.items():
        rows.append([k, v.get("n_pairs",""), f"{v.get('flip_rate_to_src',0):.3f}", f"{v.get('change_rate',0):.3f}"])
    headers=["Direction","N pairs","Flip→src","Changed"]
    return headers, rows

def table_baselines(baselines_path: Path):
    records = load_json(baselines_path)
    # each record has true_mode, gold_answer, and B0..B4 strings
    baseline_keys = [k for k in ["B0","B1","B2","B3","B4"] if k in records[0]]
    rows_out=[]

    for b in baseline_keys:
        y_true=[r["true_mode"] for r in records]
        # get text
        def get_text(r):
            return r.get(b,"") if isinstance(r.get(b,""), str) else (r.get(b) or "")
        y_pred=[classify_mode_firstline(get_text(r)) for r in records]
        m={
            "acc": acc(y_true,y_pred),
            "macro_f1": macro_f1(y_true,y_pred),
            "FAR": far_refuse_conflict(y_true,y_pred),
            "ans_sub": answer_substring(records, get_text),
            "pred_counts": dict(Counter(y_pred)),
        }
        rows_out.append([b, f"{m['acc']:.4f}", f"{m['macro_f1']:.4f}", f"{m['FAR']:.4f}", f"{m['ans_sub']:.4f}", str(m["pred_counts"])])
    headers=["Baseline","Acc","Macro-F1","FAR(ref+conf)","Ans substring","Pred counts"]
    return headers, rows_out

# -----------------------------
# Main report writer
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir", default="results/summary_tables")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_lines=[]
    report_lines.append("# Project Results Summary\n")
    report_lines.append("This report is auto-generated from saved experiment artifacts.\n")

    # 1) Prompt dependence
    hd, rows = table_prompt_dependence(results_dir)
    if hd:
        report_lines.append("## Prompt Dependence (train P0 → test P1–P5 on VAL)\n")
        report_lines.append(md_table(hd, rows) + "\n")
        save_csv(out_dir/"prompt_dependence.csv", hd, rows)

    # 2) Steering sweep on VAL (rescored)
    resc_val = results_dir/"rescored_val"/"all_rescored.json"
    if resc_val.exists():
        hd, rows = table_steering_val(resc_val)
        report_lines.append("## Steering Alpha Sweep on VAL (S3 vs S4)\n")
        report_lines.append(md_table(hd, rows) + "\n")
        save_csv(out_dir/"steering_val.csv", hd, rows)

    # 3) Systems on TEST (rescored)
    resc_test = results_dir/"rescored_test"/"ablation_test_final_rescored.json"
    if resc_test.exists():
        hd, rows = table_systems_test(resc_test)
        report_lines.append("## Final Systems on TEST (rescored)\n")
        report_lines.append(md_table(hd, rows) + "\n")
        save_csv(out_dir/"systems_test.csv", hd, rows)

    # 4) Causal patching
    causal = results_dir/"causal_patching_val.json"
    if causal.exists():
        hd, rows = table_causal_patching(causal)
        report_lines.append("## Causal Patching (router representation sufficiency)\n")
        report_lines.append(md_table(hd, rows) + "\n")
        save_csv(out_dir/"causal_patching.csv", hd, rows)

    # 5) Baselines on TEST (if exists)
    base_test = results_dir/"baselines_test.json"
    if base_test.exists():
        hd, rows = table_baselines(base_test)
        report_lines.append("## Baselines on TEST (computed from results/baselines_test.json)\n")
        report_lines.append(md_table(hd, rows) + "\n")
        save_csv(out_dir/"baselines_test.csv", hd, rows)

    # Write markdown report
    report_path = out_dir/"REPORT.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved markdown report -> {report_path}")
    print(f"Saved CSV tables -> {out_dir}")

if __name__ == "__main__":
    main()
