import json, numpy as np
from itertools import product
from collections import Counter

REFUSE_STRING = "Not enough information."
CONFLICT_STRING = "Documents contain conflicting information."

def classify_mode(text: str) -> str:
    t = (text or "").strip().lower()
    if "conflict" in t or "contradict" in t or CONFLICT_STRING.lower() in t:
        return "conflict"
    if "not enough" in t or "insufficient" in t or REFUSE_STRING.lower() in t:
        return "refuse"
    return "answer"

def macro_f1(y_true, y_pred, labels=("answer","refuse","conflict")):
    f1s=[]
    for lab in labels:
        tp=sum(1 for yt,yp in zip(y_true,y_pred) if yt==lab and yp==lab)
        fp=sum(1 for yt,yp in zip(y_true,y_pred) if yt!=lab and yp==lab)
        fn=sum(1 for yt,yp in zip(y_true,y_pred) if yt==lab and yp!=lab)
        prec = tp/(tp+fp+1e-9)
        rec  = tp/(tp+fn+1e-9)
        f1   = 2*prec*rec/(prec+rec+1e-9)
        f1s.append(f1)
    return float(np.mean(f1s))

records = json.load(open("results/baselines_val.json"))

# We'll tune B3 rule thresholds (top1, gap, entropy) on val.
# You can adjust grid density.
top1_grid = [0.2, 0.3, 0.4, 0.5, 0.6]
gap_grid  = [0.05, 0.10, 0.15, 0.20, 0.25]
ent_grid  = [0.6, 0.8, 1.0, 1.2, 1.4]

best = None
best_score = -1

for top1_th, gap_th, ent_th in product(top1_grid, gap_grid, ent_grid):
    y_true=[]
    y_pred=[]
    for r in records:
        y_true.append(r["true_mode"])
        ent = r.get("B3_ent", 0.0)
        top1 = r.get("B3_top1", 0.0)
        gap = r.get("B3_gap", 0.0)

        if ent > ent_th:
            out = CONFLICT_STRING
        elif top1 < top1_th or gap < gap_th:
            out = REFUSE_STRING
        else:
            out = r.get("B1","")  # use grounded answer as the "answer" branch

        y_pred.append(classify_mode(out))

    score = macro_f1(y_true, y_pred)
    if score > best_score:
        best_score = score
        best = (top1_th, gap_th, ent_th)

print("Best thresholds (val):")
print(" top1_thresh =", best[0])
print(" gap_thresh  =", best[1])
print(" ent_thresh  =", best[2])
print(" macroF1     =", best_score)

out = {
    "top1_thresh": best[0],
    "gap_thresh": best[1],
    "ent_thresh": best[2],
    "b4_nll_thresh": 3.5  # keep default unless you also tune B4
}
json.dump(out, open("b3_thresholds.json","w"), indent=2)
print("Saved → b3_thresholds.json")
