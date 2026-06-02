import json
from collections import defaultdict

path = "steering/steering_sweep_P0_val_hidden_refuse_unitrms_lto1_pso0_10layers_5a_n40.json"
rows = json.load(open(path))

# group by layer
by_layer = defaultdict(list)
for r in rows:
    by_layer[r["layer"]].append(r)

def key(r): 
    return r["alpha"]

print("Layer  best_alpha  comp  FAR   Δcomp  ΔFAR  (relative to alpha=0)")
for layer, lst in sorted(by_layer.items()):
    lst = sorted(lst, key=key)
    base = next((x for x in lst if abs(x["alpha"] - 0.0) < 1e-9), None)
    if base is None:
        continue
    # choose best by (comp - 0.5*FAR) as you did
    best = max(lst, key=lambda x: x["compliance_regex"] - 0.5*x["false_answer_rate_regex"])
    dcomp = best["compliance_regex"] - base["compliance_regex"]
    dfar  = best["false_answer_rate_regex"] - base["false_answer_rate_regex"]
    print(f"{layer:5d} {best['alpha']:10.2f} {best['compliance_regex']:.3f} {best['false_answer_rate_regex']:.3f}"
          f" {dcomp:+.3f} {dfar:+.3f}")
