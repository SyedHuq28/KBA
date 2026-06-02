# generate_layers_md.py
"""
Reads results/layer_probe_hidden.json and results/layer_probe_mlp.json
and writes layers.md. No retraining — just loads saved results.
"""

import json
from pathlib import Path

RESULTS_DIR = Path("results")
OUT_FILE    = Path("layers.md")

# ── Load ───────────────────────────────────────────────────────────────────────

def load_probe(fname):
    path = RESULTS_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run 3_layer_probe_sweep.py first.")
    with open(path, "r", encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f).items()}

results_h = load_probe("layer_probe_hidden.json")  # val_acc, val_f1, test_acc
results_m = load_probe("layer_probe_mlp.json")      # val_acc, val_f1

L = max(max(results_h.keys()), max(results_m.keys())) + 1

# ── Bests ──────────────────────────────────────────────────────────────────────

best_h_layer = max(results_h, key=lambda k: results_h[k]["val_acc"])
best_m_layer = max(results_m, key=lambda k: results_m[k]["val_acc"])
best_h_acc   = results_h[best_h_layer]["val_acc"]
best_m_acc   = results_m[best_m_layer]["val_acc"]

if best_h_acc >= best_m_acc:
    overall_best_feature = "Hidden State (H)"
    overall_best_layer   = best_h_layer
    overall_best         = results_h[best_h_layer]
    overall_test         = overall_best.get("test_acc")
else:
    overall_best_feature = "MLP Output (M)"
    overall_best_layer   = best_m_layer
    overall_best         = results_m[best_m_layer]
    overall_test         = None  # MLP has no test_acc in script 3

# ── Helpers ────────────────────────────────────────────────────────────────────

def pct(v): return f"{v*100:.2f}%" if v is not None else "—"
def fmt(v): return f"{v:.4f}"      if v is not None else "—"

# ── Build markdown ─────────────────────────────────────────────────────────────

lines = []

lines += [
    "# Layer Probe Sweep — Results",
    "",
    "Multinomial logistic regression probes (`C=1.0`, `lbfgs`) trained on **train** split,",
    "evaluated on **val** and **test**. Labels: `0=answer · 1=refuse · 2=conflict`.",
    "",
]

# ── Overall best ───────────────────────────────────────────────────────────────
lines += [
    "## 🏆 Overall Best",
    "",
    "| | |",
    "|---|---|",
    f"| **Feature Space** | {overall_best_feature} |",
    f"| **Layer**         | L{overall_best_layer} |",
    f"| **Val Accuracy**  | {pct(overall_best['val_acc'])} |",
    f"| **Val Macro F1**  | {fmt(overall_best['val_f1'])} |",
    f"| **Test Accuracy** | {pct(overall_test)} |",
    "",
]

# ── Best per feature ───────────────────────────────────────────────────────────
bh = results_h[best_h_layer]
bm = results_m[best_m_layer]

lines += [
    "## Best Per Feature Space",
    "",
    f"| Metric | Hidden H (L{best_h_layer}) | MLP M (L{best_m_layer}) |",
    "|---|---|---|",
    f"| Val Accuracy  | {pct(bh['val_acc'])}        | {pct(bm['val_acc'])} |",
    f"| Val Macro F1  | {fmt(bh['val_f1'])}         | {fmt(bm['val_f1'])} |",
    f"| Test Accuracy | {pct(bh.get('test_acc'))}   | — |",
    "",
]

# ── Top 5 H ────────────────────────────────────────────────────────────────────
lines += [
    "## Top 5 Layers — Hidden State (H)",
    "",
    "| Rank | Layer | Val Acc | Val F1 | Test Acc |",
    "|---|---|---|---|---|",
]
for rank, (l, v) in enumerate(
    sorted(results_h.items(), key=lambda kv: kv[1]["val_acc"], reverse=True)[:5], 1
):
    lines.append(
        f"| {rank} | L{l} | {pct(v['val_acc'])} | {fmt(v['val_f1'])} | {pct(v.get('test_acc'))} |"
    )
lines.append("")

# ── Top 5 M ────────────────────────────────────────────────────────────────────
lines += [
    "## Top 5 Layers — MLP Output (M)",
    "",
    "| Rank | Layer | Val Acc | Val F1 |",
    "|---|---|---|---|",
]
for rank, (l, v) in enumerate(
    sorted(results_m.items(), key=lambda kv: kv[1]["val_acc"], reverse=True)[:5], 1
):
    lines.append(f"| {rank} | L{l} | {pct(v['val_acc'])} | {fmt(v['val_f1'])} |")
lines.append("")

# ── Full per-layer table ───────────────────────────────────────────────────────
lines += [
    "## Full Per-Layer Results",
    "",
    "★H = best hidden layer · ★M = best MLP layer",
    "",
    "| Layer | H Val Acc | H Val F1 | H Test Acc | M Val Acc | M Val F1 |",
    "|---|---|---|---|---|---|",
]

for l in range(L):
    h = results_h.get(l)
    m = results_m.get(l)

    label = f"L{l}"
    if   l == best_h_layer == best_m_layer: label = f"**L{l} ★H★M**"
    elif l == best_h_layer:                 label = f"**L{l} ★H**"
    elif l == best_m_layer:                 label = f"**L{l} ★M**"

    lines.append(
        f"| {label} "
        f"| {pct(h['val_acc']) if h else '—'} "
        f"| {fmt(h['val_f1'])  if h else '—'} "
        f"| {pct(h.get('test_acc')) if h else '—'} "
        f"| {pct(m['val_acc']) if m else '—'} "
        f"| {fmt(m['val_f1'])  if m else '—'} |"
    )

lines += [
    "",
    "> MLP test accuracy not saved by `3_layer_probe_sweep.py` — only val is computed for M.",
    f"*Generated from `{RESULTS_DIR}/layer_probe_hidden.json` + `layer_probe_mlp.json`.*",
]

OUT_FILE.write_text("\n".join(lines), encoding="utf-8")
print(f"Written → {OUT_FILE}")
print(f"Best: {overall_best_feature} L{overall_best_layer} | "
      f"val_acc={pct(overall_best['val_acc'])} | test_acc={pct(overall_test)} | "
      f"val_f1={fmt(overall_best['val_f1'])}")
