#!/usr/bin/env python3
"""
7_ablation_final.py
Evaluate system variants on val or test:

S1: Always-answer grounded (no triage)
S2: Prompt-only triage (unified triage prompt)
S3: Router → mode-conditioned generation (exact refuse/conflict strings)
S4: Router → mode-conditioned generation + steering (for routed refuse/conflict)

Important fixes:
  - device_map="auto" safe input placement
  - --split val|test for tuning alpha on val
  - --router_pkl choose dev vs final router
  - steer_layer separate from router layer(s)
  - --alpha_unit rms to match steering sweep scaling
  - outputs per-example JSON + summary metrics JSON

This script is where you tune steer_alpha on VAL, then freeze and run TEST once.
"""

import argparse, json, pickle, re, time, gc
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from prompts import render_prompt, DECODE_CFG, REFUSE_STRING, CONFLICT_STRING

# ──────────────────────────────────────────────────────────────────────────────
# Regex mode classifier (same style as 4e)
# ──────────────────────────────────────────────────────────────────────────────
MODE_PATTERNS = {
    "refuse": re.compile(
        r"\b("
        r"not enough (information|evidence)|insufficient (information|evidence)|"
        r"cannot (answer|determine|verify)|can't (answer|determine|verify)|"
        r"don't (know|have enough)|no (information|evidence)|"
        r"unable to (answer|determine|verify)|"
        r"cannot be answered from the provided (documents|context)|"
        r"provided documents do not (contain|include) (enough|sufficient)|"
        r")\b",
        re.IGNORECASE,
    ),
    "conflict": re.compile(
        r"\b("
        r"conflict|contradict|contradiction|inconsistent|disagree|"
        r"mutually incompatible|cannot reconcile|at odds|"
        r"conflicting (information|claims)|"
        r")\b",
        re.IGNORECASE,
    ),
}

def classify_mode_regex(text: str) -> str:
    t = (text or "").strip()
    if MODE_PATTERNS["conflict"].search(t):
        return "conflict"
    if MODE_PATTERNS["refuse"].search(t):
        return "refuse"
    return "answer"

def normalize_text(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).split()

def substring_match(gold: str, pred: str) -> float:
    g = (gold or "").strip().lower()
    p = (pred or "").strip().lower()
    if not g:
        return 0.0
    return 1.0 if g in p else 0.0

# ──────────────────────────────────────────────────────────────────────────────
# device_map-safe helpers
# ──────────────────────────────────────────────────────────────────────────────
def get_layers(m):
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model.layers
    if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
        return m.transformer.h
    if hasattr(m, "model") and hasattr(m.model, "decoder") and hasattr(m.model.decoder, "layers"):
        return m.model.decoder.layers
    raise AttributeError(f"Cannot find decoder layers for {type(m)}")

def get_input_device(model):
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        for k in ["model.embed_tokens", "model.model.embed_tokens", "transformer.wte"]:
            if k in model.hf_device_map:
                return torch.device(model.hf_device_map[k])
        return torch.device(next(iter(model.hf_device_map.values())))
    return next(model.parameters()).device

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ibm-granite/granite-3.1-8b-instruct")
    ap.add_argument("--split", default="val", choices=["val","test"])
    ap.add_argument("--max_entries", type=int, default=None)

    ap.add_argument("--router_pkl", default="results/router_hidden_dev.pkl",
                    help="Use dev router on val; use final router on test.")
    ap.add_argument("--template", default="P0")

    ap.add_argument("--steer_layer", type=int, default=21,
                    help="Layer to apply steering (from 4e sweep).")
    ap.add_argument("--steer_alpha", type=float, default=0.0,
                    help="Alpha in *alpha_unit* (see --alpha_unit).")
    ap.add_argument("--alpha_unit", choices=["rms","none"], default="rms",
                    help="rms = multiply steer_alpha by rms_by_layer_hidden[steer_layer].")
    ap.add_argument("--prompt_step_only", type=int, default=0,
                    help="1=steer only at first generation step (prefill-only); 0=every step.")
    ap.add_argument("--last_token_only", type=int, default=1,
                    help="1=steer last token only; 0=steer all positions.")
    ap.add_argument("--save_path", default=None)

    args = ap.parse_args()

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    blocks = get_layers(model)
    L = len(blocks)
    D = model.config.hidden_size
    input_device = get_input_device(model)

    # Load router
    with open(args.router_pkl, "rb") as f:
        rdata = pickle.load(f)
    router = rdata["router"]
    scaler = rdata["scaler"]
    router_layers = rdata["layers"]  # [layer] or [l0,l1]
    feat_type = rdata["feature_type"]

    print(f"Router loaded: {args.router_pkl} | layers={router_layers} feat={feat_type}")
    print(f"Model: L={L} D={D} | steer_layer={args.steer_layer} alpha={args.steer_alpha} unit={args.alpha_unit}")

    # Load steering vectors + RMS
    V_hidden = np.load("steering/steering_vectors_hidden.npy").astype(np.float32)  # [L,2,D]
    if V_hidden.shape[2] != D:
        raise RuntimeError(f"Vector dim mismatch: V_hidden D={V_hidden.shape[2]} vs model D={D}. Recompute vectors for this model.")
    if args.steer_layer >= V_hidden.shape[0]:
        raise RuntimeError(f"steer_layer {args.steer_layer} out of range for vectors with L={V_hidden.shape[0]}")

    rms_by_layer = None
    if args.alpha_unit == "rms":
        rms_by_layer = np.load("steering/rms_by_layer_hidden.npy").astype(np.float32)
        if args.steer_layer >= len(rms_by_layer):
            raise RuntimeError(f"Missing rms for steer_layer={args.steer_layer}")

    def alpha_eff():
        if args.steer_alpha == 0.0:
            return 0.0
        if args.alpha_unit == "rms":
            return float(args.steer_alpha) * float(rms_by_layer[args.steer_layer])
        return float(args.steer_alpha)

    # Unified triage prompt (S2)
    TRIAGE_INSTR = (
        "You must use ONLY the provided documents. Do NOT use outside knowledge.\n"
        f"If the documents lack sufficient information, respond exactly: {REFUSE_STRING}\n"
        f"If the documents contain conflicting information, respond exactly: {CONFLICT_STRING}\n"
        "Otherwise, answer directly using only the documents.\n"
        "If documents conflict, do NOT choose a side and do NOT guess."
    )

    MODE_INSTR = {
        "answer":   "Answer the question directly using only the documents provided.",
        "refuse":   f"Respond exactly: {REFUSE_STRING}",
        "conflict": f"Respond exactly: {CONFLICT_STRING}",
    }
    LABEL_INV = {0: "answer", 1: "refuse", 2: "conflict"}

    # Steering hook state
    current_steer = {
        "layer": None,
        "vector": None,      # torch [D]
        "alpha_eff": 0.0,
        "prompt_len": 0,
        "prompt_step_only": bool(args.prompt_step_only),
        "last_token_only": bool(args.last_token_only),
    }

    def make_hook(layer_idx: int):
        def hook(_module, _inp, output):
            if current_steer["layer"] != layer_idx:
                return None
            if current_steer["alpha_eff"] == 0.0:
                return None

            is_tuple = isinstance(output, tuple)
            h = output[0] if is_tuple else output  # [B,T,D]
            if not torch.is_tensor(h) or h.ndim != 3:
                return None

            # prefill-only
            if current_steer["prompt_step_only"]:
                if h.shape[1] != current_steer["prompt_len"]:
                    return None

            v = current_steer["vector"].to(h.device).to(h.dtype)  # [D]
            a = current_steer["alpha_eff"]

            h2 = h.clone()
            if current_steer["last_token_only"]:
                h2[:, -1, :] = h2[:, -1, :] + a * v
            else:
                h2 = h2 + a * v.view(1, 1, -1)

            if is_tuple:
                return (h2,) + output[1:]
            return h2
        return hook

    hooks = []
    for li, block in enumerate(blocks):
        hooks.append(block.register_forward_hook(make_hook(li)))
    print(f"Registered {len(hooks)} hooks.")

    # Load instances
    inst_path = Path(f"instances_{args.split}.jsonl")
    instances = [json.loads(l) for l in open(inst_path, "r", encoding="utf-8")]
    if args.max_entries:
        instances = instances[:args.max_entries]
    print(f"[{time.strftime('%H:%M:%S')}] Loaded {len(instances)} instances from {inst_path}")

    def to_device(batch):
        return {k: v.to(input_device) for k, v in batch.items()}

    def route_mode(inst) -> str:
        # Build a neutral prompt for routing features (no extra mode instruction)
        prompt = render_prompt(args.template, inst["question"], inst["docs"], tokenizer=tokenizer)
        enc = to_device(tokenizer(prompt, return_tensors="pt"))
        with torch.inference_mode():
            out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states  # tuple len L+1

        if feat_type == "hidden_last_token":
            l = router_layers[0]
            vec = hs[l + 1][0, -1, :].detach().cpu().float().numpy().reshape(1, -1)
        else:
            # band2 flat
            l0, l1 = router_layers
            v0 = hs[l0 + 1][0, -1, :].detach().cpu().float().numpy()
            v1 = hs[l1 + 1][0, -1, :].detach().cpu().float().numpy()
            vec = np.concatenate([v0, v1]).reshape(1, -1)

        vec_s = scaler.transform(vec)
        pred = int(router.predict(vec_s)[0])
        return LABEL_INV[pred]

    def generate(question, docs, instruction: str, do_steer: bool, steer_mode: str) -> str:
        prompt = render_prompt(args.template, question, docs, instruction=instruction, tokenizer=tokenizer)
        enc = to_device(tokenizer(prompt, return_tensors="pt"))
        inp_len = enc["input_ids"].shape[1]

        if do_steer and args.steer_alpha != 0.0 and steer_mode in ("refuse", "conflict"):
            m_idx = {"refuse": 0, "conflict": 1}[steer_mode]
            vec = torch.tensor(V_hidden[args.steer_layer, m_idx, :], dtype=torch.float32)

            current_steer.update({
                "layer": args.steer_layer,
                "vector": vec,
                "alpha_eff": alpha_eff(),
                "prompt_len": int(inp_len),
            })
        else:
            current_steer["alpha_eff"] = 0.0
            current_steer["layer"] = None

        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=DECODE_CFG["max_new_tokens"],
                do_sample=DECODE_CFG["do_sample"],
                temperature=DECODE_CFG["temperature"],
                top_p=DECODE_CFG["top_p"],
                repetition_penalty=DECODE_CFG["repetition_penalty"],
                pad_token_id=tokenizer.eos_token_id,
            )

        # turn off steering
        current_steer["alpha_eff"] = 0.0
        current_steer["layer"] = None

        text = tokenizer.decode(out[0][inp_len:], skip_special_tokens=True).strip()
        del enc, out
        return text

    # Evaluate
    rows_out = []
    for i, inst in enumerate(tqdm(instances, desc="ablation")):
        true_mode = inst["true_mode"]
        q, docs, gold = inst["question"], inst["docs"], inst.get("gold_answer","")

        # S1: always answer (grounded)
        s1 = generate(q, docs, instruction=MODE_INSTR["answer"], do_steer=False, steer_mode="answer")

        # S2: prompt-only triage
        s2 = generate(q, docs, instruction=TRIAGE_INSTR, do_steer=False, steer_mode="answer")

        # route
        routed = route_mode(inst)

        # S3: router mode instruction (no steering)
        s3 = generate(q, docs, instruction=MODE_INSTR[routed], do_steer=False, steer_mode=routed)

        # S4: router mode instruction + steering for routed refuse/conflict
        s4 = generate(q, docs, instruction=MODE_INSTR[routed], do_steer=True, steer_mode=routed)

        def answer_metric(pred: str) -> float:
            return substring_match(gold, pred)

        def mode_success(true_mode: str, pred_text: str) -> float:
            # strict-ish: check canonical strings for refuse/conflict, and substring for answer
            if true_mode == "refuse":
                return 1.0 if REFUSE_STRING.lower() in pred_text.lower() else 0.0
            if true_mode == "conflict":
                return 1.0 if CONFLICT_STRING.lower() in pred_text.lower() else 0.0
            return answer_metric(pred_text)

        rows_out.append({
            "id": inst["id"],
            "true_mode": true_mode,
            "routed_mode": routed,
            "gold_answer": gold,
            "S1": {"text": s1, "pred_mode": classify_mode_regex(s1), "score": mode_success(true_mode, s1)},
            "S2": {"text": s2, "pred_mode": classify_mode_regex(s2), "score": mode_success(true_mode, s2)},
            "S3": {"text": s3, "pred_mode": classify_mode_regex(s3), "score": mode_success(true_mode, s3)},
            "S4": {"text": s4, "pred_mode": classify_mode_regex(s4), "score": mode_success(true_mode, s4)},
        })

        if (i + 1) % 25 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # Summaries
    def summarize(cond: str):
        pred = [r[cond]["pred_mode"] for r in rows_out]
        true = [r["true_mode"] for r in rows_out]

        # accuracy
        acc = sum(t == p for t, p in zip(true, pred)) / len(true)

        # macro-F1 (simple)
        labels = ["answer","refuse","conflict"]
        f1s = []
        for lab in labels:
            tp = sum((t==lab and p==lab) for t,p in zip(true,pred))
            fp = sum((t!=lab and p==lab) for t,p in zip(true,pred))
            fn = sum((t==lab and p!=lab) for t,p in zip(true,pred))
            prec = tp/(tp+fp+1e-9)
            rec  = tp/(tp+fn+1e-9)
            f1 = 2*prec*rec/(prec+rec+1e-9)
            f1s.append(f1)
        macro_f1 = float(np.mean(f1s))

        # FAR on true refuse/conflict
        far = sum((t in ("refuse","conflict") and p=="answer") for t,p in zip(true,pred)) / \
              max(1, sum(t in ("refuse","conflict") for t in true))

        # answer substring on true answer
        ans_idx = [i for i,t in enumerate(true) if t=="answer"]
        ans_sub = float(np.mean([substring_match(rows_out[i]["gold_answer"], rows_out[i][cond]["text"]) for i in ans_idx])) if ans_idx else 0.0

        return {"acc": acc, "macro_f1": macro_f1, "FAR_refuse+conflict": far, "answer_substring": ans_sub}

    summary = {
        "meta": {
            "model": args.model,
            "split": args.split,
            "router_pkl": args.router_pkl,
            "router_layers": router_layers,
            "steer_layer": args.steer_layer,
            "steer_alpha": args.steer_alpha,
            "alpha_unit": args.alpha_unit,
            "alpha_eff": alpha_eff(),
            "prompt_step_only": bool(args.prompt_step_only),
            "last_token_only": bool(args.last_token_only),
            "n": len(rows_out),
        },
        "S1": summarize("S1"),
        "S2": summarize("S2"),
        "S3": summarize("S3"),
        "S4": summarize("S4"),
    }

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    if args.save_path is None:
        args.save_path = f"results/ablation_{args.split}_alpha{args.steer_alpha}_{args.alpha_unit}_layer{args.steer_layer}.json"
    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows_out}, f, indent=2)

    print("\nSummary:")
    print(json.dumps(summary, indent=2))

    for h in hooks:
        h.remove()

if __name__ == "__main__":
    main()
