#!/usr/bin/env python3
"""
base_rag.py  —  RAG inference with triage-prompt support.

Fixes applied:
  1. TRIAGEINSTR defined locally (was never in prompts.py)
  2. Tokenizer loaded with trust_remote_code=True and local_files_only=False
     so a corrupt local tokenizer.model does not crash the run.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import render_prompt, DECODE_CFG, REFUSE_STRING, CONFLICT_STRING

# ---------------------------------------------------------------------------
# FIX 1: define TRIAGEINSTR locally — it was never exported from prompts.py
# ---------------------------------------------------------------------------
TRIAGEINSTR = (
    "You must use ONLY the provided documents. Do NOT use outside knowledge.\n"
    f"If the documents lack sufficient information, respond exactly: {REFUSE_STRING}\n"
    f"If the documents contain conflicting information, respond exactly: {CONFLICT_STRING}\n"
    "Otherwise, answer directly using only the documents.\n"
    "If documents conflict, do NOT choose a side and do NOT guess."
)

# ---------------------------------------------------------------------------
# Model registry  (use HF hub IDs so tokenizer download is always clean)
# ---------------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, str] = {
    "qwen3-4b":    "Qwen/Qwen3-4B",
    "qwen3-8b":    "Qwen/Qwen3-8B",
    "mistral-7b":  "mistralai/Mistral-7B-Instruct-v0.2",
    "granite-8b":  "ibm-granite/granite-3.1-8b-instruct",
    "selfrag":     "selfrag/selfrag_llama2_7b",
    "chatqa":      "nvidia/Llama3-ChatQA-1.5-8B",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_model_id(model_arg: str) -> str:
    """Accept a registry key OR a raw HF hub ID / local path."""
    return MODEL_REGISTRY.get(model_arg, model_arg)


def load_tokenizer(model_id: str, local_model_path: Optional[str] = None):
    """
    FIX 2: load tokenizer robustly.
    If a local path is supplied we try it first; on ANY error we fall back
    to the HF hub ID so a corrupt local tokenizer.model never kills the run.
    """
    # Try local path first (fast, no network)
    if local_model_path and Path(local_model_path).exists():
        try:
            tok = AutoTokenizer.from_pretrained(
                local_model_path,
                trust_remote_code=True,
                local_files_only=True,
            )
            print(f"[tokenizer] loaded from local path: {local_model_path}")
            return tok
        except Exception as e:
            print(
                f"[tokenizer] WARNING: local load failed ({e}). "
                f"Falling back to HF hub: {model_id}"
            )

    # Fall back / primary: pull from HF hub
    tok = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=False,
    )
    print(f"[tokenizer] loaded from HF hub: {model_id}")
    return tok


def load_model(model_id: str, local_model_path: Optional[str] = None):
    """Load model weights; prefer local path if valid, else HF hub."""
    load_path = (
        local_model_path
        if (local_model_path and Path(local_model_path).exists())
        else model_id
    )
    print(f"[model] loading weights from: {load_path}")
    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model


def load_instances(path: str) -> List[Dict[str, Any]]:
    instances = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                instances.append(json.loads(line))
    return instances


def classify_output(text: str) -> str:
    """Map generated text to one of: answer / refuse / conflict."""
    t = text.strip().lower()
    if REFUSE_STRING.lower() in t:
        return "refuse"
    if CONFLICT_STRING.lower() in t:
        return "conflict"
    return "answer"


def truncate_to_max_tokens(
    text: str,
    tokenizer,
    max_tokens: int = 3500,
) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > max_tokens:
        ids = ids[:max_tokens]
        text = tokenizer.decode(ids, skip_special_tokens=True)
    return text


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_triage_prompt(
    question: str,
    docs: List[str],
    tokenizer,
    use_chat_template: bool = True,
    max_ctx_tokens: int = 3500,
) -> str:
    """Build a unified triage prompt using TRIAGEINSTR."""
    doc_block = "\n\n".join(
        f"[Document {i+1}]\n{d}" for i, d in enumerate(docs)
    )
    user_content = (
        f"{TRIAGEINSTR}\n\n"
        f"Documents:\n{doc_block}\n\n"
        f"Question: {question}"
    )
    user_content = truncate_to_max_tokens(user_content, tokenizer, max_ctx_tokens)

    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": user_content}]
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return prompt
        except Exception:
            pass  # fall through to raw format

    return f"<|user|>\n{user_content}\n<|assistant|>\n"


def build_standard_prompt(
    question: str,
    docs: List[str],
    mode: str,
    template_key: str,
    tokenizer,
    max_ctx_tokens: int = 3500,
) -> str:
    """Build a standard RAG prompt via prompts.render_prompt."""
    doc_block = "\n\n".join(
        f"[Document {i+1}]\n{d}" for i, d in enumerate(docs)
    )
    raw = render_prompt(
        template_key=template_key,
        question=question,
        docs=doc_block,
        mode=mode,
    )
    return truncate_to_max_tokens(raw, tokenizer, max_ctx_tokens)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate(
    prompt: str,
    model,
    tokenizer,
    decode_cfg: Optional[Dict] = None,
) -> str:
    cfg = {**DECODE_CFG, **(decode_cfg or {})}
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    out = model.generate(
        **inputs,
        max_new_tokens=cfg.get("max_new_tokens", 256),
        do_sample=cfg.get("do_sample", False),
        temperature=cfg.get("temperature", 1.0),
        top_p=cfg.get("top_p", 1.0),
        pad_token_id=tokenizer.eos_token_id,
    )
    new_ids = out[0][input_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    instances: List[Dict[str, Any]],
    model,
    tokenizer,
    args,
) -> List[Dict[str, Any]]:
    results = []
    for idx, inst in enumerate(instances):
        question = inst["question"]
        docs     = inst.get("docs", [])
        gold     = inst.get("label", inst.get("mode", "answer"))

        if args.triage_mode:
            prompt = build_triage_prompt(
                question, docs, tokenizer,
                use_chat_template=not args.no_chat_template,
                max_ctx_tokens=args.max_ctx_tokens,
            )
        else:
            prompt = build_standard_prompt(
                question, docs,
                mode=inst.get("mode", "answer"),
                template_key=args.template,
                tokenizer=tokenizer,
                max_ctx_tokens=args.max_ctx_tokens,
            )

        raw_output = generate(prompt, model, tokenizer)
        pred_label = classify_output(raw_output)

        results.append({
            "id":         inst.get("id", idx),
            "question":   question,
            "gold":       gold,
            "pred":       pred_label,
            "output":     raw_output,
            "correct":    int(pred_label == gold),
        })

        if (idx + 1) % 50 == 0:
            acc_so_far = sum(r["correct"] for r in results) / len(results)
            print(f"  [{idx+1}/{len(instances)}]  acc={acc_so_far:.3f}")

    return results


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    total   = len(results)
    correct = sum(r["correct"] for r in results)
    by_class: Dict[str, Dict[str, int]] = {}
    for r in results:
        g = r["gold"]
        p = r["pred"]
        if g not in by_class:
            by_class[g] = {"tp": 0, "fp": 0, "fn": 0}
        if p not in by_class:
            by_class[p] = {"tp": 0, "fp": 0, "fn": 0}
        if g == p:
            by_class[g]["tp"] += 1
        else:
            by_class[g]["fn"] += 1
            by_class[p]["fp"] += 1

    metrics: Dict[str, float] = {"accuracy": correct / total if total else 0.0}
    for cls, counts in by_class.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        metrics[f"{cls}_precision"] = prec
        metrics[f"{cls}_recall"]    = rec
        metrics[f"{cls}_f1"]        = f1

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="RAG baseline inference")
    p.add_argument("--model",         required=True,
                   help="Registry key (e.g. qwen3-4b) or HF hub ID")
    p.add_argument("--model_path",    default=None,
                   help="Optional local path; falls back to HF hub if corrupt")
    p.add_argument("--data",          required=True,
                   help="Path to instances JSONL (test or val split)")
    p.add_argument("--output",        default="results.jsonl",
                   help="Path to write per-instance results")
    p.add_argument("--template",      default="P0",
                   help="Prompt template key (P0–P5) for non-triage mode")
    p.add_argument("--triage_mode",   action="store_true",
                   help="Use unified TRIAGEINSTR triage prompt")
    p.add_argument("--no_chat_template", action="store_true",
                   help="Skip apply_chat_template even if available")
    p.add_argument("--max_ctx_tokens", type=int, default=3500,
                   help="Max tokens for the context block")
    p.add_argument("--limit",         type=int, default=None,
                   help="Evaluate only the first N instances (debugging)")
    return p.parse_args()


def main():
    args = parse_args()
    model_id = get_model_id(args.model)

    print(f"Model ID : {model_id}")
    print(f"Data     : {args.data}")
    print(f"Triage   : {args.triage_mode}")

    # Load tokenizer (FIX 2 is inside load_tokenizer)
    tokenizer = load_tokenizer(model_id, local_model_path=args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = load_model(model_id, local_model_path=args.model_path)

    # Load data
    instances = load_instances(args.data)
    if args.limit:
        instances = instances[: args.limit]
    print(f"Instances: {len(instances)}")

    # Run evaluation
    results = evaluate(instances, model, tokenizer, args)

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Print metrics
    metrics = compute_metrics(results)
    print("\n=== Metrics ===")
    for k, v in sorted(metrics.items()):
        print(f"  {k:30s}: {v:.4f}")

    metrics_path = out_path.with_suffix(".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults  → {out_path}")
    print(f"Metrics  → {metrics_path}")


if __name__ == "__main__":
    main()
