import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import DECODE_CFG, REFUSE_STRING, CONFLICT_STRING


# ---------------------------------------------------------------------------
# Base policy strings
# ---------------------------------------------------------------------------

TRIAGEINSTR = (
    "You must use ONLY the provided documents. Do NOT use outside knowledge.\n"
    f"If the documents lack sufficient information, respond exactly: {REFUSE_STRING}\n"
    f"If the documents contain conflicting information, respond exactly: {CONFLICT_STRING}\n"
    "Otherwise, answer directly using only the documents.\n"
    "If documents conflict, do NOT choose a side and do NOT guess."
)

OPIN_INSTR_LINE = (
    "Instruction: Read the given information and answer the corresponding question "
    "based only on that information."
)

KRE_INSTR = (
    "You are given documents and a question. The documents may be incomplete, misleading, "
    "or may conflict with one another, so judge carefully."
)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, str] = {
    "qwen3-4b":   "Qwen/Qwen3-4B",
    "fal":   "tiiuae/Falcon-H1-Tiny-90M-Instruct",
    "falb":   "tiiuae/Falcon3-7B-Instruct-1.58bit",
    "moe":   "microsoft/Phi-mini-MoE-instruct",
    "llama_base":   "meta-llama/Llama-3.2-3B",
    "granite":   "ibm-granite/granite-3.1-8b-instruct",
    "o1":   "allenai/OLMo-2-0425-1B",
    "o2":   "allenai/OLMo-2-0425-1B-SFT",
    "o3":   "allenai/OLMo-2-0425-1B-DPO",
    "o4":   "allenai/OLMo-2-0425-1B-Instruct",
    "o5":   "allenai/OLMo-3-7B-Instruct"

}

LABEL_ID_TO_NAME = {
    0: "answer",
    1: "refuse",
    2: "conflict",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_model_id(model_arg: str) -> str:
    return MODEL_REGISTRY.get(model_arg, model_arg)


def load_tokenizer(model_id: str, local_model_path: Optional[str] = None):
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

    tok = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=False,
    )
    print(f"[tokenizer] loaded from HF hub: {model_id}")
    return tok


def load_model(model_id: str, local_model_path: Optional[str] = None):
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
    instances: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                instances.append(json.loads(line))
    return instances


def get_gold_label(inst: Dict[str, Any]) -> str:
    if "true_mode" in inst and isinstance(inst["true_mode"], str):
        return inst["true_mode"]

    if "mode" in inst and isinstance(inst["mode"], str):
        return inst["mode"]

    if "label" in inst:
        lab = inst["label"]
        if isinstance(lab, int):
            return LABEL_ID_TO_NAME.get(lab, str(lab))
        if isinstance(lab, str):
            return lab

    return "answer"


def classify_output(text: str) -> str:
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


def build_doc_block(docs: List[str]) -> str:
    return "\n\n".join(
        f"[Document {i+1}]\n{d}" for i, d in enumerate(docs)
    )


def apply_chat_or_raw(
    user_content: str,
    tokenizer,
    use_chat_template: bool = True,
) -> str:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": user_content}]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    return f"<|user|>\n{user_content}\n<|assistant|>\n"


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_opin_instr_prompt(
    question: str,
    docs: List[str],
    tokenizer,
    use_chat_template: bool = True,
    max_ctx_tokens: int = 3500,
) -> str:
    doc_block = build_doc_block(docs)
    user_content = (
        f"{OPIN_INSTR_LINE}\n\n"
        f'Bob said:\n"{doc_block}"\n\n'
        f"{TRIAGEINSTR}\n\n"
        f"Question: {question} in Bob's opinion?\n"
        "Answer:"
    )
    user_content = truncate_to_max_tokens(user_content, tokenizer, max_ctx_tokens)
    return apply_chat_or_raw(user_content, tokenizer, use_chat_template)


def build_kre_prompt(
    question: str,
    docs: List[str],
    tokenizer,
    use_chat_template: bool = True,
    max_ctx_tokens: int = 3500,
) -> str:
    doc_block = build_doc_block(docs)
    user_content = (
        f"{KRE_INSTR}\n\n"
        f"{TRIAGEINSTR}\n\n"
        f"Documents:\n{doc_block}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    user_content = truncate_to_max_tokens(user_content, tokenizer, max_ctx_tokens)
    return apply_chat_or_raw(user_content, tokenizer, use_chat_template)


PROMPT_BUILDERS = {
    "opin_instr": build_opin_instr_prompt,
    "kre": build_kre_prompt,
}


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate(
    prompt: str,
    model,
    tokenizer,
    decode_cfg: Optional[Dict[str, Any]] = None,
) -> str:
    cfg = {**DECODE_CFG, **(decode_cfg or {})}

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    out = model.generate(
        **inputs,
        max_new_tokens=cfg.get("max_new_tokens", 256),
        do_sample=cfg.get("do_sample", False),
        temperature=cfg.get("temperature", 1.0),
        top_p=cfg.get("top_p", 1.0),
        repetition_penalty=cfg.get("repetition_penalty", 1.0),
        pad_token_id=tokenizer.eos_token_id,
    )
    new_ids = out[0][input_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    instances: List[Dict[str, Any]],
    model,
    tokenizer,
    prompt_style: str,
    args,
) -> List[Dict[str, Any]]:
    builder = PROMPT_BUILDERS[prompt_style]
    results: List[Dict[str, Any]] = []

    for idx, inst in enumerate(instances):
        question = inst["question"]
        docs = inst.get("docs", [])
        gold = get_gold_label(inst)

        prompt = builder(
            question=question,
            docs=docs,
            tokenizer=tokenizer,
            use_chat_template=not args.no_chat_template,
            max_ctx_tokens=args.max_ctx_tokens,
        )

        raw_output = generate(prompt, model, tokenizer)
        pred_label = classify_output(raw_output)

        results.append({
            "id": inst.get("id", idx),
            "question": question,
            "gold": gold,
            "pred": pred_label,
            "output": raw_output,
            "correct": int(pred_label == gold),
            "prompt_style": prompt_style,
        })

        if (idx + 1) % 50 == 0:
            acc_so_far = sum(r["correct"] for r in results) / len(results)
            print(f"[{prompt_style}] [{idx+1}/{len(instances)}] acc={acc_so_far:.3f}")

    return results


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    total = len(results)
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

    metrics: Dict[str, float] = {
        "accuracy": correct / total if total else 0.0,
        "n": float(total),
    }

    macro_f1_vals = []
    for cls, counts in sorted(by_class.items()):
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        metrics[f"{cls}_precision"] = prec
        metrics[f"{cls}_recall"] = rec
        metrics[f"{cls}_f1"] = f1
        macro_f1_vals.append(f1)

    metrics["macro_f1"] = sum(macro_f1_vals) / len(macro_f1_vals) if macro_f1_vals else 0.0
    return metrics


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def style_output_path(base_output: Path, prompt_style: str) -> Path:
    if base_output.suffix:
        return base_output.with_name(f"{base_output.stem}.{prompt_style}{base_output.suffix}")
    return base_output.with_name(f"{base_output.name}.{prompt_style}.jsonl")


def save_results_and_metrics(
    results: List[Dict[str, Any]],
    metrics: Dict[str, float],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    metrics_path = out_path.with_suffix(".metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\nResults  → {out_path}")
    print(f"Metrics  → {metrics_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Run OPIN+INSTR and KRE prompt baselines")
    p.add_argument(
        "--model",
        required=True,
        help="Registry key (e.g. qwen3-4b) or HF hub ID",
    )
    p.add_argument(
        "--model_path",
        default=None,
        help="Optional local path; falls back to HF hub if corrupt",
    )
    p.add_argument(
        "--data",
        required=True,
        help="Path to instances JSONL (test or val split)",
    )
    p.add_argument(
        "--output",
        default="results.jsonl",
        help="Base output path; style names will be appended automatically",
    )
    p.add_argument(
        "--no_chat_template",
        action="store_true",
        help="Skip apply_chat_template even if available",
    )
    p.add_argument(
        "--max_ctx_tokens",
        type=int,
        default=3500,
        help="Max tokens for the context block",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N instances (debugging)",
    )
    p.add_argument(
        "--prompt_styles",
        nargs="+",
        default=["opin_instr", "kre"],
        choices=["opin_instr", "kre"],
        help="Which baselines to run",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    model_id = get_model_id(args.model)

    print(f"Model ID      : {model_id}")
    print(f"Data          : {args.data}")
    print(f"Prompt styles : {args.prompt_styles}")

    tokenizer = load_tokenizer(model_id, local_model_path=args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(model_id, local_model_path=args.model_path)

    instances = load_instances(args.data)
    if args.limit:
        instances = instances[:args.limit]
    print(f"Instances     : {len(instances)}")

    base_output = Path(args.output)

    for prompt_style in args.prompt_styles:
        print(f"\n===== Running baseline: {prompt_style} =====")
        results = evaluate(instances, model, tokenizer, prompt_style, args)
        metrics = compute_metrics(results)

        print("\n=== Metrics ===")
        for k, v in sorted(metrics.items()):
            print(f"  {k:30s}: {v:.4f}")

        out_path = style_output_path(base_output, prompt_style)
        save_results_and_metrics(results, metrics, out_path)


if __name__ == "__main__":
    main()
