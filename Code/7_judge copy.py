#!/usr/bin/env python3
import argparse
import asyncio
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Official Gemini SDK
from google import genai
from google.genai import types

MODES = ["answer", "refuse", "conflict"]

def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

def first_nonempty_line(text: str) -> str:
    for line in (text or "").splitlines():
        s = line.strip()
        if s:
            return s
    return ""

def safe_clip(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars] + "\n[TRUNCATED]"

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Robust-ish extraction if the model wraps JSON in markdown fences.
    """
    if not text:
        return None
    t = text.strip()
    # Remove ```json fences if present
    t = re.sub(r"^```json\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^```\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    # Find first {...}
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def macro_f1(y_true: List[str], y_pred: List[str]) -> float:
    f1s = []
    for lab in MODES:
        tp = sum((t == lab and p == lab) for t, p in zip(y_true, y_pred))
        fp = sum((t != lab and p == lab) for t, p in zip(y_true, y_pred))
        fn = sum((t == lab and p != lab) for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        f1s.append(f1)
    return float(sum(f1s) / len(f1s))

def accuracy(y_true: List[str], y_pred: List[str]) -> float:
    return float(sum(t == p for t, p in zip(y_true, y_pred)) / max(1, len(y_true)))

def far_refuse_conflict(y_true: List[str], y_pred: List[str]) -> float:
    denom = sum(t in ("refuse", "conflict") for t in y_true)
    num = sum((t in ("refuse", "conflict") and p == "answer") for t, p in zip(y_true, y_pred))
    return float(num / max(1, denom))

def answer_substring(rows: List[Dict[str, Any]], cond: str) -> float:
    vals = []
    for r in rows:
        if r.get("true_mode") != "answer":
            continue
        gold = (r.get("gold_answer") or "").strip().lower()
        pred = (r[cond]["text"] or "").strip().lower()
        if not gold:
            continue
        vals.append(1.0 if gold in pred else 0.0)
    return float(sum(vals) / max(1, len(vals)))

JUDGE_INSTRUCTIONS = """You are grading the assistant's OUTPUT STYLE (mode), not factual correctness.
Classify the assistant output into EXACTLY one of:
- "answer": it attempts to answer the question (even if wrong), giving a concrete answer.
- "refuse": it refuses / says insufficient info / cannot determine from provided docs.
- "conflict": it explicitly says sources/documents conflict/contradict/inconsistent and does NOT commit to a single answer.

Important:
- Do NOT label "conflict" just because the word "conflict" appears in a normal sentence (e.g., "World War II conflict").
- Only "conflict" if the output is meta-level about documents/sources disagreeing.
- If the first line is exactly "Not enough information." => refuse.
- If the first line is exactly "Documents contain conflicting information." => conflict.

Return STRICT JSON only:
{"mode": "<answer|refuse|conflict>", "confidence": <0.0-1.0>, "short_reason": "<=20 words>"}"""

def make_prompt(gen_text: str) -> str:
    # We pass first line + some context; you can increase if needed
    line1 = first_nonempty_line(gen_text)
    clipped = safe_clip(gen_text, 1400)
    return (
        JUDGE_INSTRUCTIONS
        + "\n\n=== OUTPUT (to classify) ===\n"
        + f"FIRST_LINE: {line1}\n\nFULL_OUTPUT:\n{clipped}\n"
    )

async def judge_one(
    client: genai.Client,
    model_name: str,
    gen_text: str,
    semaphore: asyncio.Semaphore,
    temperature: float = 0.0,
    max_output_tokens: int = 120,
) -> Dict[str, Any]:
    """
    Async wrapper using to_thread around the sync SDK call.
    """
    prompt = make_prompt(gen_text)

    async with semaphore:
        def _call():
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                ),
            )
            return resp.text

        text = await asyncio.to_thread(_call)

    parsed = extract_json(text)
    if not parsed:
        # fallback: very strict second try
        retry_prompt = prompt + "\n\nREMINDER: Output STRICT JSON only. No markdown. No extra text."
        async with semaphore:
            def _call2():
                resp2 = client.models.generate_content(
                    model=model_name,
                    contents=retry_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=max_output_tokens,
                    ),
                )
                return resp2.text
            text2 = await asyncio.to_thread(_call2)
        parsed = extract_json(text2)

    if not parsed:
        # last resort fallback
        return {"mode": "answer", "confidence": 0.0, "short_reason": "parse_failed"}

    mode = str(parsed.get("mode", "")).strip().lower()
    if mode not in MODES:
        mode = "answer"
    conf = float(parsed.get("confidence", 0.0) or 0.0)
    conf = max(0.0, min(1.0, conf))
    reason = str(parsed.get("short_reason", ""))[:200]
    return {"mode": mode, "confidence": conf, "short_reason": reason}

def load_ablation(path: Path) -> Dict[str, Any]:
    obj = json.load(open(path, "r", encoding="utf-8"))
    if "rows" not in obj:
        raise ValueError(f"{path} missing 'rows' field (expected ablation output).")
    return obj

async def main_async(args):
    api_key = ""
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY in your environment first.")

    client = genai.Client(api_key=api_key)  # SDK supports env var too :contentReference[oaicite:1]{index=1}

    in_paths = []
    for pat in args.inputs:
        in_paths += [Path(p) for p in sorted(Path().glob(pat))] if any(ch in pat for ch in "*?[]") else [Path(pat)]
    in_paths = [p for p in in_paths if p.exists()]
    if not in_paths:
        raise SystemExit("No input files found. Pass e.g. results/ablation_val_a0.json or a glob.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(args.concurrency)

    for in_path in in_paths:
        obj = load_ablation(in_path)
        rows = obj["rows"]
        if args.max_rows:
            rows = rows[:args.max_rows]

        # Build tasks with caching by generation hash (saves tons of calls)
        cache_path = out_dir / f"{in_path.stem}_judge_cache.json"
        cache: Dict[str, Dict[str, Any]] = {}
        if cache_path.exists() and args.resume_cache:
            cache = json.load(open(cache_path, "r", encoding="utf-8"))

        results_jsonl = out_dir / f"{in_path.stem}_judge.jsonl"
        fout = open(results_jsonl, "a" if args.append else "w", encoding="utf-8")

        # Determine which conditions to judge
        conds = args.conditions.split(",") if args.conditions else ["S1", "S2", "S3", "S4"]

        # For metric aggregation
        per_cond_true: Dict[str, List[str]] = {c: [] for c in conds}
        per_cond_pred: Dict[str, List[str]] = {c: [] for c in conds}

        pending: List[Tuple[int, str, str]] = []  # (row_idx, cond, hash)
        texts_by_key: Dict[str, str] = {}

        for i, r in enumerate(rows):
            true_mode = r.get("true_mode")
            for c in conds:
                text = r[c]["text"]
                h = sha(text)
                key = f"{c}:{h}"
                texts_by_key[key] = text
                if key in cache:
                    pass
                else:
                    pending.append((i, c, key))

        print(f"\nJudging file: {in_path}")
        print(f"Rows={len(rows)} conditions={conds} pending_calls={len(pending)} (concurrency={args.concurrency})")

        async def run_batch(batch_keys: List[str]):
            tasks = [judge_one(client, args.judge_model, texts_by_key[k], sem) for k in batch_keys]
            outs = await asyncio.gather(*tasks)
            return list(zip(batch_keys, outs))

        # Run in chunks (prevents huge memory)
        chunk = args.chunk_size
        for start in range(0, len(pending), chunk):
            sub = pending[start:start+chunk]
            keys = [k for (_i, _c, k) in sub]
            kvs = await run_batch(keys)
            for k, out in kvs:
                cache[k] = out
            if args.save_cache_every and (start + chunk) % args.save_cache_every == 0:
                json.dump(cache, open(cache_path, "w", encoding="utf-8"), indent=2)

        # Save cache at end
        json.dump(cache, open(cache_path, "w", encoding="utf-8"), indent=2)

        # Write judged outputs + compute metrics
        for r in rows:
            true_mode = r.get("true_mode")
            rid = r.get("id")
            rec = {"id": rid, "true_mode": true_mode, "source_file": str(in_path)}
            for c in conds:
                text = r[c]["text"]
                key = f"{c}:{sha(text)}"
                j = cache.get(key, {"mode": "answer", "confidence": 0.0, "short_reason": "missing"})
                rec[c] = {
                    "judge_mode": j["mode"],
                    "judge_conf": j["confidence"],
                    "judge_reason": j["short_reason"],
                }
                per_cond_true[c].append(true_mode)
                per_cond_pred[c].append(j["mode"])
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

        fout.close()

        summary = {}
        for c in conds:
            y_t = per_cond_true[c]
            y_p = per_cond_pred[c]
            summary[c] = {
                "acc": accuracy(y_t, y_p),
                "macro_f1": macro_f1(y_t, y_p),
                "FAR_refuse+conflict": far_refuse_conflict(y_t, y_p),
                "answer_substring": answer_substring(rows, c),
            }

        summary_path = out_dir / f"{in_path.stem}_judge_summary.json"
        json.dump(summary, open(summary_path, "w", encoding="utf-8"), indent=2)

        print("Saved judge JSONL   :", results_jsonl)
        print("Saved judge summary :", summary_path)
        print("Summary:", json.dumps(summary, indent=2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="One or more ablation json files or globs, e.g. results/ablation_val_a*.json")
    ap.add_argument("--judge_model", default="gemini-2.5-flash",
                    help="Gemini model for judging (cheap is best).")
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--chunk_size", type=int, default=200)
    ap.add_argument("--max_rows", type=int, default=None)
    ap.add_argument("--conditions", default="S1,S2,S3,S4")
    ap.add_argument("--out_dir", default="results/judge")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--resume_cache", action="store_true")
    ap.add_argument("--save_cache_every", type=int, default=1000)
    args = ap.parse_args()

    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()