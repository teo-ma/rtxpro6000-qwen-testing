#!/usr/bin/env python3

import argparse
import concurrent.futures
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests


@dataclass(frozen=True)
class EvalItem:
    id: str
    prompt: str
    expected: str
    type: str  # "exact" | "number" | "regex" | "contains"


def _normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\t\n\r]", " ", s)
    s = re.sub(r"[^0-9a-z\u4e00-\u9fff\-\. ]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_first_number(s: str) -> Optional[float]:
    # Handles ints/floats, optional sign, commas.
    m = re.search(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def _extract_last_number(s: str) -> Optional[float]:
    matches = re.findall(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?", s)
    if not matches:
        return None
    for token in reversed(matches):
        try:
            return float(token.replace(",", ""))
        except ValueError:
            continue
    return None


def _last_nonempty_line(s: str) -> str:
    lines = [ln.strip() for ln in s.splitlines()]
    for ln in reversed(lines):
        if ln:
            return ln
    return s.strip()


def _score(item: EvalItem, output_text: str) -> Tuple[bool, str]:
    if item.type == "number":
        # Many instruction-tuned models may restate the question numbers;
        # use the last number as the best proxy for the final answer.
        got = _extract_last_number(output_text)
        exp = _extract_first_number(item.expected)
        if got is None or exp is None:
            return False, "number_parse_failed"
        ok = abs(got - exp) < 1e-9
        return ok, "number_exact" if ok else f"number_mismatch(got={got},exp={exp})"

    if item.type == "regex":
        ok = re.search(item.expected, output_text, flags=re.IGNORECASE | re.MULTILINE) is not None
        return ok, "regex" if ok else "regex_mismatch"

    if item.type == "contains":
        ok = _normalize_text(item.expected) in _normalize_text(output_text)
        return ok, "contains" if ok else "contains_mismatch"

    # default exact
    candidate = _last_nonempty_line(output_text)
    ok = _normalize_text(candidate) == _normalize_text(item.expected)
    return ok, "exact" if ok else "exact_mismatch"


def _chat_completion(url: str, model: str, prompt: str, max_tokens: int, temperature: float, timeout_s: float) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict evaluator. Reply with ONLY the final answer. Do not output <think> blocks or reasoning.",
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    with requests.Session() as session:
        r = session.post(url, json=payload, timeout=timeout_s)
        r.raise_for_status()
        return r.json()


def _extract_text(resp: Dict[str, Any]) -> str:
    try:
        return resp["choices"][0]["message"]["content"]
    except Exception:
        return ""


def load_jsonl(path: str) -> List[EvalItem]:
    items: List[EvalItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append(
                EvalItem(
                    id=str(obj["id"]),
                    prompt=str(obj["prompt"]),
                    expected=str(obj["expected"]),
                    type=str(obj.get("type", "exact")),
                )
            )
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate accuracy via an OpenAI-compatible chat completion endpoint.")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--endpoint", default="/v1/chat/completions")
    ap.add_argument("--model", required=True)
    ap.add_argument("--evalset", required=True, help="Path to JSONL evalset")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=1200)
    ap.add_argument("--out", required=True, help="Output JSON file")
    args = ap.parse_args()

    url = args.base_url.rstrip("/") + args.endpoint
    items = load_jsonl(args.evalset)

    # Warmup
    _ = _chat_completion(url, args.model, items[0].prompt, min(args.max_tokens, 16), args.temperature, args.timeout)

    per_item: List[Dict[str, Any]] = []
    t0 = time.perf_counter()

    def run_one(item: EvalItem) -> Dict[str, Any]:
        t_req0 = time.perf_counter()
        resp = _chat_completion(url, args.model, item.prompt, args.max_tokens, args.temperature, args.timeout)
        t_req1 = time.perf_counter()
        text = _extract_text(resp)
        ok, reason = _score(item, text)
        usage = resp.get("usage") or {}
        return {
            "id": item.id,
            "type": item.type,
            "ok": bool(ok),
            "reason": reason,
            "latency_s": t_req1 - t_req0,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "expected": item.expected,
            "output": text,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(run_one, it) for it in items]
        for fut in concurrent.futures.as_completed(futs):
            per_item.append(fut.result())

    t1 = time.perf_counter()

    ok_n = sum(1 for r in per_item if r.get("ok"))
    total_n = len(per_item)
    acc = ok_n / total_n if total_n else 0.0

    prompt_tokens = [r.get("prompt_tokens") for r in per_item if isinstance(r.get("prompt_tokens"), int)]
    completion_tokens = [r.get("completion_tokens") for r in per_item if isinstance(r.get("completion_tokens"), int)]

    out_obj = {
        "endpoint": url,
        "model": args.model,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "count": total_n,
        "ok": ok_n,
        "accuracy": acc,
        "total_time_s": t1 - t0,
        "latency_s": {
            "mean": sum(r["latency_s"] for r in per_item) / total_n if total_n else None,
        },
        "token_usage": {
            "prompt_tokens_mean": (sum(prompt_tokens) / len(prompt_tokens)) if prompt_tokens else None,
            "completion_tokens_mean": (sum(completion_tokens) / len(completion_tokens)) if completion_tokens else None,
        },
        "items": sorted(per_item, key=lambda x: x["id"]),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print(json.dumps({k: out_obj[k] for k in out_obj if k != "items"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
