#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import os
import statistics
import time
from typing import Any, Dict, List, Optional, Tuple

import requests


def _parse_sse_lines(lines) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
    """Parse OpenAI-style SSE stream.

    Returns (ttft_s, last_json_chunk).
    """
    ttft_s: Optional[float] = None
    last_chunk: Optional[Dict[str, Any]] = None
    t0 = time.perf_counter()
    for raw_line in lines:
        if raw_line is None:
            continue
        line = raw_line.decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            break
        if ttft_s is None:
            ttft_s = time.perf_counter() - t0
        try:
            last_chunk = json.loads(payload)
        except json.JSONDecodeError:
            # Some servers may emit partial / non-JSON lines.
            continue
    return ttft_s, last_chunk


def _one_request(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
    stream: bool,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": bool(stream),
    }

    # requests.Session is not thread-safe; create a fresh session per request.
    with requests.Session() as session:
        t0 = time.perf_counter()
        r = session.post(url, json=payload, timeout=timeout_s, stream=bool(stream))
        r.raise_for_status()

        if not stream:
            data = r.json()
            t1 = time.perf_counter()
            usage = data.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            return {
                "latency_s": t1 - t0,
                "ttft_s": None,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "raw": data,
            }

        # Streaming response
        ttft_s, last_chunk = _parse_sse_lines(r.iter_lines())
        t1 = time.perf_counter()
        usage = (last_chunk or {}).get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        return {
            "latency_s": t1 - t0,
            "ttft_s": ttft_s,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "raw": last_chunk,
        }


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark an OpenAI-compatible chat completion endpoint.")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--endpoint", default="/v1/chat/completions")
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--requests", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming responses and measure TTFT (time-to-first-token).",
    )
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read()

    url = args.base_url.rstrip("/") + args.endpoint

    # Warmup single request
    _one_request(
        url,
        args.model,
        prompt,
        min(args.max_tokens, 32),
        args.temperature,
        args.timeout,
        args.stream,
    )

    results: List[Dict[str, Any]] = []
    t_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [
            ex.submit(
                _one_request,
                url,
                args.model,
                prompt,
                args.max_tokens,
                args.temperature,
                args.timeout,
                args.stream,
            )
            for _ in range(args.requests)
        ]
        for fut in concurrent.futures.as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:
                results.append({"error": str(e)})

    t_end = time.perf_counter()

    ok = [r for r in results if "latency_s" in r]
    errs = [r for r in results if "error" in r]

    latencies = [r["latency_s"] for r in ok]
    ttfts = [r["ttft_s"] for r in ok if isinstance(r.get("ttft_s"), (int, float))]
    total_time = t_end - t_start
    qps = (len(ok) / total_time) if total_time > 0 else 0.0

    # Token stats are best-effort (depends on server returning usage)
    prompt_tokens = [r["prompt_tokens"] for r in ok if isinstance(r.get("prompt_tokens"), int)]
    completion_tokens = [r["completion_tokens"] for r in ok if isinstance(r.get("completion_tokens"), int)]

    prompt_tokens_mean: Optional[float] = (sum(prompt_tokens) / len(prompt_tokens)) if prompt_tokens else None
    completion_tokens_mean: Optional[float] = (
        (sum(completion_tokens) / len(completion_tokens)) if completion_tokens else None
    )

    prompt_tps: Optional[float] = (prompt_tokens_mean * qps) if prompt_tokens_mean is not None else None
    decode_tps: Optional[float] = (
        (completion_tokens_mean * qps) if completion_tokens_mean is not None else None
    )
    ms_per_output_token: Optional[float] = None
    # Rough estimate: average wall latency per request divided by avg output tokens.
    latency_mean_s: Optional[float] = (sum(latencies) / len(latencies)) if latencies else None
    if completion_tokens_mean and latency_mean_s:
        ms_per_output_token = (latency_mean_s * 1000.0) / completion_tokens_mean

    out_obj: Dict[str, Any] = {
        "endpoint": url,
        "model": args.model,
        "concurrency": args.concurrency,
        "requests": args.requests,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": bool(args.stream),
        "ok": len(ok),
        "errors": len(errs),
        "total_time_s": total_time,
        "qps": qps,
        "latency_s": {
            "p50": statistics.median(latencies) if latencies else None,
            "p95": statistics.quantiles(latencies, n=20)[-1] if len(latencies) >= 20 else None,
            "mean": latency_mean_s,
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "ttft_s": {
            "p50": statistics.median(ttfts) if ttfts else None,
            "p95": statistics.quantiles(ttfts, n=20)[-1] if len(ttfts) >= 20 else None,
            "mean": (sum(ttfts) / len(ttfts)) if ttfts else None,
            "min": min(ttfts) if ttfts else None,
            "max": max(ttfts) if ttfts else None,
        },
        "token_usage": {
            "prompt_tokens_mean": prompt_tokens_mean,
            "completion_tokens_mean": completion_tokens_mean,
        },
        "derived": {
            "prompt_tps": prompt_tps,
            "decode_tps": decode_tps,
            "ms_per_output_token": ms_per_output_token,
        },
    }

    print(json.dumps(out_obj, ensure_ascii=False, indent=2))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
