#!/usr/bin/env python3
"""Summarize lm-eval JSON outputs into a small markdown table.

This repo runs lm-eval on a remote Azure VM. The resulting JSON files can be
copied back into `artifacts/` and summarized locally with this script.

Usage:
  python tools/summarize_lmeval_results.py artifacts/**/lmeval_*.json

Notes:
- lm-eval output schema can vary by version; this script is defensive.
- It extracts the primary metric per task when possible.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class Row:
    file: str
    task: str
    metric: str
    value: float | None
    stderr: float | None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _pick_primary_metric(task_metrics: dict[str, Any]) -> tuple[str, float | None, float | None]:
    """Pick a reasonable primary metric from a task metrics dict."""

    # Common keys in lm-eval results:
    # - exact_match
    # - acc
    # - pass@1
    # - exact_match,stderr
    # - acc,stderr
    # - exact_match_stderr
    # - acc_stderr
    preferred = [
        "exact_match",
        "acc",
        "accuracy",
        "pass@1",
        "pass_at_1",
    ]

    for k in preferred:
        if k in task_metrics:
            val = _as_float(task_metrics.get(k))
            stderr = _as_float(task_metrics.get(f"{k}_stderr") or task_metrics.get(f"{k},stderr"))
            return k, val, stderr

    # Fallback: first float-ish metric.
    for k, v in task_metrics.items():
        if k.endswith("_stderr") or ",stderr" in k:
            continue
        val = _as_float(v)
        if val is not None:
            stderr = _as_float(task_metrics.get(f"{k}_stderr") or task_metrics.get(f"{k},stderr"))
            return k, val, stderr

    return "(unknown)", None, None


def _iter_rows(path: str, data: dict[str, Any]) -> Iterable[Row]:
    results = data.get("results")
    if isinstance(results, dict):
        for task, metrics in results.items():
            if not isinstance(metrics, dict):
                continue
            metric, value, stderr = _pick_primary_metric(metrics)
            yield Row(file=os.path.basename(path), task=task, metric=metric, value=value, stderr=stderr)
        return

    # Some versions might store per-task under a different key.
    return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="lm-eval JSON result files")
    args = ap.parse_args()

    all_rows: list[Row] = []
    for path in args.paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_rows.extend(list(_iter_rows(path, data)))

    all_rows.sort(key=lambda r: (r.file, r.task))

    print("| file | task | metric | value | stderr |")
    print("|---|---|---|---:|---:|")
    for r in all_rows:
        v = "" if r.value is None else f"{r.value:.6f}"
        s = "" if r.stderr is None else f"{r.stderr:.6f}"
        print(f"| {r.file} | {r.task} | {r.metric} | {v} | {s} |")


if __name__ == "__main__":
    main()
