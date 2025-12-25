#!/usr/bin/env python3
"""Compute a more reasonable AIME24 score from lm-eval samples JSONL.

Why:
- lm-eval's default exact_match is a strict string match.
- For AIME-style tasks the intended final answer is an integer (typically 0-999).
- Models may output LaTeX, fractions, or extra text; strict exact_match can understate accuracy.

This script computes:
- integer_match: extract a single integer prediction and compare to gold integer.

Input:
- samples_aime24_*.jsonl produced by lm-eval with --log_samples

Output:
- prints n, integer_match, and optionally per-sample debug lines.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from typing import Optional, Tuple


_INT_RE = re.compile(r"[-+]?\d+")
_FRAC_RE = re.compile(r"\\frac\{\s*([-+]?\d+)\s*\}\{\s*([-+]?\d+)\s*\}")
_INLINE_FRAC_RE = re.compile(r"\b([-+]?\d+)\s*/\s*([-+]?\d+)\b")

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


@dataclass
class Parsed:
    value: Optional[int]
    reason: str


def _clean_text(text: str) -> str:
    # Remove common LaTeX wrappers.
    t = text.strip()
    t = t.replace("\\boxed", "")
    # Strip $$...$$ and $...$ markers (keep content).
    t = t.replace("$$", " ").replace("$", " ")
    return t


def _split_think(text: str) -> Tuple[str, str]:
    """Return (think_part, after_think_part). Either part may be empty."""
    if text is None:
        return "", ""
    s = str(text)
    lower = s.lower()
    start = lower.find("<think>")
    if start < 0:
        return "", s
    end = lower.find("</think>", start)
    if end < 0:
        return s, ""
    end_close = end + len("</think>")
    return s[start:end_close], s[end_close:]


def strip_think_blocks(text: str) -> str:
    if text is None:
        return ""
    return _THINK_BLOCK_RE.sub(" ", str(text))


def parse_aime_integer(output: str) -> Parsed:
    """Best-effort extract an intended integer answer from model output."""
    if output is None:
        return Parsed(None, "no_output")

    t = _clean_text(str(output))

    # Prefer \frac{a}{b} if it yields an integer.
    m = _FRAC_RE.search(t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if b != 0 and a % b == 0:
            return Parsed(a // b, "latex_frac_int")
        return Parsed(None, "latex_frac_nonint")

    # Prefer inline a/b if it yields an integer.
    m = _INLINE_FRAC_RE.search(t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if b != 0 and a % b == 0:
            return Parsed(a // b, "inline_frac_int")
        return Parsed(None, "inline_frac_nonint")

    # Otherwise, pick the last integer token in the text (common pattern: "Answer: 123").
    ints = _INT_RE.findall(t)
    if not ints:
        return Parsed(None, "no_int_found")

    try:
        return Parsed(int(ints[-1]), "last_int")
    except Exception:
        return Parsed(None, "int_parse_error")


def parse_gold_integer(target: str) -> Optional[int]:
    if target is None:
        return None
    s = str(target).strip()
    m = _INT_RE.fullmatch(s)
    if not m:
        # Some harness versions store target with whitespace.
        m2 = _INT_RE.search(s)
        if not m2:
            return None
        return int(m2.group(0))
    return int(m.group(0))


def extract_resp(sample_obj) -> Optional[str]:
    resps = sample_obj.get("resps")
    if isinstance(resps, list) and resps and isinstance(resps[0], list) and resps[0]:
        return resps[0][0]
    return None


def parse_aime_integer_prefer_final(output: str) -> Parsed:
    """Prefer parsing from the "final" region after </think>, then fallback."""
    think_part, after_think = _split_think(output)
    after_think = after_think.strip()
    if after_think:
        parsed_after = parse_aime_integer(after_think)
        if parsed_after.value is not None:
            return Parsed(parsed_after.value, f"after_think:{parsed_after.reason}")

    # If no usable content after </think>, strip think blocks and parse remaining.
    stripped = strip_think_blocks(output)
    parsed_stripped = parse_aime_integer(stripped)
    if parsed_stripped.value is not None:
        return Parsed(parsed_stripped.value, f"no_think:{parsed_stripped.reason}")

    # As a last resort, parse the raw output.
    parsed_raw = parse_aime_integer(output)
    return Parsed(parsed_raw.value, f"raw:{parsed_raw.reason}")


def score_file(path: str, debug: bool = False, prefer_final: bool = True) -> Tuple[int, int]:
    total = 0
    correct = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            total += 1

            gold = parse_gold_integer(obj.get("target"))
            pred_raw = extract_resp(obj)
            parsed = parse_aime_integer_prefer_final(pred_raw) if prefer_final else parse_aime_integer(pred_raw)

            ok = (gold is not None) and (parsed.value is not None) and (parsed.value == gold)
            if ok:
                correct += 1

            if debug:
                doc_id = obj.get("doc_id")
                print(
                    f"doc_id={doc_id} gold={gold} pred={parsed.value} ok={ok} reason={parsed.reason} raw={repr(pred_raw)[:200]}"
                )

    return total, correct


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("samples_jsonl", help="Path to samples_aime24_*.jsonl")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument(
        "--raw",
        action="store_true",
        help="Use raw parsing (no <think> stripping / no preference for after </think>).",
    )
    args = ap.parse_args()

    n, c = score_file(args.samples_jsonl, debug=args.debug, prefer_final=not args.raw)
    acc = (c / n) if n else 0.0
    mode = "prefer_final" if not args.raw else "raw"
    print(f"n={n} integer_match[{mode}]={acc:.4f} ({c}/{n})")


if __name__ == "__main__":
    main()
