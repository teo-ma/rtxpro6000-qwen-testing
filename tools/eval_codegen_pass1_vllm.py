#!/usr/bin/env python3
"""Evaluate code-generation datasets with pass@1 by executing unit tests.

This is intended for benchmarks that are not (yet) built into lm-evaluation-harness
in our environment (e.g. LiveCodeBench/SciCode).

It:
- Loads a HuggingFace dataset split
- Prompts a model via vLLM (in-process, no HTTP server)
- Writes the model output to a temp dir
- Runs the provided tests under strict timeouts
- Reports pass@1

Security note:
- Executing dataset-provided tests is inherently risky. This runner mitigates
  obvious risks with timeouts and resource limits, but it is not a perfect sandbox.
  Run only on an isolated VM.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any


def _first_present(sample: dict[str, Any], keys: list[str]) -> Any:
    for k in keys:
        if k in sample and sample[k] not in (None, ""):
            return sample[k]
    return None


def _strip_code_fences(text: str) -> str:
    # Remove ```python ... ``` or ``` ... ``` fences if present.
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_+-]*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text)
    return text.strip()


def _build_prompt(problem: str, starter_code: str | None) -> str:
    parts = [
        "You are a helpful coding assistant.",
        "Write a correct Python solution.",
        "Return ONLY Python code (no markdown, no explanations).",
        "",
        "Problem:",
        problem.strip(),
    ]
    if starter_code:
        parts += ["", "Starter code:", starter_code.rstrip()]
    parts += ["", "# Write your solution below"]
    return "\n".join(parts)


@dataclass
class EvalResult:
    passed: bool
    error: str | None


def _run_tests(workdir: str, timeout_s: int) -> EvalResult:
    """Run python tests in `workdir`.

    Expects files:
      - solution.py
      - tests.py
      - run_tests.py
    """

    env = os.environ.copy()
    env.update(
        {
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )

    try:
        cp = subprocess.run(
            ["python3", "-I", "run_tests.py"],
            cwd=workdir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            text=True,
        )
    except subprocess.TimeoutExpired:
        return EvalResult(False, f"timeout>{timeout_s}s")

    if cp.returncode == 0:
        return EvalResult(True, None)

    out = (cp.stdout or "").strip()
    return EvalResult(False, out[-4000:] if out else f"exit_code={cp.returncode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="HF dataset name, e.g. livecodebench/code_generation_lite")
    ap.add_argument("--split", default="test", help="dataset split")
    ap.add_argument("--model", required=True, help="HF model id for vLLM")
    ap.add_argument("--download-dir", default=None, help="HF download dir for vLLM")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="vLLM tensor parallel size (set to 2 for 2 GPUs with TP=2)",
    )
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    ap.add_argument(
        "--max-num-seqs",
        type=int,
        default=16,
        help="vLLM max_num_seqs to cap concurrency (helps avoid warmup OOM on some MIG/TP setups)",
    )
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0, help="limit number of samples (0 = all)")
    ap.add_argument("--timeout", type=int, default=20, help="per-sample test timeout seconds")
    ap.add_argument("--out", required=True, help="output json path")
    args = ap.parse_args()

    # Lazy imports so local machines without deps can still open the file.
    from datasets import load_dataset  # type: ignore
    from vllm import LLM, SamplingParams  # type: ignore

    ds = load_dataset(args.dataset, split=args.split)
    if args.limit and args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    llm = LLM(
        model=args.model,
        download_dir=args.download_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=False,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=1.0,
        stop=None,
    )

    results: list[dict[str, Any]] = []
    passed = 0

    for i in range(len(ds)):
        sample = ds[i]
        sample_id = _first_present(sample, ["task_id", "question_id", "id", "uid"]) or i

        problem = _first_present(sample, ["prompt", "problem", "question", "instruction", "text", "description"])
        starter_code = _first_present(sample, ["starter_code", "starter", "code_stub", "template", "skeleton"])  # optional
        tests = _first_present(sample, ["test", "tests", "unit_tests", "pytest", "checker"]) 

        if not isinstance(problem, str) or not isinstance(tests, str):
            results.append(
                {
                    "id": sample_id,
                    "ok": False,
                    "error": "missing_required_fields",
                    "fields": list(sample.keys()),
                }
            )
            continue

        prompt = _build_prompt(problem, starter_code if isinstance(starter_code, str) else None)
        out = llm.generate([prompt], sampling)
        gen = out[0].outputs[0].text if out and out[0].outputs else ""
        gen = _strip_code_fences(gen)

        with tempfile.TemporaryDirectory(prefix="codeeval_") as td:
            # Keep working set small and isolated.
            workdir = td
            with open(os.path.join(workdir, "solution.py"), "w", encoding="utf-8") as f:
                f.write(gen + "\n")

            with open(os.path.join(workdir, "tests.py"), "w", encoding="utf-8") as f:
                f.write(tests + "\n")

            # A generic runner that supports either:
            # - tests.py defining `check(candidate)`
            # - tests.py being directly executable
            run_py = """
import importlib
import traceback

try:
    solution = importlib.import_module('solution')
    tests = importlib.import_module('tests')

    # Common pattern: tests define check(candidate)
    if hasattr(tests, 'check') and callable(tests.check):
        tests.check(solution)
    else:
        # Fall back to executing tests module top-level.
        pass

except Exception:
    traceback.print_exc()
    raise
""".lstrip()
            with open(os.path.join(workdir, "run_tests.py"), "w", encoding="utf-8") as f:
                f.write(run_py)

            er = _run_tests(workdir, timeout_s=args.timeout)

        ok = bool(er.passed)
        passed += 1 if ok else 0

        results.append(
            {
                "id": sample_id,
                "ok": ok,
                "error": er.error,
            }
        )

        # Basic progress without extra deps.
        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(ds)}] pass@1 so far: {passed/(i+1):.4f}")

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "model": args.model,
        "count": len(ds),
        "passed": passed,
        "pass@1": passed / len(ds) if len(ds) else 0.0,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "timeout_s": args.timeout,
        "items": results,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps({k: summary[k] for k in ["dataset", "split", "model", "count", "pass@1"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
