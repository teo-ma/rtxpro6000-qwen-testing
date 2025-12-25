"""No-calibration NVFP4 quantization for Qwen/Qwen3-14B via LLM Compressor.

This script mirrors the calibrated workflow documented in qwen3_14b_nvfp4_quant_and_eval.md,
but supports running with NUM_CALIBRATION_SAMPLES=0 (no dataset calibration).

Typical usage on the VM:
  NUM_CALIBRATION_SAMPLES=0 \
    OUT_DIR=/data/models/Qwen3-14B-NVFP4-NO-CALIB \
  python -u qwen3_14b_nvfp4_nocalib.py

Notes:
- NVFP4 benefits from calibration data to set global activation scales.
  With NUM_CALIBRATION_SAMPLES=0, accuracy may degrade.
- If you *do* want calibration, set NUM_CALIBRATION_SAMPLES>0 and provide DATASET_ID/DATASET_SPLIT.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _messages_to_text(messages: list[dict[str, Any]]) -> str:
    # UltraChat style: [{'role': 'user'|'assistant'|..., 'content': '...'}, ...]
    lines: list[str] = []
    for m in messages:
        role = str(m.get("role", ""))
        content = str(m.get("content", ""))
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _build_dataset(num_calibration_samples: int):
    """Return a datasets.Dataset-like object for oneshot().

    For no-calib mode (num_calibration_samples==0), we intentionally avoid
    downloading any dataset and provide a tiny in-memory dataset that should not
    be iterated.
    """

    if num_calibration_samples == 0:
        try:
            from datasets import Dataset
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "NUM_CALIBRATION_SAMPLES=0 still requires 'datasets' installed "
                "for the in-memory placeholder Dataset. Install with: pip install datasets"
            ) from exc
        return Dataset.from_dict({"text": [""]})

    dataset_id = os.environ.get("DATASET_ID")
    dataset_split = os.environ.get("DATASET_SPLIT", "train")

    if not dataset_id:
        raise RuntimeError(
            "NUM_CALIBRATION_SAMPLES>0 requires DATASET_ID (and optionally DATASET_SPLIT)."
        )

    from datasets import load_dataset

    ds = load_dataset(dataset_id, split=dataset_split)

    # Normalize to a 'text' column.
    if "text" in ds.column_names:
        return ds

    if "messages" in ds.column_names:
        ds = ds.map(lambda row: {"text": _messages_to_text(row["messages"])})
        return ds

    raise RuntimeError(
        f"Dataset {dataset_id!r} split {dataset_split!r} has no 'text' or 'messages' column: {ds.column_names}"
    )


def main() -> None:
    model_id = os.environ.get("MODEL_ID", "Qwen/Qwen3-14B")
    out_dir = os.environ.get("OUT_DIR", "/data/models/Qwen3-14B-NVFP4-NO-CALIB")
    max_seq_length = _env_int("MAX_SEQUENCE_LENGTH", 2048)
    num_calibration_samples = _env_int("NUM_CALIBRATION_SAMPLES", 0)

    enable_spinquant = _env_bool("ENABLE_SPINQUANT", False)
    pipeline = os.environ.get("PIPELINE")  # optional, e.g. 'datafree'

    print(
        "[qwen3_14b_nvfp4_nocalib] config:\n"
        f"  MODEL_ID={model_id}\n"
        f"  OUT_DIR={out_dir}\n"
        f"  MAX_SEQUENCE_LENGTH={max_seq_length}\n"
        f"  NUM_CALIBRATION_SAMPLES={num_calibration_samples}\n"
        f"  ENABLE_SPINQUANT={int(enable_spinquant)}\n"
        f"  PIPELINE={pipeline!r}\n"
    )

    print("[qwen3_14b_nvfp4_nocalib] loading model/tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print("[qwen3_14b_nvfp4_nocalib] building dataset...")
    ds = _build_dataset(num_calibration_samples)

    print("[qwen3_14b_nvfp4_nocalib] building recipe...")
    recipe: Any
    quant = QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=["lm_head"])

    if enable_spinquant:
        from llmcompressor.modifiers.transform import SpinQuantModifier

        recipe = [
            SpinQuantModifier(
                rotations=["R1", "R2", "R4"],
                transform_block_size=128,
                transform_type="hadamard",
            ),
            quant,
        ]
    else:
        recipe = quant

    print("[qwen3_14b_nvfp4_nocalib] running oneshot...")
    oneshot_kwargs: dict[str, Any] = dict(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=max_seq_length,
        num_calibration_samples=num_calibration_samples,
    )
    if pipeline:
        oneshot_kwargs["pipeline"] = pipeline

    oneshot(**oneshot_kwargs)

    print(f"[qwen3_14b_nvfp4_nocalib] saving compressed model to: {out_dir}")
    model.save_pretrained(out_dir, save_compressed=True)
    tokenizer.save_pretrained(out_dir)

    print("[qwen3_14b_nvfp4_nocalib] done.")


if __name__ == "__main__":
    # Ensure torch is initialized (some environments delay CUDA init)
    _ = torch.__version__
    main()
