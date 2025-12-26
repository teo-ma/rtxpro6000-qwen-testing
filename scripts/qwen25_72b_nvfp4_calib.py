"""Calibrated NVFP4 quantization for Qwen2.5-72B via LLM Compressor.

This follows the same workflow used in this repo for Qwen3-14B NVFP4 quantization,
but targets Qwen2.5-72B and is intended to run on the Azure VM(s) used in this repo.

Typical usage on the VM:
    mkdir -p /mnt/data/nvfp4_work/logs /mnt/data/models

  nohup env \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/mnt/data/hf_home \
    HUGGINGFACE_HUB_CACHE=/mnt/data/hf_home/hub \
    HF_DATASETS_CACHE=/mnt/data/hf_home/datasets \
    XDG_CACHE_HOME=/mnt/data/hf_home/xdg_cache \
    TMPDIR=/mnt/data/tmp_user \
    MODEL_ID=Qwen/Qwen2.5-72B-Instruct \
    DATASET_ID=HuggingFaceH4/ultrachat_200k \
    DATASET_SPLIT=train_sft \
    NUM_CALIBRATION_SAMPLES=1024 \
    MAX_SEQUENCE_LENGTH=2048 \
    ENABLE_SPINQUANT=1 \
    OUT_DIR=/mnt/data/models/Qwen2.5-72B-Instruct-NVFP4 \
    /path/to/python -u /mnt/data/nvfp4_work/qwen25_72b_nvfp4_calib.py \
    > /mnt/data/nvfp4_work/logs/qwen25_72b_nvfp4_$(date +%Y%m%d_%H%M%S).log 2>&1 &

Notes:
- Qwen2.5-72B is large; you likely need 2 GPUs or CPU offload.
- Set USE_DEVICE_MAP=1 (default) to let Accelerate shard the model across available devices.
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
    for message in messages:
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _build_dataset(num_calibration_samples: int):
    dataset_id = os.environ.get("DATASET_ID")
    dataset_split = os.environ.get("DATASET_SPLIT", "train")

    if not dataset_id:
        raise RuntimeError(
            "Calibration requires DATASET_ID (and optionally DATASET_SPLIT). "
            "Example: DATASET_ID=HuggingFaceH4/ultrachat_200k DATASET_SPLIT=train_sft"
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
    model_id = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-72B-Instruct")
    out_dir = os.environ.get("OUT_DIR", "/mnt/data/models/Qwen2.5-72B-Instruct-NVFP4")
    max_seq_length = _env_int("MAX_SEQUENCE_LENGTH", 2048)
    num_calibration_samples = _env_int("NUM_CALIBRATION_SAMPLES", 1024)

    enable_spinquant = _env_bool("ENABLE_SPINQUANT", True)
    pipeline = os.environ.get("PIPELINE")  # optional, e.g. 'datafree'

    trust_remote_code = _env_bool("TRUST_REMOTE_CODE", False)
    use_device_map = _env_bool("USE_DEVICE_MAP", True)
    max_memory_cuda_gib = _env_int("MAX_MEMORY_CUDA_GIB", 80)
    max_memory_cpu_gib = _env_int("MAX_MEMORY_CPU_GIB", 450)
    offload_dir = os.environ.get("OFFLOAD_DIR", "/mnt/data/tmp_user/hf_offload")

    print(
        "[qwen25_72b_nvfp4_calib] config:\n"
        f"  MODEL_ID={model_id}\n"
        f"  OUT_DIR={out_dir}\n"
        f"  MAX_SEQUENCE_LENGTH={max_seq_length}\n"
        f"  NUM_CALIBRATION_SAMPLES={num_calibration_samples}\n"
        f"  ENABLE_SPINQUANT={int(enable_spinquant)}\n"
        f"  PIPELINE={pipeline!r}\n"
        f"  TRUST_REMOTE_CODE={int(trust_remote_code)}\n"
        f"  USE_DEVICE_MAP={int(use_device_map)}\n"
        f"  MAX_MEMORY_CUDA_GIB={max_memory_cuda_gib}\n"
        f"  MAX_MEMORY_CPU_GIB={max_memory_cpu_gib}\n"
        f"  OFFLOAD_DIR={offload_dir}\n"
    )

    print("[qwen25_72b_nvfp4_calib] loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[qwen25_72b_nvfp4_calib] loading model...")
    model_kwargs: dict[str, Any] = dict(
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )

    # Qwen2.5-72B is too large for a single 96GB GPU in BF16.
    # Let Accelerate infer a placement across devices (or CPU offload) when enabled.
    if use_device_map:
        model_kwargs["device_map"] = "auto"
        model_kwargs["max_memory"] = {
            0: f"{max_memory_cuda_gib}GiB",
            "cpu": f"{max_memory_cpu_gib}GiB",
        }
        model_kwargs["offload_folder"] = offload_dir

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    print("[qwen25_72b_nvfp4_calib] building calibration dataset...")
    ds = _build_dataset(num_calibration_samples)

    print("[qwen25_72b_nvfp4_calib] building recipe...")
    quant = QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=["lm_head"])

    recipe: Any
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

    print("[qwen25_72b_nvfp4_calib] running oneshot...")
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

    print(f"[qwen25_72b_nvfp4_calib] saving compressed model to: {out_dir}")
    model.save_pretrained(out_dir, save_compressed=True)
    tokenizer.save_pretrained(out_dir)

    print("[qwen25_72b_nvfp4_calib] done.")


if __name__ == "__main__":
    _ = torch.__version__
    main()
