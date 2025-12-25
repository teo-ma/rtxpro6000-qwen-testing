#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/bench/qwen3_32b_vllm_stdbench_tp2_3hr_20251223
VENV=/data/bench/qwen25vl72b_vllm_bench_20251222/.venv

export HF_HOME=/data/bench/hf
export XDG_CACHE_HOME=/data/bench/hf
export TMPDIR=/data/bench/tmp
export CUDA_VISIBLE_DEVICES=0,1

mkdir -p "$ROOT/logs" "$ROOT/results"

# Limits tuned for ~3h wall-clock (best-effort).
LIMIT_MMLU=200
LIMIT_GPQA=100
LIMIT_MATH=50
LIMIT_HLE=50
LIMIT_LCB=20
LIMIT_SCICODE=20

GPU_MEM_UTIL=0.92
MAX_MODEL_LEN=4096

run_lmeval() {
  local tag="$1"; shift
  local model_id="$1"; shift
  local tasks="$1"; shift
  local limit="$1"; shift
  local gen_kwargs="$1"; shift

  local ts
  ts=$(date -u +%Y%m%dT%H%M%SZ)
  local log="$ROOT/logs/lmeval_${tag}_${tasks}_limit${limit}_${ts}.log"

  echo "[lmeval] tag=$tag tasks=$tasks limit=$limit model=$model_id"

  set -x
  "$VENV/bin/python" -m lm_eval \
    --model vllm \
    --model_args "pretrained=${model_id},tensor_parallel_size=2,gpu_memory_utilization=${GPU_MEM_UTIL},max_model_len=${MAX_MODEL_LEN}" \
    --tasks "$tasks" \
    --batch_size 1 \
    --apply_chat_template \
    ${gen_kwargs:+--gen_kwargs "$gen_kwargs"} \
    --limit "$limit" \
    --log_samples \
    --output_path "$ROOT/results/lmeval_${tag}_${tasks}.json" \
    2>&1 | tee "$log"
  set +x
}

run_lmeval_nolimit() {
  local tag="$1"; shift
  local model_id="$1"; shift
  local tasks="$1"; shift
  local gen_kwargs="$1"; shift

  local ts
  ts=$(date -u +%Y%m%dT%H%M%SZ)
  local log="$ROOT/logs/lmeval_${tag}_${tasks}_full_${ts}.log"

  echo "[lmeval] tag=$tag tasks=$tasks full model=$model_id"

  set -x
  "$VENV/bin/python" -m lm_eval \
    --model vllm \
    --model_args "pretrained=${model_id},tensor_parallel_size=2,gpu_memory_utilization=${GPU_MEM_UTIL},max_model_len=${MAX_MODEL_LEN}" \
    --tasks "$tasks" \
    --batch_size 1 \
    --apply_chat_template \
    ${gen_kwargs:+--gen_kwargs "$gen_kwargs"} \
    --log_samples \
    --output_path "$ROOT/results/lmeval_${tag}_${tasks}.json" \
    2>&1 | tee "$log"
  set +x
}

run_code_pass1() {
  local tag="$1"; shift
  local model_id="$1"; shift
  local dataset="$1"; shift
  local limit="$1"; shift

  local ts
  ts=$(date -u +%Y%m%dT%H%M%SZ)
  local out="$ROOT/results/pass1_${tag}_$(echo "$dataset" | tr "/" "_")_limit${limit}_${ts}.json"
  local log="$ROOT/logs/pass1_${tag}_$(echo "$dataset" | tr "/" "_")_limit${limit}_${ts}.log"

  echo "[pass1] tag=$tag dataset=$dataset limit=$limit model=$model_id"

  set -x
  "$VENV/bin/python" "$ROOT/tools/eval_codegen_pass1_vllm.py" \
    --dataset "$dataset" \
    --model "$model_id" \
    --max-model-len "$MAX_MODEL_LEN" \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-tokens 512 \
    --temperature 0.0 \
    --limit "$limit" \
    --timeout 20 \
    --out "$out" \
    2>&1 | tee "$log"
  set +x
}

MODEL_BF16="Qwen/Qwen3-32B"
MODEL_FP8="Qwen/Qwen3-32B-FP8"
MODEL_NVFP4="RedHatAI/Qwen3-32B-NVFP4"

# BF16
run_lmeval_nolimit "bf16" "$MODEL_BF16" "aime24" "max_gen_toks=1024"
run_lmeval         "bf16" "$MODEL_BF16" "hendrycks_math500" "$LIMIT_MATH" "max_gen_toks=512"
run_lmeval         "bf16" "$MODEL_BF16" "mmlu_pro" "$LIMIT_MMLU" ""
run_lmeval         "bf16" "$MODEL_BF16" "gpqa_diamond_zeroshot" "$LIMIT_GPQA" ""
run_lmeval         "bf16" "$MODEL_BF16" "hle" "$LIMIT_HLE" "" || true
run_code_pass1     "bf16" "$MODEL_BF16" "livecodebench/code_generation_lite" "$LIMIT_LCB"
run_code_pass1     "bf16" "$MODEL_BF16" "Zilinghan/scicode" "$LIMIT_SCICODE"

# FP8
run_lmeval_nolimit "fp8" "$MODEL_FP8" "aime24" "max_gen_toks=1024"
run_lmeval         "fp8" "$MODEL_FP8" "hendrycks_math500" "$LIMIT_MATH" "max_gen_toks=512"
run_lmeval         "fp8" "$MODEL_FP8" "mmlu_pro" "$LIMIT_MMLU" ""
run_lmeval         "fp8" "$MODEL_FP8" "gpqa_diamond_zeroshot" "$LIMIT_GPQA" ""
run_lmeval         "fp8" "$MODEL_FP8" "hle" "$LIMIT_HLE" "" || true
run_code_pass1     "fp8" "$MODEL_FP8" "livecodebench/code_generation_lite" "$LIMIT_LCB"
run_code_pass1     "fp8" "$MODEL_FP8" "Zilinghan/scicode" "$LIMIT_SCICODE"

# NVFP4
run_lmeval_nolimit "nvfp4" "$MODEL_NVFP4" "aime24" "max_gen_toks=1024"
run_lmeval         "nvfp4" "$MODEL_NVFP4" "hendrycks_math500" "$LIMIT_MATH" "max_gen_toks=512"
run_lmeval         "nvfp4" "$MODEL_NVFP4" "mmlu_pro" "$LIMIT_MMLU" ""
run_lmeval         "nvfp4" "$MODEL_NVFP4" "gpqa_diamond_zeroshot" "$LIMIT_GPQA" ""
run_lmeval         "nvfp4" "$MODEL_NVFP4" "hle" "$LIMIT_HLE" "" || true
run_code_pass1     "nvfp4" "$MODEL_NVFP4" "livecodebench/code_generation_lite" "$LIMIT_LCB"
run_code_pass1     "nvfp4" "$MODEL_NVFP4" "Zilinghan/scicode" "$LIMIT_SCICODE"

echo "DONE: results in $ROOT/results"
