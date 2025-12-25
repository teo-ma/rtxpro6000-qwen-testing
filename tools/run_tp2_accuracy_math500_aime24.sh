#!/usr/bin/env bash
set -euo pipefail

# Run small-scale accuracy eval (TP=2) for Qwen3-32B across NVFP4/FP8/BF16.
# Uses lm-eval with vLLM backend (in-process), matching repo's stdbench approach.

VENV_PY="/data/bench/qwen25vl72b_vllm_bench_20251222/.venv/bin/python"
OUT_DIR="/data/bench/qwen3_32b_vllm_tp2_acc_20251224/results"
RUN_HOME="/data/bench/home/azureuser"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
# /home 分区可能空间不足；将 HOME 指到 /data，避免各类默认缓存写入 /home。
export HOME=${HOME:-$RUN_HOME}
export HF_HOME=${HF_HOME:-/data/bench/hf}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/data/bench/hf}
export TMPDIR=${TMPDIR:-/data/bench/tmp}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/data/bench/cache/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/data/bench/cache/torchinductor}
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-/data/bench/cache/nv}

mkdir -p "$OUT_DIR"
mkdir -p "$HOME" "$TMPDIR"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH"

echo "[env] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[env] HOME=$HOME"
echo "[env] HF_HOME=$HF_HOME"
echo "[env] OUT_DIR=$OUT_DIR"
echo "[env] TRITON_CACHE_DIR=$TRITON_CACHE_DIR"

declare -a MODELS=(
  "RedHatAI/Qwen3-32B-NVFP4"
  "Qwen/Qwen3-32B-FP8"
  "Qwen/Qwen3-32B"
)

declare -a TASKS=(
  "hendrycks_math500:100"
  "aime24:50"
)

for MODEL in "${MODELS[@]}"; do
  for TASK_SPEC in "${TASKS[@]}"; do
    TASK="${TASK_SPEC%%:*}"
    LIMIT="${TASK_SPEC##*:}"

    SAFE_MODEL_TAG=$(echo "$MODEL" | tr '/:' '__')
    OUT_JSON="$OUT_DIR/lmeval_tp2_${SAFE_MODEL_TAG}_${TASK}_limit${LIMIT}.json"

    echo "[run] model=$MODEL task=$TASK limit=$LIMIT"

    "$VENV_PY" -m lm_eval \
      --model vllm \
      --model_args "pretrained=${MODEL},tensor_parallel_size=2,gpu_memory_utilization=0.90,max_model_len=4096,max_num_seqs=32,download_dir=/data/bench/hf" \
      --tasks "$TASK" \
      --batch_size 1 \
      --limit "$LIMIT" \
      --apply_chat_template \
      --gen_kwargs max_gen_toks=1024 \
      --log_samples \
      --output_path "$OUT_JSON"

    echo "[ok] wrote $OUT_JSON"
  done
done

echo "[done] all runs finished"
