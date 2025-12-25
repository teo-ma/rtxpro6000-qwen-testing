#!/usr/bin/env bash
set -euo pipefail

# Quick GSM8K accuracy comparison for Qwen3-32B across NVFP4/FP8/BF16.
# B method: enable_thinking=True, but force final output to be ONLY one line: "#### <number>".
# - Inference: vLLM via lm-evaluation-harness (in-process)
# - Parallelism: TP=2 (2 GPUs)
# - Scope: GSM8K limit=50 (fast sanity)

VENV_PY=${VENV_PY:-"/data/bench/qwen25vl72b_vllm_bench_20251222/.venv/bin/python"}
RUN_HOME=${RUN_HOME:-"/data/bench/home/azureuser"}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export HOME=${HOME:-$RUN_HOME}
export HF_HOME=${HF_HOME:-/data/bench/hf}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/data/bench/hf}
export TMPDIR=${TMPDIR:-/data/bench/tmp}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/data/bench/cache/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/data/bench/cache/torchinductor}
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-/data/bench/cache/nv}

limit=${LIMIT:-50}
ROOT=${ROOT:-"/data/bench/qwen3_32b_vllm_tp2_gsm8k_limit${limit}_think_finalnum_$(date -u +%Y%m%d)"}
OUT_DIR="$ROOT/results"
LOG_DIR="$ROOT/logs"

mkdir -p "$OUT_DIR" "$LOG_DIR" "$HOME" "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH"

echo "[env] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[env] VENV_PY=$VENV_PY"
echo "[env] ROOT=$ROOT"

task="gsm8k"

read -r -d '' SYS_INSTRUCTION <<'EOF' || true
You are solving a math word problem.
You may think silently.
If you use <think>...</think>, put all reasoning inside it.
After </think>, output EXACTLY one line in the format: "#### <answer>".
Do not include any other words, punctuation, units, or formatting.
EOF

GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
MAX_GEN_TOKS=${MAX_GEN_TOKS:-512}

run_one() {
  local tag="$1"; shift
  local model_id="$1"; shift

  local ts
  ts=$(date -u +%Y-%m-%dT%H-%M-%S)

  local out_json="$OUT_DIR/lmeval_${tag}_${task}_tp2_limit${limit}_think_finalnum_${ts}.json"
  local log="$LOG_DIR/lmeval_${tag}_${task}_tp2_limit${limit}_think_finalnum_${ts}.log"

  echo "[run] tag=$tag model=$model_id task=$task limit=$limit"

  set -x
  "$VENV_PY" -m lm_eval \
    --model vllm \
    --model_args "pretrained=${model_id},tensor_parallel_size=2,gpu_memory_utilization=${GPU_MEM_UTIL},max_model_len=${MAX_MODEL_LEN},max_num_seqs=${MAX_NUM_SEQS},enforce_eager=True,disable_custom_all_reduce=True,enable_thinking=True,download_dir=/data/bench/hf" \
    --tasks "$task" \
    --batch_size 1 \
    --limit "$limit" \
    --apply_chat_template \
    --system_instruction "$SYS_INSTRUCTION" \
    --gen_kwargs "temperature=0.0,max_gen_toks=${MAX_GEN_TOKS}" \
    --log_samples \
    --output_path "$out_json" \
    2>&1 | tee "$log"
  set +x

  echo "[ok] wrote $out_json"
}

MODEL_BF16="Qwen/Qwen3-32B"
MODEL_FP8="Qwen/Qwen3-32B-FP8"
MODEL_NVFP4="RedHatAI/Qwen3-32B-NVFP4"

run_one "bf16" "$MODEL_BF16"
run_one "fp8" "$MODEL_FP8"
run_one "nvfp4" "$MODEL_NVFP4"

echo "DONE: $OUT_DIR"