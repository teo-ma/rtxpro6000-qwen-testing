#!/usr/bin/env bash
set -euo pipefail

# LiveBench evaluation for Qwen3-32B across BF16 / FP8 / NVFP4 on TP=2.
#
# LiveBench 官方推荐通过 OpenAI-compatible API（例如 vLLM server）进行评测。
# 默认只跑 non-coding 类别，避免 agentic coding 依赖 Docker 镜像占用大量磁盘（官方提示可达 ~150GB）。
# 如需包含 coding：把 BENCH_NAMES 里加上 live_bench/coding。

VLLM_PY=${VLLM_PY:-"/data/bench/qwen25vl72b_vllm_bench_20251222/.venv/bin/python"}
LIVEBENCH_PY=${LIVEBENCH_PY:-"/data/bench/LiveBench/.venv/bin/python"}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

RUN_HOME=${RUN_HOME:-"/data/bench/home/azureuser"}
export HOME=${HOME:-"$RUN_HOME"}
export TMPDIR=${TMPDIR:-"/data/bench/tmp"}
mkdir -p "$HOME" "$TMPDIR" >/dev/null 2>&1 || true

# HuggingFace cache 默认会落在 ~/.cache（本机 root 分区往往很小）。
# 为避免下载权重时磁盘不足，这里默认把 cache 指到 /data。
export HF_HOME=${HF_HOME:-"/data/bench/hf"}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-"$HF_HOME"}
export HF_HUB_CACHE=${HF_HUB_CACHE:-"$HF_HOME/hub"}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"$HF_HUB_CACHE"}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"$HF_HOME/datasets"}
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" >/dev/null 2>&1 || true

GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}

PORT=${PORT:-8000}
HOST=${HOST:-127.0.0.1}
API_BASE=${API_BASE:-"http://${HOST}:${PORT}/v1"}

LIVEBENCH_RELEASE_OPTION=${LIVEBENCH_RELEASE_OPTION:-"2024-11-25"}
MODE=${MODE:-"single"}
PARALLEL_REQUESTS=${PARALLEL_REQUESTS:-4}
# 注意：vLLM 会要求 input_tokens + max_tokens <= max_model_len。
# 这里 max_model_len 默认 4096，因此 max_tokens 不能也设为 4096，否则较长 prompt 会直接 400。
MAX_TOKENS=${MAX_TOKENS:-1024}

# LiveBench 的 run_livebench.py 会把 --venv 当作 "activate 脚本路径" 来 source
LIVEBENCH_VENV_ACTIVATE=${LIVEBENCH_VENV_ACTIVATE:-"/data/bench/LiveBench/.venv/bin/activate"}

# 注意：LiveBench 当前的 HuggingFace question_source 路径解析在指定单个 category（例如 live_bench/data_analysis）时
# 会把 category_name 截断为 data/instruction/...，从而尝试加载不存在的 HF dataset（livebench/data）。
# 为避免这个问题，默认使用整体 bench：live_bench（会覆盖所有类别）。
BENCH_NAMES=${BENCH_NAMES:-"live_bench"}

# Optional slicing (leave empty for full)
QUESTION_BEGIN=${QUESTION_BEGIN:-""}
QUESTION_END=${QUESTION_END:-""}

RUN_ID=${RUN_ID:-"$(date -u +%Y-%m-%dT%H-%M-%S)"}
ROOT=${ROOT:-"/data/bench/qwen3_32b_vllm_tp2_livebench_release${LIVEBENCH_RELEASE_OPTION}_${RUN_ID}"}
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

wait_api() {
  local url="$1"
  local sleep_s=2
  local wait_seconds=${WAIT_SECONDS:-1800}
  local tries=$((wait_seconds / sleep_s))
  for i in $(seq 1 $tries); do
    if curl -fsS "$url/models" >/dev/null 2>&1; then
      return 0
    fi
    if (( i % 15 == 0 )); then
      echo "[wait] vLLM not ready yet (${i}/${tries}) url=${url}/models" >&2
    fi
    sleep "$sleep_s"
  done
  return 1
}

run_one() {
  local tag="$1"; shift
  local model_id="$1"; shift

  local served="qwen3-32b-${tag}"
  local server_log="$LOG_DIR/vllm_server_${tag}.log"
  local run_log="$LOG_DIR/livebench_${tag}.log"
  local marker="$ROOT/.livebench_marker_${tag}"

  local server_pid=""
  cleanup() {
    if [ -n "$server_pid" ] && kill -0 "$server_pid" >/dev/null 2>&1; then
      echo "[cleanup] stopping vLLM pid=$server_pid" >&2
      kill "$server_pid" >/dev/null 2>&1 || true
      for _ in $(seq 1 30); do
        if ! kill -0 "$server_pid" >/dev/null 2>&1; then
          break
        fi
        sleep 1
      done
      if kill -0 "$server_pid" >/dev/null 2>&1; then
        echo "[cleanup] vLLM still alive; SIGKILL pid=$server_pid" >&2
        kill -9 "$server_pid" >/dev/null 2>&1 || true
      fi
      sleep 2
    fi

    # vLLM may leave orphan VLLM::Worker_* processes if killed hard.
    # Best-effort cleanup to avoid GPU memory leaks across runs.
    if command -v nvidia-smi >/dev/null 2>&1; then
      local pids
      pids=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | awk -F',' '$2 ~ /VLLM::Worker_/ {gsub(/^[[:space:]]+/,"",$1); print $1}' | tr '\n' ' ')
      if [ -n "${pids:-}" ]; then
        echo "[cleanup] killing orphan vLLM workers: $pids" >&2
        kill -9 $pids >/dev/null 2>&1 || true
        sleep 2
      fi
    fi
  }
  trap cleanup RETURN

  echo "[run] tag=$tag model=$model_id served_model_name=$served"

  # stop any existing server on port
  if lsof -iTCP:${PORT} -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "[warn] port ${PORT} already in use; killing listener"
    lsof -iTCP:${PORT} -sTCP:LISTEN -t | xargs -r kill -9 || true
    sleep 2
  fi

  # start vLLM OpenAI server
  nohup "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port "$PORT" \
    --model "$model_id" \
    --served-model-name "$served" \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --enforce-eager \
    --disable-custom-all-reduce \
    >"$server_log" 2>&1 &

  server_pid=$!
  echo "[server] pid=$server_pid log=$server_log"

  if ! wait_api "$API_BASE"; then
    echo "[error] vLLM server not ready: $API_BASE"
    tail -n 80 "$server_log" || true
    kill -9 "$server_pid" || true
    return 1
  fi

  # build livebench args
  local extra=( )
  if [ -n "$QUESTION_BEGIN" ]; then extra+=(--question-begin "$QUESTION_BEGIN"); fi
  if [ -n "$QUESTION_END" ]; then extra+=(--question-end "$QUESTION_END"); fi

  # run livebench
  cd /data/bench/LiveBench
  touch "$marker" || true
  export PATH="/data/bench/LiveBench/.venv/bin:$PATH"

  # run_livebench.py 会在当前目录执行 `python -u gen_api_answer.py` / `gen_ground_truth_judgment.py`
  # 但这两个脚本实际在 livebench/ 子目录中。用软链接兼容其相对路径假设。
  ln -sf livebench/gen_api_answer.py gen_api_answer.py
  ln -sf livebench/gen_ground_truth_judgment.py gen_ground_truth_judgment.py

  "$LIVEBENCH_PY" livebench/run_livebench.py \
    --model "$served" \
    --model-display-name "$served" \
    --bench-name $BENCH_NAMES \
    --question-source huggingface \
    --livebench-release-option "$LIVEBENCH_RELEASE_OPTION" \
    --venv "$LIVEBENCH_VENV_ACTIVATE" \
    --api-base "$API_BASE" \
    --api-key "EMPTY" \
    --mode "$MODE" \
    --parallel-requests "$PARALLEL_REQUESTS" \
    --max-tokens "$MAX_TOKENS" \
    "${extra[@]}" \
    2>&1 | tee "$run_log"

  # If livebench failed, stop early to avoid burning time on the next precision.
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "[error] livebench failed for tag=$tag (see $run_log)" >&2
    return 1
  fi

  # archive data files created/updated by this run
  local archive="$ROOT/livebench_${tag}_data_${RUN_ID}.tar.gz"
  (cd /data/bench/LiveBench && \
    find data -type f -newer "$marker" -print0 | \
    tar --null -czf "$archive" --files-from -) || true
  echo "[ok] archived $archive"

  echo "[ok] finished tag=$tag log=$run_log"
}

MODEL_BF16="Qwen/Qwen3-32B"
MODEL_FP8="Qwen/Qwen3-32B-FP8"
MODEL_NVFP4="RedHatAI/Qwen3-32B-NVFP4"

run_one "bf16" "$MODEL_BF16"
run_one "fp8" "$MODEL_FP8"
run_one "nvfp4" "$MODEL_NVFP4"

echo "DONE: $ROOT"
