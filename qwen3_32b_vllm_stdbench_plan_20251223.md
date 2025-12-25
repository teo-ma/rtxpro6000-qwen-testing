# Qwen3-32B（1× RTX Pro 6000 / MIG）三精度标准基准准确度测试计划（NVFP4 / FP8 / BF16）

> 日期：2025-12-23
>
> 目标：在 Azure VM（Standard_NC256ds_xl_RTXPRO6000BSE_v6）上仅使用 **1 个 MIG device（1×GPU）**，以 **vLLM** 作为推理引擎，对 Qwen3-32B 的 **NVFP4 / FP8 / BF16** 三种精度进行 7 个维度的标准基准测试，并保存可复现的结果 JSON 与日志。

## 0. 前置说明与约束

- **必须使用 vLLM**（不使用其它推理引擎）。
- **只用 1 个 GPU（1 个 MIG device）**：固定 `CUDA_VISIBLE_DEVICES=0`。
- 由于根分区空间紧张：HF 模型与 datasets cache 必须落到 `/data`。
- HLE（`cais/hle`）为 gated 数据集：需要先在 HF 账号申请权限并在 VM 上登录，否则将无法跑通（默认标注 N/A）。
- 部分基准（如 LiveCodeBench）数据量/依赖较重：建议先 `--limit` 冒烟，再决定全量。

## 1. 模型（3 精度）

- NVFP4：`RedHatAI/Qwen3-32B-NVFP4`
- FP8：`Qwen/Qwen3-32B-FP8`
- BF16：`Qwen/Qwen3-32B`

## 2. 任务与工具映射（7 个方面）

### 2.1 lm-evaluation-harness（lm-eval）

用于可直接在当前环境跑通的维度：

1) MMLU Pro：`mmlu_pro`
2) GPQA Diamond：`gpqa_diamond_zeroshot`
6) MATH-500：优先使用 lm-eval 内置 `hendrycks_math500`（注意：口径可能与 HF 上 “MATH-500” 不完全一致，报告中需注明）
7) AIME 2024：`aime24`

### 2.2 自定义 code pass@1（vLLM in-process）

用于环境中 lm-eval 未内置或更适合“生成+执行 tests”的维度：

4) LiveCodeBench：`livecodebench/code_generation_lite`（建议先用 lite）
5) SciCode：`Zilinghan/scicode`

评测脚本：`tools/eval_codegen_pass1_vllm.py`（pass@1：每题生成 1 次代码，执行 tests）

### 2.3 HLE（gated）

3) HLE：`cais/hle`（需要 HF 权限；未授权则跳过）

## 3. VM 连接

```bash
ssh -i ~/.ssh/azure_id_rsa azureuser@20.112.150.26
```

## 4. VM 环境变量（强烈建议统一设置）

```bash
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/data/bench/hf
export XDG_CACHE_HOME=/data/bench/hf
export TMPDIR=/data/bench/tmp
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
```

## 5. 输出目录（建议）

在 VM 上创建（按日期）：

- 工作目录：`/data/bench/qwen3_32b_vllm_stdbench_20251223/`
- 结果：`/data/bench/qwen3_32b_vllm_stdbench_20251223/results/`
- 日志：`/data/bench/qwen3_32b_vllm_stdbench_20251223/logs/`

## 6. 执行顺序（建议）

1) 冒烟：BF16 跑 `aime24` 或 `hendrycks_math500`，加 `--limit` 确认链路
2) 三精度跑 lm-eval：`aime24`、`hendrycks_math500`、`gpqa_diamond_zeroshot`、`mmlu_pro`
   - 其中 `mmlu_pro` 可能很慢，建议先 `--limit`，再按需要跑全量
3) 三精度跑 code pass@1：LiveCodeBench lite、SciCode（必要时先 `--limit`）
4) HLE：确认权限后再跑；否则 N/A

## 7. 命令模板

### 7.1 lm-eval（vLLM backend）

> 使用 VM 上已有 venv（示例）：`/data/bench/qwen25vl72b_vllm_bench_20251222/.venv/bin/python`

示例（BF16 + AIME24 + 冒烟）：

```bash
/data/bench/qwen25vl72b_vllm_bench_20251222/.venv/bin/python -m lm_eval \
  --model vllm \
  --model_args pretrained=Qwen/Qwen3-32B,tensor_parallel_size=1,gpu_memory_utilization=0.95,max_model_len=4096 \
  --tasks aime24 \
  --batch_size 1 \
  --apply_chat_template \
  --limit 20 \
  --log_samples \
  --output_path /data/bench/qwen3_32b_vllm_stdbench_20251223/results/lmeval_bf16_aime24_limit20.json
```

### 7.2 Code pass@1（LiveCodeBench / SciCode）

```bash
/data/bench/qwen25vl72b_vllm_bench_20251222/.venv/bin/python \
  /data/bench/qwen3_32b_vllm_stdbench_20251223/tools/eval_codegen_pass1_vllm.py \
  --dataset livecodebench/code_generation_lite \
  --split test \
  --model Qwen/Qwen3-32B \
  --download-dir /data/bench/hf \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.95 \
  --max-tokens 512 \
  --temperature 0.0 \
  --limit 50 \
  --timeout 20 \
  --out /data/bench/qwen3_32b_vllm_stdbench_20251223/results/lcb_lite_bf16_pass1_limit50.json
```

## 8. 本地汇总（在本仓库执行）

- lm-eval 结果：可用 `tools/summarize_lmeval_results.py` 汇总多个 JSON 到表格

```bash
python tools/summarize_lmeval_results.py artifacts/**/lmeval_*.json
```

