# Qwen3-32B（1× RTX Pro 6000 / MIG）标准基准“准确度”对比（NVFP4 vs FP8 vs BF16）

> 日期：2025-12-22
>
> 目标：在同一台 Azure VM（Standard_NC256ds_xl_RTXPRO6000BSE_v6，MIG 强制开启）上，仅使用 **1 个 MIG device**，以 **vLLM** 作为推理引擎，对 Qwen3-32B 三种精度权重按以下维度做准确度评测：
>
> - MMLU Pro
> - GPQA Diamond
> - HLE
> - LiveCodeBench
> - SciCode
> - MATH-500
> - AIME 2024

## 1. 评测对象（同口径）

- NVFP4：`RedHatAI/Qwen3-32B-NVFP4`
- FP8：`Qwen/Qwen3-32B-FP8`
- BF16：`Qwen/Qwen3-32B`

## 2. 环境与关键约束

- VM：Standard_NC256ds_xl_RTXPRO6000BSE_v6
- GPU：2× RTX Pro 6000 Blackwell，MIG 强制开启
- 本次仅使用 1 个 MIG device：`CUDA_VISIBLE_DEVICES=0`
- 磁盘：根分区空间紧张，HF/datasets cache 必须放到 `/data`

建议在 VM 上统一设置：

```bash
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/data/bench/hf
export XDG_CACHE_HOME=/data/bench/hf
export TMPDIR=/data/bench/tmp
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
```

## 3. 评测方法与任务映射

> 说明：为尽量对齐“标准基准维度”，这里采用两条路径：
>
> - 可直接用 lm-eval 的：MMLU Pro、GPQA Diamond、MATH-500（近似项）、AIME 2024
> - lm-eval 当前环境未内置/难以直接跑通的：LiveCodeBench、SciCode（使用 vLLM in-process + 执行 tests 得到 pass@1）
>
> HLE 数据集为 gated（需要 HF 权限），默认标注 N/A。

### 3.1 lm-evaluation-harness（lm-eval）任务

- MMLU Pro：`mmlu_pro`（lm-eval 内部为 14 个子域，默认 5-shot）
- GPQA Diamond：`gpqa_diamond_zeroshot`
- MATH-500：优先使用 lm-eval 内置的 `hendrycks_math500`（注意：这不是 HuggingFaceH4/MATH-500 的同名实现；口径差异需在报告中说明）
- AIME 2024：`aime24`

### 3.2 自定义 code pass@1（LiveCodeBench / SciCode）

- LiveCodeBench：`livecodebench/code_generation_lite`（避免全量数据集占用巨大缓存；报告中明确这一点）
- SciCode：`Zilinghan/scicode`
- 评测脚本：`tools/eval_codegen_pass1_vllm.py`
  - 指标：pass@1（每题生成 1 次代码，执行 tests，成功即 pass）

## 4. VM 上可复现命令

### 4.1 lm-eval（vLLM backend）

> venv：复用既有环境：`/data/bench/qwen25vl72b_vllm_bench_20251222/.venv/`

示例（BF16 + AIME24）：

```bash
/data/bench/qwen25vl72b_vllm_bench_20251222/.venv/bin/python -m lm_eval \
  --model vllm \
  --model_args pretrained=Qwen/Qwen3-32B,tensor_parallel_size=1,gpu_memory_utilization=0.95,max_model_len=4096 \
  --tasks aime24 \
  --batch_size 1 \
  --apply_chat_template \
  --log_samples \
  --output_path /data/bench/qwen3_32b_vllm_stdbench_20251222/results/lmeval_bf16_aime24.json
```

> 提示：MMLU Pro 运行时间可能很长（默认 5-shot 且题量大）。建议先用 `--limit` 做 smoke，再决定是否全量跑。

### 4.2 LiveCodeBench / SciCode（vLLM in-process）

```bash
/data/bench/qwen25vl72b_vllm_bench_20251222/.venv/bin/python \
  /data/bench/qwen3_32b_vllm_stdbench_20251222/tools/eval_codegen_pass1_vllm.py \
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
  --out /data/bench/qwen3_32b_vllm_stdbench_20251222/results/lcb_lite_bf16_pass1.json
```

同理可把 `--dataset` 改为 `Zilinghan/scicode`。

## 5. 结果（待回填）

> 当前阻塞项：需要恢复/确认 VM SSH 连通性与公网 IP；以及 HLE 数据集访问权限。

| 维度 | 指标 | NVFP4 | FP8 | BF16 | 备注 |
|---|---|---:|---:|---:|---|
| MMLU Pro | exact_match |  |  |  | lm-eval `mmlu_pro`（默认 5-shot） |
| GPQA Diamond | exact_match |  |  |  | lm-eval `gpqa_diamond_zeroshot` |
| HLE | exact_match | N/A | N/A | N/A | `cais/hle` gated（需 HF 权限） |
| LiveCodeBench | pass@1 |  |  |  | 先用 `code_generation_lite` + limit（见方法） |
| SciCode | pass@1 |  |  |  | 执行 tests 得 pass@1（见方法） |
| MATH-500 | exact_match |  |  |  | lm-eval `hendrycks_math500`（口径注意） |
| AIME 2024 | exact_match |  |  |  | lm-eval `aime24` |

## 6. 下一步

- 恢复 VM SSH 连通性后：按同口径依次跑 NVFP4 vs FP8 vs BF16，落盘 JSON 到 `artifacts/` 并回填表格。
- 如要跑 HLE：需要用 HF 账号申请 `cais/hle` 访问权限，并在 VM 上 `huggingface-cli login`。
