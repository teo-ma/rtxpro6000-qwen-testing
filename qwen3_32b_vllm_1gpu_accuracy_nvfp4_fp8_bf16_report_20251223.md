# Qwen3-32B（1× RTX Pro 6000 / MIG）vLLM 推理“准确度”测试（NVFP4 / FP8 / BF16）

> 日期：2025-12-23
>
> 目标：在同一台 Azure VM（Standard_NC256ds_xl_RTXPRO6000BSE_v6）上，仅使用 **1 个 GPU（1 个 MIG device）**，使用 **vLLM（OpenAI-compatible）** 对 Qwen3-32B 的三个精度权重做小规模、可复现的“准确度/一致性”检查。
>
> 说明：这里的“准确度”使用一套自包含的、可判定答案的 40 题小集合（算术/格式化/基础逻辑），用于对比不同精度下的输出是否正确、是否出现明显退化。它不是 MMLU/GSM8K 等标准学术基准。

## 1. 测试计划（可复现口径）

### 1.1 测试对象

- NVFP4：`RedHatAI/Qwen3-32B-NVFP4`
- FP8：`Qwen/Qwen3-32B-FP8`
- BF16：`Qwen/Qwen3-32B`

### 1.2 评测集与指标

- 评测集：`evalsets/qwen3_32b_accuracy_suite_v1.jsonl`
  - 题量：40
  - 类型：
    - `number`：期望输出最终数值
    - `exact/contains`：期望输出指定字符串（允许少量格式差异）
- 指标：`accuracy = ok / count`

### 1.3 推理与评测参数

- 推理引擎：vLLM（OpenAI-compatible）
- 设备：`CUDA_VISIBLE_DEVICES=0`（仅 1 个 MIG device）
- 评测脚本：`tools/eval_openai_accuracy.py`
- 生成参数：`temperature=0.0`、`max_tokens=256`
- 并发：`concurrency=4`

> 备注：Qwen3 有时会输出 `<think>` 段。评测脚本通过 system message 强制“只输出最终答案”，并在 `number` 题型下取“最后一个数字”作为最终答案进行对比。

## 2. 环境与命令（VM 上执行）

### 2.1 SSH 登录

```bash
ssh -i ~/.ssh/azure_id_rsa azureuser@20.112.150.26
```

### 2.2 远端工作目录

- 远端目录：`/data/bench/qwen3_32b_vllm_acc_20251223/`

### 2.3 启动 vLLM（示例：NVFP4）

```bash
export CUDA_VISIBLE_DEVICES=0
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
export HF_HOME=/data/bench/hf
export XDG_CACHE_HOME=/data/bench/qwen3_32b_vllm_acc_20251223/cache
export TMPDIR=/data/bench/qwen3_32b_vllm_acc_20251223/tmp

nohup /data/bench/qwen25vl72b_vllm_bench_20251222/.venv/bin/vllm serve \
  RedHatAI/Qwen3-32B-NVFP4 \
  --download-dir /data/bench/hf \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 16 \
  --enable-force-include-usage \
  > /data/bench/qwen3_32b_vllm_acc_20251223/logs/vllm_nvfp4.log 2>&1 &

curl -s http://127.0.0.1:8000/v1/models | jq .
```

### 2.4 运行评测脚本

```bash
/data/bench/qwen25vl72b_vllm_bench_20251222/.venv/bin/python \
  /data/bench/qwen3_32b_vllm_acc_20251223/tools/eval_openai_accuracy.py \
  --base-url http://127.0.0.1:8000 \
  --model RedHatAI/Qwen3-32B-NVFP4 \
  --evalset /data/bench/qwen3_32b_vllm_acc_20251223/evalsets/qwen3_32b_accuracy_suite_v1.jsonl \
  --concurrency 4 \
  --max-tokens 256 \
  --temperature 0.0 \
  --timeout 1200 \
  --out /data/bench/qwen3_32b_vllm_acc_20251223/results/acc_nvfp4.json
```

## 3. 结果

- NVFP4：accuracy = 0.85（34/40）
  - 结果：`artifacts/qwen3_32b_vllm_acc_20251223/acc_nvfp4.json`
- FP8：accuracy = 0.875（35/40）
  - 结果：`artifacts/qwen3_32b_vllm_acc_20251223/acc_fp8.json`
- BF16：accuracy = 0.875（35/40）
  - 结果：`artifacts/qwen3_32b_vllm_acc_20251223/acc_bf16.json`

汇总表：

| 精度 | 模型 | ok / count | accuracy |
|---|---|---:|---:|
| NVFP4 | RedHatAI/Qwen3-32B-NVFP4 | 34/40 | 0.85 |
| FP8 | Qwen/Qwen3-32B-FP8 | 35/40 | 0.875 |
| BF16 | Qwen/Qwen3-32B | 35/40 | 0.875 |

![Qwen3-32B（40 题）可判定答案准确度对比](images/qwen3_32b_vllm_1gpu_accuracy_nvfp4_fp8_bf16_40questions.png)

## 4. 结论与下一步

- 在这套 40 题的可判定集合上，FP8 与 BF16 的通过率一致，NVFP4 略低。
- 如果你希望更贴近“真实任务准确率”（如代码、数学、对话能力），建议引入标准数据集（例如 GSM8K/MMLU 子集）并定义更严格的评分规则；我也可以在保持 vLLM 推理引擎的前提下，把评测脚本扩展为支持 Hugging Face `datasets` 的小样本评测。
