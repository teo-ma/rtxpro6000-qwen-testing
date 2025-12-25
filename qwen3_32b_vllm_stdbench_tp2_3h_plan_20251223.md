# Qwen3-32B（2× RTX Pro 6000 / MIG，TP=2）三精度“3 小时内”标准基准准确度测试计划（NVFP4 / FP8 / BF16）

> 日期：2025-12-23
>
> 目标：在 Azure VM（Standard_NC256ds_xl_RTXPRO6000BSE_v6）上使用 **2 个 MIG device（2×GPU）**，以 **vLLM** 作为推理引擎，开启 **TP=2**，对 Qwen3-32B 的 **NVFP4 / FP8 / BF16** 三种精度进行 7 个维度的“中小量样本”准确度测试，并保证 **总时长约 3 小时内**可跑完一轮，产出可复现的 JSON 与日志。

## 0. 前置说明与约束

- **必须使用 vLLM**（不使用其它推理引擎）。
- **使用 2×GPU（2×MIG device）且 TP=2**：
  - `export CUDA_VISIBLE_DEVICES=0,1`
  - lm-eval：`--model_args ... tensor_parallel_size=2 ...`
  - code pass@1：`--tensor-parallel-size 2`
- 若 vLLM 在 TP=2 初始化阶段出现 warmup OOM（常见报错："warming up sampler with 1024 dummy requests"），建议：
  - 将 `max_num_seqs` 下调（例如 `max_num_seqs=16`）
  - 将 `gpu_memory_utilization` 略降（例如 `0.90`）
- 根分区空间紧张：HF 模型与 datasets cache 必须落到 `/data`。
- HLE（`cais/hle`）为 gated 数据集：若 HF 账号未获授权则无法跑通，按 **N/A** 处理（不计入 3 小时预算）。

## 1. 模型（3 精度）

- NVFP4：`RedHatAI/Qwen3-32B-NVFP4`
- FP8：`Qwen/Qwen3-32B-FP8`
- BF16：`Qwen/Qwen3-32B`

## 2. 任务与工具映射（7 个方面）

### 2.1 lm-evaluation-harness（lm-eval + vLLM backend）

1) MMLU Pro：`mmlu_pro`
2) GPQA Diamond：`gpqa_diamond_zeroshot`
6) MATH-500：`hendrycks_math500`
7) AIME 2024：`aime24`

### 2.2 自定义 code pass@1（vLLM in-process）

4) LiveCodeBench：`livecodebench/code_generation_lite`
5) SciCode：`Zilinghan/scicode`

评测脚本：`tools/eval_codegen_pass1_vllm.py`

### 2.3 HLE（gated）

3) HLE：`cais/hle`（需 HF 权限；未授权则跳过）

## 3. 建议样本量（确保 3 小时内）

> 这是“中小量样本”策略：用较小 `--limit` 获得趋势性对比。

- AIME24：`--limit 15`（半量）
- hendrycks_math500：`--limit 10`
- GPQA Diamond：`--limit 50`
- MMLU Pro：`--limit 100`
- LiveCodeBench lite（pass@1）：`--limit 5`
- SciCode（pass@1）：`--limit 5`
- HLE：默认跳过（N/A）

生成长度建议：
- AIME24 / MATH-500：`--gen_kwargs max_gen_toks=1024`（避免截断）
- MMLU Pro / GPQA：`--gen_kwargs max_gen_toks=64`（加速）

## 4. VM 连接

```bash
ssh -i ~/.ssh/azure_id_rsa azureuser@20.112.150.26
```

## 5. VM 环境变量（统一设置）

```bash
export CUDA_VISIBLE_DEVICES=0,1
export HF_HOME=/data/bench/hf
export XDG_CACHE_HOME=/data/bench/hf
export TMPDIR=/data/bench/tmp
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
```

## 6. 输出目录（建议）

- 工作目录：`/data/bench/qwen3_32b_vllm_stdbench_tp2_3h_20251223/`
- 结果：`/data/bench/qwen3_32b_vllm_stdbench_tp2_3h_20251223/results/`
- 日志：`/data/bench/qwen3_32b_vllm_stdbench_tp2_3h_20251223/logs/`

## 7. 命令模板

> 使用 VM 上已有 venv（示例）：`/data/bench/qwen25vl72b_vllm_bench_20251222/.venv/bin/python`

### 7.1 lm-eval（AIME24 + MATH-500，小样本，TP=2）

```bash
/data/bench/qwen25vl72b_vllm_bench_20251222/.venv/bin/python -m lm_eval \
  --model vllm \
  --model_args pretrained=Qwen/Qwen3-32B,tensor_parallel_size=2,gpu_memory_utilization=0.95,max_model_len=4096 \
  --tasks aime24,hendrycks_math500 \
  --batch_size 1 \
  --apply_chat_template \
  --limit 15 \
  --gen_kwargs max_gen_toks=1024 \
  --log_samples \
  --output_path /data/bench/qwen3_32b_vllm_stdbench_tp2_3h_20251223/results/lmeval_bf16_math_aime_tp2.json
```

> 注：`--limit` 对多任务时会同时作用于每个 task；若需分别控制，可拆成两条命令。

### 7.2 lm-eval（MMLU Pro + GPQA，小样本，TP=2）

```bash
/data/bench/qwen25vl72b_vllm_bench_20251222/.venv/bin/python -m lm_eval \
  --model vllm \
  --model_args pretrained=Qwen/Qwen3-32B,tensor_parallel_size=2,gpu_memory_utilization=0.95,max_model_len=4096 \
  --tasks mmlu_pro,gpqa_diamond_zeroshot \
  --batch_size 1 \
  --apply_chat_template \
  --limit 100 \
  --gen_kwargs max_gen_toks=64 \
  --log_samples \
  --output_path /data/bench/qwen3_32b_vllm_stdbench_tp2_3h_20251223/results/lmeval_bf16_mmlu_gpqa_tp2.json
```

### 7.3 Code pass@1（LiveCodeBench / SciCode，小样本，TP=2）

```bash
/data/bench/qwen25vl72b_vllm_bench_20251222/.venv/bin/python \
  /data/bench/qwen3_32b_vllm_stdbench_tp2_3h_20251223/tools/eval_codegen_pass1_vllm.py \
  --dataset livecodebench/code_generation_lite \
  --split test \
  --model Qwen/Qwen3-32B \
  --download-dir /data/bench/hf \
  --max-model-len 4096 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.95 \
  --max-tokens 512 \
  --temperature 0.0 \
  --limit 5 \
  --timeout 20 \
  --out /data/bench/qwen3_32b_vllm_stdbench_tp2_3h_20251223/results/lcb_lite_bf16_pass1_tp2_limit5.json
```

## 8. 执行顺序（建议）

- 以“**每个精度一组**”的方式跑，便于中途对比与控时：
  1) BF16：AIME+MATH → MMLU+GPQA → LCB → SciCode
  2) FP8：同上
  3) NVFP4：同上

## 9. 本地汇总（在本仓库执行）

```bash
python tools/summarize_lmeval_results.py artifacts/**/lmeval_*.json
```
