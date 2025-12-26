# Qwen3-32B（NVFP4 vs FP8 vs BF16）2×GPU（TP=2）3 小时内小样本准确度测试计划（vLLM）

目标：在 Azure VM `Standard_NC256ds_xl_RTXPRO6000BSE_v6`（2× RTX Pro 6000 Blackwell，MIG 开启）上，使用 **2 个 GPU（TP=2）**，对 Qwen3-32B 的 **3 种精度（NVFP4 vs FP8 vs BF16）** 在 7 项基准上做 **中小量样本**准确度评测，整体 **≤ 3 小时**完成并产出可对比结果。

约束与原则：
- 必须使用 vLLM（lm-eval 的 vLLM backend + code pass@1 的 in-process vLLM）。
- 使用 2 张卡：`CUDA_VISIBLE_DEVICES=0,1`，并设置 `tensor_parallel_size=2`。
- 由于 7 项全量耗时过长（尤其是 MATH-500 / 代码类），本计划只跑 **小样本 limit**；后续如需全量再单独跑。
- HLE 为 HF gated 数据集：无权限则跳过并标注 N/A。

## 1. 模型与精度
- BF16：`Qwen/Qwen3-32B`
- FP8：`Qwen/Qwen3-32B-FP8`
- NVFP4：`RedHatAI/Qwen3-32B-NVFP4`

## 2. 任务与小样本配置（建议值）
为了保证 3 小时内完成，建议每个精度使用以下 limits：

- **AIME 2024**：全量（题量很小）
- **MATH-500（hendrycks_math500）**：`--limit 50`，并设 `max_gen_toks=512`（更快）
- **MMLU Pro**：`--limit 200`
- **GPQA Diamond**：`--limit 100`
- **HLE**：`--limit 50`（仅在具备 gated 权限时）
- **LiveCodeBench（code_generation_lite）**：`--limit 20`，pass@1
- **SciCode**：`--limit 20`，pass@1

说明：
- AIME24 / Math500 属于生成式任务，必须显式设置 `max_gen_toks`，避免默认值过大导致超出 `max_model_len`。
- 代码类任务单题耗时高（生成 + 执行测试），因此限制样本数。

## 3. 目录与环境约定（VM）
统一输出到：`/data/bench/qwen3_32b_vllm_stdbench_tp2_3hr_20251223/`
- `results/`：lm-eval JSON、samples JSONL、pass@1 JSON
- `logs/`：完整 stdout/stderr

缓存（避免系统盘压力）：
- `HF_HOME=/data/bench/hf`
- `XDG_CACHE_HOME=/data/bench/hf`
- `TMPDIR=/data/bench/tmp`

## 4. 执行方式（脚本化，串行跑 3 精度）
### 4.1 lm-eval（AIME24 / MATH-500 / MMLU Pro / GPQA / 可选 HLE）
对每个精度依次运行，参数要点：
- `--model vllm`
- `--model_args pretrained=...,tensor_parallel_size=2,gpu_memory_utilization=0.92,max_model_len=4096`
- `CUDA_VISIBLE_DEVICES=0,1`

### 4.2 代码 pass@1（LiveCodeBench / SciCode）
使用仓库脚本：`tools/eval_codegen_pass1_vllm.py`
- 新增参数：`--tensor-parallel-size 2`
- `CUDA_VISIBLE_DEVICES=0,1`

## 5. 结果回收与汇总
- 从 VM 拉取 `results/` 与 `logs/` 回本仓库 `artifacts/qwen3_32b_vllm_stdbench_tp2_3hr_20251223/`
- lm-eval 结果可用 `tools/summarize_lmeval_results.py` 生成汇总表（如需我可继续补齐汇总脚本调用与报告模板）。

## 6. HLE（gated）前置条件
- 若 `cais/hle` 无法访问：在报告中标注 `N/A (HF gated)`。
- 若已具备权限：再启用 `--tasks hle`（本计划 limit 50）。
