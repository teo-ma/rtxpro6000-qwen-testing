# Qwen3-32B（2× RTX Pro 6000 / MIG，TP=2）准确率小规模对比（NVFP4 vs FP8 vs BF16）

> 日期：2025-12-24
>
> 目标：在同一台 Azure VM（Standard_NC256ds_xl_RTXPRO6000BSE_v6，2× GPU，MIG 强制开启）上，使用 **vLLM** 推理引擎，开启 **TP=2**，对 Qwen3-32B 三种精度权重做小规模准确率对比。
>
> 任务与题量（3 小时内目标）：
> - MATH-500：抽样/限制 100 题
> - AIME 2024：抽样/限制 50 题

## 1. 评测对象（同口径）

- NVFP4：`RedHatAI/Qwen3-32B-NVFP4`
- FP8：`Qwen/Qwen3-32B-FP8`
- BF16：`Qwen/Qwen3-32B`

## 2. 方法选择（为什么不用 vllm bench）

- `vllm bench` 更偏性能（吞吐/延迟）基准测试；
- 本计划的“准确率”使用 **lm-evaluation-harness（lm-eval）** 来做标准数据集评分；推理后端使用 `--model vllm`（in-process），确保“必须使用 vLLM”。

## 3. 关键约束与缓存（VM 上必须设置）

根分区空间紧张，HF/datasets cache 必须放到 `/data`。

建议在 VM 上统一设置：

```bash
export CUDA_VISIBLE_DEVICES=0,1
# /home 分区可能空间不足；将 HOME 指到 /data，避免默认缓存写入 /home。
export HOME=/data/bench/home/azureuser
export HF_HOME=/data/bench/hf
export XDG_CACHE_HOME=/data/bench/hf
export TMPDIR=/data/bench/tmp
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
export TRITON_CACHE_DIR=/data/bench/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/data/bench/cache/torchinductor
export CUDA_CACHE_PATH=/data/bench/cache/nv
```

## 4. 任务映射与口径说明

- MATH-500：使用 lm-eval 内置任务 `hendrycks_math500`
  - 注意：该任务名与常见的“HF 上的 MATH-500”并非必然同一实现；报告中需要明确口径。
- AIME 2024：使用 lm-eval 内置任务 `aime24`

## 5. 执行目录与产出

建议在 VM 上建立独立工作目录：

- 工作目录：`/data/bench/qwen3_32b_vllm_tp2_acc_20251224/`
- 输出目录：`/data/bench/qwen3_32b_vllm_tp2_acc_20251224/results/`

输出为 lm-eval JSON（每个模型×每个任务各一份），后续拷回本仓库 `artifacts/` 统一归档。

## 6. VM 上可复现命令（单次运行示例）

> 复用 venv（与既有报告一致）：`/data/bench/qwen25vl72b_vllm_bench_20251222/.venv/`

### 6.1 BF16 + AIME24（limit=50）

```bash
/data/bench/qwen25vl72b_vllm_bench_20251222/.venv/bin/python -m lm_eval \
  --model vllm \
  --model_args pretrained=Qwen/Qwen3-32B,tensor_parallel_size=2,gpu_memory_utilization=0.90,max_model_len=4096,max_num_seqs=32,download_dir=/data/bench/hf \
  --tasks aime24 \
  --batch_size 1 \
  --limit 50 \
  --apply_chat_template \
  --gen_kwargs max_gen_toks=1024 \
  --log_samples \
  --output_path /data/bench/qwen3_32b_vllm_tp2_acc_20251224/results/lmeval_tp2_bf16_aime24_limit50.json
```

### 6.2 BF16 + MATH500（limit=100）

```bash
/data/bench/qwen25vl72b_vllm_bench_20251222/.venv/bin/python -m lm_eval \
  --model vllm \
  --model_args pretrained=Qwen/Qwen3-32B,tensor_parallel_size=2,gpu_memory_utilization=0.90,max_model_len=4096,max_num_seqs=32,download_dir=/data/bench/hf \
  --tasks hendrycks_math500 \
  --batch_size 1 \
  --limit 100 \
  --apply_chat_template \
  --gen_kwargs max_gen_toks=1024 \
  --log_samples \
  --output_path /data/bench/qwen3_32b_vllm_tp2_acc_20251224/results/lmeval_tp2_bf16_hendrycks_math500_limit100.json
```

## 7. 建议执行顺序（3 小时内）

按“单模型跑两项任务”顺序串行执行，避免并行多模型抢显存：

1) NVFP4：MATH500(100) → AIME24(50)
2) FP8：MATH500(100) → AIME24(50)
3) BF16：MATH500(100) → AIME24(50)

如果时间不足，优先保证三精度都完成 AIME24(50)，再补 MATH500。

## 8. 结果汇总（在本仓库本地运行）

把 VM 上的结果 JSON 拷回本仓库 `artifacts/<日期目录>/` 后，可用：

```bash
python tools/summarize_lmeval_results.py artifacts/**/lmeval_tp2_*.json
```

输出将给出每个任务的主指标（例如 exact_match / acc 等）。
