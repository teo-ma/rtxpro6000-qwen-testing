## Qwen on RTX Pro 6000（Blackwell / MIG）评测与报告

这个仓库用于沉淀在同一台 Azure VM（Standard_NC256ds_xl_RTXPRO6000BSE_v6，2× RTX Pro 6000 Blackwell，MIG 强制开启）上完成的：

- vLLM 推理吞吐压测（OpenAI-compatible `/v1/chat/completions`）
- Qwen3-32B 三精度（NVFP4 / FP8 / BF16）准确度/一致性小集合评测
- Qwen3-32B 三精度标准基准（MMLU Pro / GPQA / HLE / LiveCodeBench / SciCode / MATH-500 / AIME 2024）评测方案与进展

> README 只做“概览与导航”。详细过程、完整命令与原始结果以各自报告为准。

## 报告导航（按测试类型）

### 1) Qwen2.5-VL-72B（FP8 dynamic）吞吐：1× vs 2× MIG（跨 NUMA）

- 报告：[`qwen2_5_vl_72b_fp8_vllm_1gpu_vs_2gpu_report_20251222.md`](qwen2_5_vl_72b_fp8_vllm_1gpu_vs_2gpu_report_20251222.md)
- 结果速览（c=16 / r=64 / max_tokens=256）：
  - 1× MIG（TP=1）：decode_tps ≈ 260.37 tokens/s
  - 2× MIG（TP=2）：decode_tps ≈ 348.01 tokens/s
  - 注：2× MIG 启动时为规避 warmup OOM 使用了 `--max-num-seqs 32`，详见报告

![1gpu-vs-2gpu-rtxpro6000](images/1gpu-vs-2gpu-rtxpro6000.jpg)

### 2) Qwen3-32B（NVFP4 / FP8 / BF16）吞吐（1× MIG）

- 报告：[`qwen3_32b_vllm_1gpu_nvfp4_fp8_bf16_report_20251222.md`](qwen3_32b_vllm_1gpu_nvfp4_fp8_bf16_report_20251222.md)
- 结果速览（同口径：c=16 / r=64 / max_tokens=256）：
  - NVFP4：decode_tps ≈ 606.47 tokens/s（最高）
  - FP8：decode_tps ≈ 381.87 tokens/s
  - BF16：decode_tps ≈ 322.09 tokens/s

![Throughput-qwen-30B-3Precision](images/Throughput-qwen-30B-3Precision.png)
![QPS-qwen-30B-3Precision](images/QPS-qwen-30B-3Precision.png)
![Latency-qwen-30B-3Precision](images/Latency-qwen-30B-3Precision.png)

### 3) Qwen3-32B（NVFP4 / FP8 / BF16）“可判定答案”准确度（40 题小集合）

- 报告：[`qwen3_32b_vllm_1gpu_accuracy_nvfp4_fp8_bf16_report_20251223.md`](qwen3_32b_vllm_1gpu_accuracy_nvfp4_fp8_bf16_report_20251223.md)
- 结果速览（40 题）：
  - NVFP4：34/40（0.85）
  - FP8：35/40（0.875）
  - BF16：35/40（0.875）

### 4) Qwen3-32B 标准基准（7 维度）评测方案与记录

- 报告：[`qwen3_32b_vllm_1gpu_standard_benchmarks_nvfp4_fp8_bf16_report_20251222.md`](qwen3_32b_vllm_1gpu_standard_benchmarks_nvfp4_fp8_bf16_report_20251222.md)
- 说明：
  - 该报告聚焦“如何在单 MIG + vLLM 下跑通 7 个维度”的方法与可复现命令，表格结果随进展回填
  - HLE（`cais/hle`）为 gated；GPQA（`Idavidrein/gpqa`）也可能需要权限/登录，未授权时会标注 N/A 或无法下载

## 仓库结构

- `tools/`
  - `bench_openai.py`：OpenAI-compatible 压测脚本（统计 QPS / prompt_tps / decode_tps / latency）
  - `eval_openai_accuracy.py`：40 题小集合准确度评测（温度 0，提取最终答案评分）
  - `eval_codegen_pass1_vllm.py`：代码类数据集 pass@1（生成代码并执行 tests）
  - `summarize_lmeval_results.py`：汇总 lm-eval 输出 JSON 到 Markdown
- `evalsets/`
  - `qwen3_32b_accuracy_suite_v1.jsonl`：40 题“可判定答案”小集合
- `artifacts/`
  - 评测原始 JSON、stdout、日志等（以日期目录分组）
- `images/`
  - 报告中引用的图表

## 复现提示（最小原则）

- 强约束：推理必须使用 vLLM；Qwen3-32B 三精度对比默认只用 1 个 MIG device（例如 `CUDA_VISIBLE_DEVICES=0`）
- 缓存建议：将 HF cache / datasets cache / TMPDIR 指向 `/data`（避免根分区空间不足）
- 具体可复现命令：请直接以对应报告中的 “环境与命令（VM 上执行）” 小节为准

## 注意事项

- gated 数据集：遇到 `DatasetNotFoundError: ... is a gated dataset` 时，需要在 Hugging Face 申请访问并在 VM 上登录后再跑。
- 指标口径：标准基准与小集合准确度不是同一类指标；对比时请以报告中注明的口径为准。
  "derived": {
    "prompt_tps": 120.98629232134637,
    "decode_tps": 348.00551499173787,
    "ms_per_output_token": 45.94596824713126
  }
}
```

### 5.3 对比汇总（同口径：c=16 / r=64 / max_tokens=256）

| 配置 | QPS | prompt_tps | decode_tps | p50 latency (s) | p95 latency (s) | ms/output_token |
|---|---:|---:|---:|---:|---:|---:|
| 1GPU（TP=1） | 1.0171 | 90.5189 | 260.3688 | 15.6923 | 15.8496 | 61.4226 |
| 2GPU（TP=2） | 1.3594 | 120.9863 | 348.0055 | 11.7613 | 11.7944 | 45.9460 |

2GPU 相对 1GPU 的提升：
- QPS：约 1.34×
- decode_tps：约 1.34×
- 平均延迟：约 0.75×（降低约 25%）

#### 5.3.1 Bar chart（Token 吞吐对比）

> 说明：下图只对比同量纲的 token 吞吐（prompt_tps / decode_tps），避免把 QPS 与 latency 混在同一坐标系。

![1GPU vs 2GPU token throughput（prompt_tps / decode_tps）](images/1gpu-vs-2gpu-rtxpro6000.jpg)

## 6. 结论与备注

- 结论（基于本次固定口径）：在该 VM 上，2× MIG device（跨 PCIe/SYS，跨 NUMA）相对 1× MIG device 的吞吐提升约 **1.34×**，未达到线性 2×。
- 主要影响因素推断：
  - GPU0<->GPU1 为 `SYS` 且跨 NUMA，TP=2 时的张量并行通信开销更高；
  - MIG 强制开启（每卡仅 1× `4g.96gb`），可用计算/带宽形态与“整卡直通”不同。
- 工程备注：
  - 2GPU 启动阶段需要通过 `--max-num-seqs` 降低 warmup 峰值占用，否则会在 sampler warmup 时 OOM；本次设置为 32，不影响 c=16 的压测并发。
  - vLLM 日志提示 FP8 attention scaling factor 可能未校准（q/prob_scale=1.0），这主要影响精度风险；本次聚焦吞吐对比。
