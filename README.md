# Qwen2.5-VL-72B（FP8 dynamic）vLLM 吞吐对比：1× vs 2× RTX Pro 6000（MIG / PCIe / 跨 NUMA）

> 目标：在同一台 Azure VM（Standard_NC256ds_xl_RTXPRO6000BSE_v6，2× RTX Pro 6000 Blackwell，MIG 强制开启不可关闭）上，对比 **1× MIG device** 与 **2× MIG device（跨 PCIe/SYS，跨 NUMA）** 的推理吞吐量。

## 1. 测试环境

- 云平台：Azure
- VM：Standard_NC256ds_xl_RTXPRO6000BSE_v6（256 vCPU，2 GPU）
- OS：Ubuntu 22.04.5 LTS（kernel 6.8.0-1044-azure）
- GPU：NVIDIA RTX Pro 6000 Blackwell DC-4-96Q ×2
- Driver/CUDA：NVIDIA Driver 580.105.08 / CUDA 13.0（nvidia-smi 报告）
- MIG：强制开启且不可关闭
  - GPU0：1× `MIG 4g.96gb Device 0`
  - GPU1：1× `MIG 4g.96gb Device 0`
- GPU 拓扑：GPU0 <-> GPU1 为 `SYS`（PCIe + 跨 NUMA 的 SMP interconnect）
  - GPU0 CPU Affinity：0-63（NUMA 0）
  - GPU1 CPU Affinity：128-191（NUMA 2）
- 磁盘：`/data` 4TB 数据盘用于 HF cache / 日志 / 结果（根分区 `/` 约 29G，空间紧张）

## 2. 测试对象与精度说明

- 模型：`RedHatAI/Qwen2.5-VL-72B-Instruct-FP8-dynamic`
- 推理引擎：vLLM 0.13.0（OpenAI-compatible `vllm serve`）
- 精度/量化：
  - 权重：使用模型 checkpoint 自带的 **compressed-tensors 量化配置**（vLLM 日志显示 `quantization=compressed-tensors`）。
  - KV cache：显式启用 FP8（`--kv-cache-dtype fp8`）。
  - 计算 dtype：vLLM 日志显示 `dtype=torch.bfloat16`（即除权重/kv-cache 外，算子计算为 BF16）。

说明：在该 FP8 dynamic 模型上，显式传 `--quantization fp8` 会与模型自带配置冲突，因此本次以 checkpoint 配置为准，确保“权重 FP8 + KV cache FP8”。

## 3. 测试方法（统一口径）

- 统一点：同一台 VM、同一套 vLLM/torch 环境、同一模型与 `max_model_len`、同一压测脚本与参数
- 设备选择：
  - 1GPU：`CUDA_VISIBLE_DEVICES=0`（1 个 MIG device）
  - 2GPU：`CUDA_VISIBLE_DEVICES=0,1`（2 个 MIG device，分别来自两张物理卡）+ `--tensor-parallel-size 2`
- Attention backend：为避免 flashinfer JIT 对 nvcc 的依赖，固定使用 TRITON attention backend（通过环境变量）
- 压测工具：本仓库的 `tools/bench_openai.py`（拷贝到 VM 执行），对 `/v1/chat/completions` 发请求，并基于返回 `usage` 统计 token 吞吐

## 4. 实测过程与命令

### 4.1 目录与缓存

- 远端工作目录：`/data/bench/qwen25vl72b_vllm_bench_20251222/`
- HF 下载目录：`/data/bench/hf`

### 4.2 启动 vLLM（1GPU / TP=1）

> 通过 `ps aux` 捕获到的实际启动命令如下。

```bash
/data/bench/qwen25vl72b_vllm_bench_20251222/.venv/bin/vllm serve \
  RedHatAI/Qwen2.5-VL-72B-Instruct-FP8-dynamic \
  --download-dir /data/bench/hf \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype fp8 \
  --enable-force-include-usage
```

### 4.3 启动 vLLM（2GPU / TP=2）

首次尝试以 `--gpu-memory-utilization 0.95` 直接启动 TP=2 时，vLLM 在 `warming up sampler with 1024 dummy requests` 阶段触发 CUDA OOM。

为保证服务可稳定启动且不影响本次压测并发（c=16），增加 `--max-num-seqs 32` 以降低 warmup 峰值占用：

```bash
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
export TMPDIR=/data/bench/tmp
export XDG_CACHE_HOME=/data/bench/cache

nohup /data/bench/qwen25vl72b_vllm_bench_20251222/.venv/bin/vllm serve \
  RedHatAI/Qwen2.5-VL-72B-Instruct-FP8-dynamic \
  --download-dir /data/bench/hf \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype fp8 \
  --max-num-seqs 32 \
  --enable-force-include-usage \
  > logs/vllm_2gpu_retry.log 2>&1 &
```

### 4.4 压测命令（两组完全一致）

```bash
python bench_openai.py \
  --base-url http://127.0.0.1:8000 \
  --model RedHatAI/Qwen2.5-VL-72B-Instruct-FP8-dynamic \
  --concurrency 16 \
  --requests 64 \
  --max-tokens 256 \
  --temperature 0.0 \
  --timeout 1200 \
  --prompt-file qwen_bench_prompt.txt \
  --out results/bench_*.json
```

## 5. 结果

### 5.1 1GPU（TP=1）

- 结果文件（VM）：`/data/bench/qwen25vl72b_vllm_bench_20251222/results/bench_1gpu_c16_r64_mt256.json`

关键指标：
- QPS：1.0171
- prompt_tps：90.5189 tokens/s
- decode_tps：260.3688 tokens/s
- latency：p50=15.6923s，p95=15.8496s，mean=15.7242s

原始 JSON：
```json
{
  "endpoint": "http://127.0.0.1:8000/v1/chat/completions",
  "model": "RedHatAI/Qwen2.5-VL-72B-Instruct-FP8-dynamic",
  "concurrency": 16,
  "requests": 64,
  "max_tokens": 256,
  "temperature": 0.0,
  "stream": false,
  "ok": 64,
  "errors": 0,
  "total_time_s": 62.92611732300065,
  "qps": 1.0170657705049095,
  "latency_s": {
    "p50": 15.692299124998499,
    "p95": 15.849577032000525,
    "mean": 15.724194198515647,
    "min": 15.664447743998608,
    "max": 15.861515120999684
  },
  "ttft_s": {
    "p50": null,
    "p95": null,
    "mean": null,
    "min": null,
    "max": null
  },
  "token_usage": {
    "prompt_tokens_mean": 89.0,
    "completion_tokens_mean": 256.0
  },
  "derived": {
    "prompt_tps": 90.51885357493695,
    "decode_tps": 260.36883724925684,
    "ms_per_output_token": 61.42263358795175
  }
}
```

### 5.2 2GPU（TP=2，跨 PCIe/SYS）

- 结果文件（VM）：`/data/bench/qwen25vl72b_vllm_bench_20251222/results/bench_2gpu_c16_r64_mt256.json`

关键指标：
- QPS：1.3594
- prompt_tps：120.9863 tokens/s
- decode_tps：348.0055 tokens/s
- latency：p50=11.7613s，p95=11.7944s，mean=11.7622s

原始 JSON：
```json
{
  "endpoint": "http://127.0.0.1:8000/v1/chat/completions",
  "model": "RedHatAI/Qwen2.5-VL-72B-Instruct-FP8-dynamic",
  "concurrency": 16,
  "requests": 64,
  "max_tokens": 256,
  "temperature": 0.0,
  "stream": false,
  "ok": 64,
  "errors": 0,
  "total_time_s": 47.079713666000316,
  "qps": 1.359396542936476,
  "latency_s": {
    "p50": 11.761318839000523,
    "p95": 11.794378725249771,
    "mean": 11.762167871265603,
    "min": 11.728382113999032,
    "max": 11.799717240999598
  },
  "ttft_s": {
    "p50": null,
    "p95": null,
    "mean": null,
    "min": null,
    "max": null
  },
  "token_usage": {
    "prompt_tokens_mean": 89.0,
    "completion_tokens_mean": 256.0
  },
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
