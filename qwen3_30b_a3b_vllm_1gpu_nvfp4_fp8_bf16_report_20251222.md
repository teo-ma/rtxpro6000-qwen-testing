# Qwen3-30B-A3B（1× RTX Pro 6000 / MIG）vLLM 推理吞吐测试（NVFP4 / FP8 / BF16）

> 日期：2025-12-22
>
> 目标：在同一台 Azure VM（Standard_NC256ds_xl_RTXPRO6000BSE_v6）上，仅使用 **1 个 GPU（1 个 MIG device）**，使用 **vLLM（OpenAI-compatible）** 对 Qwen3-30B-A3B 的不同权重格式做推理吞吐测试。

## 1. 测试计划（可复现口径）

### 1.1 测试对象

- NVFP4：`nvidia/Qwen3-30B-A3B-NVFP4`
- FP8：`nvidia/Qwen3-30B-A3B-FP8`
- BF16：`Qwen/Qwen3-30B-A3B`

### 1.2 统一口径（吞吐测试）

- 推理引擎：vLLM `0.13.0`，以 `vllm serve` 启动 OpenAI-compatible API
- 接口：`/v1/chat/completions`
- 负载：`concurrency=16`、`requests=64`、`max_tokens=256`、`temperature=0.0`
- Prompt：单条长文本（≥400 words），尽量保证输出会打满 256 tokens
- 设备：强制 `CUDA_VISIBLE_DEVICES=0`（仅 1 个 MIG device）

### 1.3 产出

- 每个模型一份结果 JSON（QPS、prompt_tps、decode_tps、p50/p95 等）
- vLLM 启动日志（用于追溯量化/后端/报错）

## 2. 测试环境

- 云平台：Azure
- VM：Standard_NC256ds_xl_RTXPRO6000BSE_v6（256 vCPU，2 GPU）
- OS：Ubuntu 22.04.5 LTS（kernel 6.8.0-1044-azure）
- GPU：NVIDIA RTX Pro 6000 Blackwell DC-4-96Q ×2
- Driver/CUDA：NVIDIA Driver 580.105.08 / CUDA 13.0（`nvidia-smi` 报告）
- MIG：强制开启且不可关闭
  - GPU0：1× `MIG 4g.96gb Device 0`
  - GPU1：1× `MIG 4g.96gb Device 0`
- 拓扑：GPU0 <-> GPU1 为 `SYS`；但本次只用 1 GPU
- 磁盘：`/data` 4TB（用于缓存/日志/结果）

## 3. 过程与命令（VM 上执行）

### 3.1 SSH 登录

```bash
ssh -i ~/.ssh/azure_id_rsa azureuser@20.112.150.26
```

### 3.2 工作目录

- 远端目录：`/data/bench/qwen3_30b_vllm_bench_20251222/`
- 复用既有 vLLM venv：`/data/bench/qwen25vl72b_vllm_bench_20251222/.venv/`（vLLM 0.13.0）

### 3.3 启动 vLLM（1GPU）

通用环境变量：

```bash
export CUDA_VISIBLE_DEVICES=0
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
export HF_HOME=/data/bench/hf
export XDG_CACHE_HOME=/data/bench/qwen3_30b_vllm_bench_20251222/cache
export TMPDIR=/data/bench/qwen3_30b_vllm_bench_20251222/tmp
```

以 NVFP4 为例：

```bash
nohup /data/bench/qwen25vl72b_vllm_bench_20251222/.venv/bin/vllm serve \
  nvidia/Qwen3-30B-A3B-NVFP4 \
  --download-dir /data/bench/hf \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 32 \
  --enable-force-include-usage \
  > /data/bench/qwen3_30b_vllm_bench_20251222/logs/vllm_nvfp4.log 2>&1 &

curl -s http://127.0.0.1:8000/v1/models | jq .
```

### 3.4 吞吐压测命令

```bash
cd /data/bench/qwen3_30b_vllm_bench_20251222

/data/bench/qwen25vl72b_vllm_bench_20251222/.venv/bin/python bench_openai.py \
  --base-url http://127.0.0.1:8000 \
  --model <MODEL_ID> \
  --concurrency 16 \
  --requests 64 \
  --max-tokens 256 \
  --temperature 0.0 \
  --timeout 1200 \
  --prompt-file qwen3_bench_prompt.txt \
  --out results/<OUT_JSON>
```

## 4. 结果

> 说明：`prompt_tps/decode_tps` 的统计依赖服务端返回 `usage` 字段；本次通过 vLLM 的 `--enable-force-include-usage` 强制输出 usage。

### 4.1 NVFP4（nvidia/Qwen3-30B-A3B-NVFP4）

- 结果 JSON：`artifacts/qwen3_30b_vllm_bench_20251222/bench_nvfp4_1gpu_c16_r64_mt256.json`
- vLLM 日志：`artifacts/qwen3_30b_vllm_bench_20251222/vllm_nvfp4.log`

关键指标：
- QPS：5.7538
- prompt_tps：632.9213 tokens/s
- decode_tps：1472.9804 tokens/s
- latency：p50=2.7731s，p95=2.9285s
- token_usage：prompt_tokens_mean=110.0，completion_tokens_mean=256.0

### 4.2 FP8（nvidia/Qwen3-30B-A3B-FP8）

- 状态：本次 **未能完成**（拉取模型 `config.json` 时返回 401 Unauthorized）
- vLLM 日志：`artifacts/qwen3_30b_vllm_bench_20251222/vllm_fp8.log`

如何解锁后重试：

1) 确认你的 Hugging Face 账号已获得该模型仓库访问权限（可能需要在网页端同意条款）
2) 在 VM 上登录 Hugging Face：

```bash
huggingface-cli login
# 或设置环境变量：export HF_TOKEN=...
```

3) 重新启动 vLLM：

```bash
/data/bench/qwen25vl72b_vllm_bench_20251222/.venv/bin/vllm serve nvidia/Qwen3-30B-A3B-FP8 ...
```

### 4.3 BF16（Qwen/Qwen3-30B-A3B）

- 结果 JSON：`artifacts/qwen3_30b_vllm_bench_20251222/bench_bf16_1gpu_c16_r64_mt256.json`
- vLLM 日志：`artifacts/qwen3_30b_vllm_bench_20251222/vllm_bf16.log`

关键指标：
- QPS：5.3453
- prompt_tps：587.9872 tokens/s
- decode_tps：1368.4067 tokens/s
- latency：p50=2.9831s，p95=3.2879s
- token_usage：prompt_tokens_mean=110.0，completion_tokens_mean=256.0

### 4.4 汇总对比（同口径：c=16 / r=64 / max_tokens=256）

| 权重 | 模型 | QPS | prompt_tps | decode_tps | p50 latency (s) | p95 latency (s) |
|---|---|---:|---:|---:|---:|---:|
| NVFP4 | nvidia/Qwen3-30B-A3B-NVFP4 | 5.7538 | 632.9213 | 1472.9804 | 2.7731 | 2.9285 |
| BF16 | Qwen/Qwen3-30B-A3B | 5.3453 | 587.9872 | 1368.4067 | 2.9831 | 3.2879 |
| FP8 | nvidia/Qwen3-30B-A3B-FP8 | N/A | N/A | N/A | N/A | N/A |

## 5. 结论与下一步

- 在本次 1GPU（MIG 4g.96gb）口径下：NVFP4 相比 BF16 体现出更高吞吐（decode_tps 约 +7.6%，p50 latency 更低）。
- FP8 版本本次因 Hugging Face 访问权限/鉴权问题未能完成；完成 `huggingface-cli login` 并确保账号具备权限后，可按相同命令快速补齐数据。
