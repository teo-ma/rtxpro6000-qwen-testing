# Qwen3-32B（1× RTX Pro 6000 / MIG）vLLM 推理吞吐测试（NVFP4 vs FP8 vs BF16）

> 日期：2025-12-22
>
> 目标：在同一台 Azure VM（Standard_NC256ds_xl_RTXPRO6000BSE_v6）上，仅使用 **1 个 GPU（1 个 MIG device）**，使用 **vLLM（OpenAI-compatible）** 对 Qwen3-32B 的三个精度权重做推理吞吐测试。

## 1. 测试计划（可复现口径）

### 1.1 测试对象

- NVFP4：`RedHatAI/Qwen3-32B-NVFP4`
- FP8：`Qwen/Qwen3-32B-FP8`
- BF16：`Qwen/Qwen3-32B`

### 1.2 统一口径（吞吐测试）

- 推理引擎：vLLM `0.13.0`（VM 上既有 venv）
- 接口：`/v1/chat/completions`
- 负载：`concurrency=16`、`requests=64`、`max_tokens=256`、`temperature=0.0`
- Prompt：单条长文本（本次统计 `prompt_tokens_mean≈110`）
- 设备：强制 `CUDA_VISIBLE_DEVICES=0`（仅 1 个 MIG device）

### 1.3 产出

- 每个模型一份结果 JSON（QPS、prompt_tps、decode_tps、p50/p95 latency 等）
- 每个模型一份 vLLM 启动与运行日志（用于追溯量化/后端/报错）

## 2. 测试环境

- 云平台：Azure
- VM：Standard_NC256ds_xl_RTXPRO6000BSE_v6（256 vCPU，2 GPU）
- OS：Ubuntu 22.04.5 LTS（kernel 6.8.0-1044-azure）
- GPU：NVIDIA RTX Pro 6000 Blackwell DC-4-96Q ×2
- MIG：强制开启且不可关闭；本次仅使用 GPU0 上的 1× `4g.96gb` MIG device
- 磁盘：`/data` 4TB（用于缓存/日志/结果）

## 3. 过程与命令（VM 上执行）

### 3.1 SSH 登录

```bash
ssh -i ~/.ssh/azure_id_rsa azureuser@<ip address>
```

### 3.2 工作目录

- 远端目录：`/data/bench/qwen3_32b_vllm_bench_20251222/`
- 复用 vLLM venv：`/data/bench/qwen25vl72b_vllm_bench_20251222/.venv/`

### 3.3 启动 vLLM（1GPU）

通用环境变量：

```bash
export CUDA_VISIBLE_DEVICES=0
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
export HF_HOME=/data/bench/hf
export XDG_CACHE_HOME=/data/bench/qwen3_32b_vllm_bench_20251222/cache
export TMPDIR=/data/bench/qwen3_32b_vllm_bench_20251222/tmp
```

以 NVFP4 为例：

```bash
nohup /data/bench/qwen25vl72b_vllm_bench_20251222/.venv/bin/vllm serve \
  RedHatAI/Qwen3-32B-NVFP4 \
  --download-dir /data/bench/hf \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 32 \
  --enable-force-include-usage \
  > /data/bench/qwen3_32b_vllm_bench_20251222/logs/vllm_nvfp4.log 2>&1 &

curl -s http://127.0.0.1:8000/v1/models | jq .
```

### 3.4 吞吐压测命令

```bash
cd /data/bench/qwen3_32b_vllm_bench_20251222

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

> 注意：`--base-url` 必须是 `http://127.0.0.1:8000`（不要额外带 `/v1`），否则会变成 `/v1/v1/chat/completions` 导致 404。

## 4. 结果

> 说明：`prompt_tps/decode_tps` 的统计依赖服务端返回 `usage` 字段；本次通过 vLLM 的 `--enable-force-include-usage` 强制输出 usage。

### 4.1 NVFP4（RedHatAI/Qwen3-32B-NVFP4）

- 结果 JSON：`artifacts/qwen3_32b_vllm_bench_20251222/bench_nvfp4_1gpu_c16_r64_mt256.json`
- vLLM 日志：`artifacts/qwen3_32b_vllm_bench_20251222/vllm_nvfp4.log`

关键指标：
- QPS：2.3690
- prompt_tps：260.5918 tokens/s
- decode_tps：606.4682 tokens/s
- latency：p50=6.7577s，p95=6.7950s
- token_usage：prompt_tokens_mean=110.0，completion_tokens_mean=256.0

### 4.2 FP8（Qwen/Qwen3-32B-FP8）

- 结果 JSON：`artifacts/qwen3_32b_vllm_bench_20251222/bench_fp8_1gpu_c16_r64_mt256.json`
- vLLM 日志：`artifacts/qwen3_32b_vllm_bench_20251222/vllm_fp8.log`

关键指标：
- QPS：1.4917
- prompt_tps：164.0849 tokens/s
- decode_tps：381.8702 tokens/s
- latency：p50=10.6898s，p95=10.8637s
- token_usage：prompt_tokens_mean=110.0，completion_tokens_mean=256.0

### 4.3 BF16（Qwen/Qwen3-32B）

- 结果 JSON：`artifacts/qwen3_32b_vllm_bench_20251222/bench_bf16_1gpu_c16_r64_mt256.json`
- vLLM 日志：`artifacts/qwen3_32b_vllm_bench_20251222/vllm_bf16.log`

关键指标：
- QPS：1.2582
- prompt_tps：138.3967 tokens/s
- decode_tps：322.0868 tokens/s
- latency：p50=12.7105s，p95=12.7653s
- token_usage：prompt_tokens_mean=110.0，completion_tokens_mean=256.0

### 4.4 汇总对比（同口径：c=16 / r=64 / max_tokens=256）

| 精度 | 模型 | QPS | prompt_tps | decode_tps | p50 latency (s) | p95 latency (s) |
|---|---|---:|---:|---:|---:|---:|
| NVFP4 | RedHatAI/Qwen3-32B-NVFP4 | 2.3690 | 260.5918 | 606.4682 | 6.7577 | 6.7950 |
| FP8 | Qwen/Qwen3-32B-FP8 | 1.4917 | 164.0849 | 381.8702 | 10.6898 | 10.8637 |
| BF16 | Qwen/Qwen3-32B | 1.2582 | 138.3967 | 322.0868 | 12.7105 | 12.7653 |

![Latency-qwen-30B-3Precision](images/Latency-qwen-30B-3Precision.png)

![QPS-qwen-30B-3Precision](images/QPS-qwen-30B-3Precision.png)

![Throughput-qwen-30B-3Precision](images/Throughput-qwen-30B-3Precision.png)

## 5. 结论与下一步

- 在本次 1GPU（MIG 4g.96gb）与固定负载口径下：NVFP4 的吞吐最高，FP8 次之，BF16 最低。
- 由于 Qwen3-32B 相比 30B-A3B 更大，本次 p50 延迟明显更高；如要进一步压榨吞吐，可以尝试不同的 `concurrency/max-num-seqs/max-model-len` 组合，但需要保持口径一致再比较。
