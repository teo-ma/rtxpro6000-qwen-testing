# Qwen/Qwen3-14B → NVFP4（无校准 / 无数据校准）：环境与量化步骤

目标：参考 [qwen3_14b_nvfp4_quant_and_eval.md](qwen3_14b_nvfp4_quant_and_eval.md) 的流程，将 `Qwen/Qwen3-14B`（BF16）量化为 **NVFP4（compressed-tensors）**，但**不使用任何真实数据进行校准**。

> 重要说明：NVFP4 通常需要少量校准样本来估计“全局 activation scale”。
> 无校准（`NUM_CALIBRATION_SAMPLES=0`）会跳过这一步，**准确度可能明显退化**。
> 本文的目标是得到一个“完全不依赖校准数据”的可加载模型，用于对比/实验。

---

## 1. VM 登录

```bash
ssh -i ~/.ssh/azure_id_rsa azureuser@20.112.150.26
```

本 VM 上数据盘为 `/data`（不是 `/mnt/data`），因此本文默认使用 `/data/...`。

远端工作目录建议：`/data/nvfp4_work`

---

## 2. HF 缓存位置（建议）

避免写满根分区 `/`：

```bash
export HF_HOME=/data/hf_home
export HUGGINGFACE_HUB_CACHE=/data/hf_home/hub
export HF_DATASETS_CACHE=/data/hf_home/datasets
export XDG_CACHE_HOME=/data/hf_home/xdg_cache
export TMPDIR=/data/tmp

# 可选：下载加速
# export HF_HUB_ENABLE_HF_TRANSFER=1
```

---

## 3. 将无校准量化脚本上传到 VM

本仓库提供了一个脚本：
- [scripts/qwen3_14b_nvfp4_nocalib.py](scripts/qwen3_14b_nvfp4_nocalib.py)

在本机执行（从仓库根目录）：

```bash
scp -i ~/.ssh/azure_id_rsa \
  scripts/qwen3_14b_nvfp4_nocalib.py \
  azureuser@20.112.150.26:/data/nvfp4_work/
```

---

## 4. 依赖（一次性）

如果你已经按 [qwen3_14b_nvfp4_quant_and_eval.md](qwen3_14b_nvfp4_quant_and_eval.md) 配过 venv，可跳过。

建议在 VM 上使用 venv：

```bash
sudo mkdir -p /data/nvfp4_work
sudo chown -R azureuser:azureuser /data/nvfp4_work
cd /data/nvfp4_work

python3 -m venv venv
source venv/bin/activate

# 关键依赖（版本以你现有环境为准；下面仅给出最小集合）
python -m pip install -U pip
python -m pip install -U "transformers" "datasets" "accelerate"

# LLM Compressor（建议用官方仓库 editable 安装以匹配 vLLM）
# git clone https://github.com/vllm-project/llm-compressor.git
# cd llm-compressor && pip install -e .
```

---

## 5. 执行无校准 NVFP4 量化

建议把输出目录与“有校准版本”区分开，避免覆盖：
- 无校准输出：`/data/models/Qwen3-14B-NVFP4-NO-CALIB`

在 VM 上执行（后台 nohup）：

```bash
mkdir -p /data/nvfp4_work/logs
mkdir -p /data/models

nohup env \
  PYTHONUNBUFFERED=1 \
  HF_HOME=/data/hf_home \
  HUGGINGFACE_HUB_CACHE=/data/hf_home/hub \
  HF_DATASETS_CACHE=/data/hf_home/datasets \
  XDG_CACHE_HOME=/data/hf_home/xdg_cache \
  TMPDIR=/data/tmp \
  MODEL_ID=Qwen/Qwen3-14B \
  OUT_DIR=/data/models/Qwen3-14B-NVFP4-NO-CALIB \
  MAX_SEQUENCE_LENGTH=2048 \
  NUM_CALIBRATION_SAMPLES=0 \
  ENABLE_SPINQUANT=0 \
  /data/nvfp4_work/venv/bin/python -u /data/nvfp4_work/qwen3_14b_nvfp4_nocalib.py \
  > /data/nvfp4_work/logs/qwen3_14b_nvfp4_nocalib_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

观察进度：

```bash
tail -n 120 /data/nvfp4_work/logs/qwen3_14b_nvfp4_nocalib_*.log
nvidia-smi
```

完成判定（日志中会出现类似）：
- `saving compressed model to: /mnt/data/models/Qwen3-14B-NVFP4-NO-CALIB`
- `done.`

---

## 6. 验收：vLLM 加载与生成

```bash
source /data/nvfp4_work/venv/bin/activate

vllm serve /data/models/Qwen3-14B-NVFP4-NO-CALIB \
  --quantization compressed-tensors \
  --dtype auto \
  --tensor-parallel-size 1

# 更保守：
#   --enforce-eager
```

---

## 7. 常见问题

1) `NUM_CALIBRATION_SAMPLES=0` 仍然下载了数据集？
- 本脚本在无校准模式下不会 `load_dataset()`，只用一个内存占位 dataset 来满足接口要求。

2) 无校准结果和 BF16 / 有校准 NVFP4 差距很大？
- 这是预期现象之一。NVFP4 的 activation global scale 没有经过校准，误差会放大。

3) 想试试 data-free pipeline？
- 可以设置 `PIPELINE=datafree`（是否对 NVFP4 生效取决于 llm-compressor 版本/实现）。

---

## 8. 准确度对比（四模型，PPL 代理）

本节对齐 [qwen3_14b_nvfp4_quant_and_eval.md](qwen3_14b_nvfp4_quant_and_eval.md) 的口径：使用 vLLM `prompt_logprobs` 计算 token-level NLL / PPL。

说明：
- **本次仅新增评测无校准模型**；其余 3 个模型的数值直接复用之前的结果（不重复跑）。
- 无校准模型最初在本口径下出现 **`prompt_logprobs/logprobs` 全量为 NaN**，定位到量化导出里大量 `*.input_global_scale` 张量为 `NaN/Inf`，导致 logits 数值污染。
  - 为了能完成“同口径”评测，本次在**不重新量化**的前提下，对这些 `input_global_scale` 做了一个最小补丁（将其统一设为 `1.0`），使 vLLM 能输出正常 logprobs。
  - 注意：这不是标准意义上的“无数据校准”，更像是“导出产物的数值修复/兜底”，因此该分数仅用于对比与排障参考。

### 8.1 对比模型

- BF16 baseline：`Qwen/Qwen3-14B`
- 自制量化 NVFP4（有校准）：`/mnt/data/models/Qwen3-14B-NVFP4`
- NVIDIA NVFP4：`/mnt/data/models/nvidia-Qwen3-14B-NVFP4`（HF：`nvidia/Qwen3-14B-NVFP4`）
- 自制量化 NVFP4（无校准）：`/data/models/Qwen3-14B-NVFP4-NO-CALIB`

### 8.2 数据集与采样规则（同上一份文档）

1) **WikiText-2**：`wikitext/wikitext-2-raw-v1` 的 `test` split
- 取前 `200` 条非空文本
- tokenizer 编码后：长度 < 16 tokens 的样本丢弃
- 超过 `max_model_len - 8` 的样本截断
- 最终进入评测：`kept_texts=132`

2) **UltraChat**：`HuggingFaceH4/ultrachat_200k` 的 `test_sft` split
- 取前 `200` 条
- 将 `messages` 拼成多行文本：每行 `role: content`
- 最终进入评测：`kept_texts=200`

### 8.3 结果（数值越低越好）

**WikiText-2（test）**

| 模型 | PPL | NLL | tokens |
|---|---:|---:|---:|
| BF16（Qwen/Qwen3-14B） | 1.687084 | 0.523002 | 23922 |
| 自制量化 NVFP4（有校准） | 1.709212 | 0.536032 | 23922 |
| NVIDIA NVFP4 | 1.739074 | 0.553353 | 23922 |
| 自制量化 NVFP4（无校准） | 1.706852 | 0.534650 | 23922 |

**UltraChat-200K（test_sft）**

| 模型 | PPL | NLL | tokens |
|---|---:|---:|---:|
| BF16（Qwen/Qwen3-14B） | 1.461068 | 0.379167 | 228323 |
| 自制量化 NVFP4（有校准） | 1.470275 | 0.385449 | 228323 |
| NVIDIA NVFP4 | 1.481108 | 0.392791 | 228323 |
| 自制量化 NVFP4（无校准） | 1.484568 | 0.395124 | 228323 |

### 8.4 四模型“总对比”（BF16=100%，PPL 代理分数）

复用上一份文档的定义：两套数据的 NLL 按 token 数加权汇总，再换算 overall PPL，并将 BF16 归一化为 100%。

$$
	ext{Score} = \frac{\text{PPL}_{\text{BF16,overall}}}{\text{PPL}_{\text{model,overall}}} \times 100
$$

| 模型 | Overall PPL | 相对分数（BF16=100） |
|---|---:|---:|
| BF16（Qwen/Qwen3-14B） | 1.481134 | 100.00 |
| 自制量化 NVFP4（有校准） | 1.491422 | 99.31 |
| NVIDIA NVFP4 | 1.503834 | 98.49 |
| 自制量化 NVFP4（无校准） | 1.504343 | 98.46 |

![Qwen3-14B Relative Score（BF16=100，PPL proxy）](images/qwen3_14b_nvfp4_ppl_proxy_score.png)

### 8.5 复现（仅无校准评测）

在 VM 上：

```bash
source /data/nvfp4_work/venv/bin/activate

export HF_HOME=/data/hf_home
export HUGGINGFACE_HUB_CACHE=/data/hf_home/hub
export HF_DATASETS_CACHE=/data/hf_home/datasets
export XDG_CACHE_HOME=/data/hf_home/xdg_cache
export TMPDIR=/data/tmp

python -u /data/nvfp4_work/compare_nocalib_ppl.py
```

产物：
- JSON：`/data/nvfp4_work/logs/compare_nocalib_results.json`
- 日志（本次运行示例）：`/data/nvfp4_work/logs/compare_nocalib_20251225_130245_patched_igs.log`

无校准 NaN 修复（仅用于让评测可运行）：
- 在模型目录下新增补丁权重文件：`/data/models/Qwen3-14B-NVFP4-NO-CALIB/patched_input_global_scale_20251225_130010.safetensors`
- 并更新 `model.safetensors.index.json` 的 `weight_map`，把所有 `*.input_global_scale` 指向该补丁文件（原 index 已备份为 `model.safetensors.index.json.bak_patch_igs_20251225_130010`）。
