# Qwen/Qwen3-14B → NVFP4：量化全过程与三模型准确度对比（vLLM / llm-compressor）

本文将两部分合并为一份完整记录：

1) 在 Azure VM（1× NVIDIA RTX Pro 6000 Blackwell 96GB）上，将 Hugging Face 的 `Qwen/Qwen3-14B` 做 PTQ 并导出为 vLLM 可加载的 **NVFP4（compressed-tensors）** 模型
2) 使用 vLLM 的 `prompt_logprobs` 计算 token-level NLL / PPL，对 **BF16 vs 自制量化 NVFP4 vs NVIDIA NVFP4** 做代理准确度对比（含总对比表格与 Bar Chart）

---

## 1. 目标与产物

- 目标：把 `Qwen/Qwen3-14B` 量化为 NVFP4，并能被 vLLM 通过 `quantization=compressed-tensors` 稳定加载与生成
- 产物目录（自制量化 NVFP4）：`/mnt/data/models/Qwen3-14B-NVFP4`

---

## 2. 测试环境（Azure VM）

- GPU：NVIDIA RTX Pro 6000 Blackwell 96GB（MIG `4g.96gb`）
- Driver：`580.105.08`
- CUDA：`13.0`
- Python：`3.12.3`
- PyTorch：`2.9.0+cu128`
- vLLM：`0.13.0`（V1 engine；评测中 `enforce_eager=True`）
- compressed-tensors：`0.13.1.a20251215`

远端工作目录：`/mnt/data/nvfp4_work`

---

## 3. HF 缓存（强烈建议放到 /mnt/data）

根分区 `/` 容量很小，建议把 Hugging Face 模型与 datasets 缓存统一放到数据盘：

```bash
export HF_HOME=/mnt/data/hf_home
export HUGGINGFACE_HUB_CACHE=/mnt/data/hf_home/hub
export HF_DATASETS_CACHE=/mnt/data/hf_home/datasets
export XDG_CACHE_HOME=/mnt/data/hf_home/xdg_cache
export TMPDIR=/mnt/data/tmp

# 可选：下载加速
# export HF_HUB_ENABLE_HF_TRANSFER=1
```

若遇到权限问题（例如缓存目录被 root 创建），修复：

```bash
sudo chown -R azureuser:azureuser /mnt/data/hf_home
```

---

## 4. 量化工具链与脚本

- 量化：vllm-project 的 `llm-compressor`（oneshot PTQ 流程）
- 推理与评测：`vllm`
- 量化格式：`compressed-tensors`

关键脚本（远端）：`/mnt/data/nvfp4_work/qwen3_14b_nvfp4.py`

核心 recipe（概念）：
- `QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=["lm_head"])`
- 精度优化（可选）：`SpinQuantModifier`（Hadamard rotations）

---

## 5. 量化执行（PTQ + 导出）

本次校准数据：
- `DATASET_ID=HuggingFaceH4/ultrachat_200k`
- `DATASET_SPLIT=train_sft`

为提升精度恢复，增加校准样本数与序列长度：
- `NUM_CALIBRATION_SAMPLES=1024`
- `MAX_SEQUENCE_LENGTH=2048`

后台启动（nohup）：

```bash
nohup env \
  PYTHONUNBUFFERED=1 \
  HF_HOME=/mnt/data/hf_home \
  HUGGINGFACE_HUB_CACHE=/mnt/data/hf_home/hub \
  HF_DATASETS_CACHE=/mnt/data/hf_home/datasets \
  HF_HUB_ENABLE_HF_TRANSFER=1 \
  MODEL_ID=Qwen/Qwen3-14B \
  DATASET_ID=HuggingFaceH4/ultrachat_200k \
  DATASET_SPLIT=train_sft \
  NUM_CALIBRATION_SAMPLES=1024 \
  MAX_SEQUENCE_LENGTH=2048 \
  ENABLE_SPINQUANT=1 \
  /mnt/data/nvfp4_work/venv/bin/python -u /mnt/data/nvfp4_work/qwen3_14b_nvfp4.py \
  > /mnt/data/nvfp4_work/logs/qwen3_14b_nvfp4_YYYY-MM-DD_HHMMSS.log 2>&1 &
```

过程观察：

```bash
tail -n 120 /mnt/data/nvfp4_work/logs/qwen3_14b_nvfp4_*.log
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits
```

完成判定：日志出现类似
- `Saving compressed model to: /mnt/data/models/Qwen3-14B-NVFP4`
- `Done.`

---

## 6. 结果验证（vLLM 加载与生成）

验收脚本（远端）：`/mnt/data/nvfp4_work/post_quant_validate.sh`

```bash
/mnt/data/nvfp4_work/post_quant_validate.sh /mnt/data/nvfp4_work/logs/qwen3_14b_nvfp4_*.pid /mnt/data/models/Qwen3-14B-NVFP4
```

说明：短命脚本退出时，vLLM v1 多进程模式可能会打印 `EngineCore died unexpectedly` 之类清理噪声。
本次通过在退出前显式调用：

```python
llm.llm_engine.engine_core.shutdown()
```

实现干净退出。

---

## 7. 推理服务（长驻）

```bash
source /mnt/data/nvfp4_work/venv/bin/activate

vllm serve /mnt/data/models/Qwen3-14B-NVFP4 \
  --quantization compressed-tensors \
  --dtype auto \
  --tensor-parallel-size 1

# 更保守（禁用 compile/cudagraph）：
#   --enforce-eager
```

---

## 8. 三模型准确度对比（PPL 代理指标）

这里的“PPL 代理”指的是：不做问答/判题，而是用模型对给定文本的逐 token 预测概率来衡量“语言建模贴合度”。直观上，模型越确定下一个 token 应该是什么，PPL 越低；越不确定，PPL 越高。它不能替代标准基准的准确率，但很适合用来快速比较量化前后是否出现明显退化。

### 8.1 对比模型

- BF16 baseline：`Qwen/Qwen3-14B`
- 自制量化 NVFP4：`/mnt/data/models/Qwen3-14B-NVFP4`
- NVIDIA NVFP4：`/mnt/data/models/nvidia-Qwen3-14B-NVFP4`（HF：`nvidia/Qwen3-14B-NVFP4`）

### 8.2 数据集与采样规则

1) **WikiText-2**：`wikitext/wikitext-2-raw-v1` 的 `test` split
- 取前 `200` 条非空文本
- tokenizer 编码后：长度 < 16 tokens 的样本丢弃
- 超过 `max_model_len - 8` 的样本截断
- 最终进入评测：`kept_texts=132`

2) **UltraChat**：`HuggingFaceH4/ultrachat_200k` 的 `test_sft` split
- 取前 `200` 条
- 将 `messages` 拼成多行文本：每行 `role: content`
- 最终进入评测：`kept_texts=200`

### 8.3 指标定义（vLLM prompt_logprobs）

- 使用 `prompt_logprobs=1` 获取 prompt token 的 logprob
- 忽略第 1 个 token

$$
\text{NLL} = -\frac{1}{N}\sum_{i=1}^{N}\ell_i
\quad\quad
\text{PPL} = \exp(\text{NLL})
$$

### 8.4 结果（数值越低越好）

**WikiText-2（test）**

| 模型 | PPL | NLL | tokens |
|---|---:|---:|---:|
| BF16（Qwen/Qwen3-14B） | 1.687084 | 0.523002 | 23922 |
| 自制量化 NVFP4 | 1.709212 | 0.536032 | 23922 |
| NVIDIA NVFP4 | 1.739074 | 0.553353 | 23922 |

**UltraChat-200K（test_sft）**

| 模型 | PPL | NLL | tokens |
|---|---:|---:|---:|
| BF16（Qwen/Qwen3-14B） | 1.461068 | 0.379167 | 228323 |
| 自制量化 NVFP4 | 1.470275 | 0.385449 | 228323 |
| NVIDIA NVFP4 | 1.481108 | 0.392791 | 228323 |

### 8.5 三模型“总对比”（BF16=100%，PPL 代理分数）

这里将两套数据的 NLL 按 token 数加权汇总，再换算成 overall PPL，并将 BF16 归一化为 100%。

定义（数值越高越好）：

$$
\text{Score} = \frac{\text{PPL}_{\text{BF16,overall}}}{\text{PPL}_{\text{model,overall}}} \times 100
$$

| 模型 | Overall PPL | 相对分数（BF16=100） |
|---|---:|---:|
| BF16（Qwen/Qwen3-14B） | 1.481134 | 100.00 |
| 自制量化 NVFP4（有校准） | 1.491422 | 99.31 |
| NVIDIA NVFP4 | 1.503834 | 98.49 |

补充：自制量化 NVFP4（无校准）结果复用自无校准报告：Overall PPL=1.504343，Score=98.46。

![Qwen3-14B Relative Score（BF16=100，PPL proxy）](images/qwen3_14b_nvfp4_ppl_proxy_score.png)

---

## 9. 复现（三模型评测）

仓库脚本：`scripts/compare_3models_ppl.py`

结果文件：`results/compare_3models_results.json`

在远端运行：

```bash
source /mnt/data/nvfp4_work/venv/bin/activate

export HF_HOME=/mnt/data/hf_home
export HUGGINGFACE_HUB_CACHE=/mnt/data/hf_home/hub
export HF_DATASETS_CACHE=/mnt/data/hf_home/datasets
export XDG_CACHE_HOME=/mnt/data/hf_home/xdg_cache
export TMPDIR=/mnt/data/tmp

# 可选：调参
# export N_WIKITEXT=200
# export N_ULTRACHAT=200
# export BATCH_SIZE=16

python /mnt/data/nvfp4_work/compare_3models_ppl.py
```

---

## 10. 局限与改进方向

- 本评测为有限样本的 PPL 代理指标，不能替代标准基准（MMLU/GSM8K 等）。
- vLLM 加载自制量化 NVFP4 的 tokenizer 时曾出现“regex pattern”警告；如需更严格的对比，建议强制 3 个模型使用同一 tokenizer 配置/版本后复测。
- 若要更贴近真实推理场景，可增加：更长上下文、不同领域数据、以及生成任务的定性/定量评测。
