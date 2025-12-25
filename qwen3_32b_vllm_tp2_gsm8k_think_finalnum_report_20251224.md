# Qwen3-32B TP=2（BF16 / FP8 / NVFP4）GSM8K 准确度对比（B 方法：think + final numeric）

## 1. 目的

在相同硬件与相同推理栈（vLLM + lm-eval，TP=2）下，对比 Qwen3-32B 三种精度（BF16 / FP8 / NVFP4）在 GSM8K 上的可判定答案准确度。

由于数学题常见“推理过程 + 最终答案”输出，本次采用 **B 方法**：允许模型输出 `<think>...</think>`，但要求 `</think>` 后用单行 `#### <number>` 给出最终答案；评分使用 lm-eval 内置 filter。

## 2. 环境与口径

- VM：Azure `Standard_NC256ds_xl_RTXPRO6000BSE_v6`
- GPU：2× RTX Pro 6000 Blackwell，MIG 强制开启（每卡 1× `4g.96gb`）
- vLLM：TP=2（跨两张 GPU/MIG），并启用稳定性参数：`enforce_eager=True`、`disable_custom_all_reduce=True`
- 生成：`temperature=0.0`、`max_gen_toks=512`
- 提示：开启 thinking（`enable_thinking=True`），并要求最终输出为纯数字

### 指标说明（lm-eval / GSM8K）

- `exact_match,flexible-extract`：更接近“只要最终数字对就算对”（本次作为**主对比指标**）
- `exact_match,strict-match`：更严格的格式匹配，能反映“最终答案输出是否干净/规范”

补充：GSM8K 的 `strict-match` 通常要求模型用 `#### <number>` 这类标准格式输出最终答案。若 system instruction 改成 “`</think>` 后只输出纯数字”，会与 `strict-match` 的格式期望冲突，导致大量 `[invalid]`。因此如需解读/对齐 `strict-match`，应统一要求输出 `#### <number>`。

## 3. 结果

### 3.1 GSM8K（limit=50）

| Precision | exact_match,flexible-extract | exact_match,strict-match |
|---|---:|---:|
| BF16 | 0.76 ± 0.0610 | 0.58 ± 0.0705 |
| FP8  | 0.70 ± 0.0655 | 0.52 ± 0.0714 |
| NVFP4 | 0.84 ± 0.0524 | 0.36 ± 0.0686 |

- 原始结果：
  - `artifacts/qwen3_32b_vllm_tp2_gsm8k_limit50_think_finalnum_20251224/`

### 3.2 GSM8K（limit=200）

| Precision | exact_match,flexible-extract | exact_match,strict-match |
|---|---:|---:|
| BF16 | 0.755 ± 0.0305 | 0.585 ± 0.0349 |
| FP8  | 0.750 ± 0.0307 | 0.590 ± 0.0349 |
| NVFP4 | 0.780 ± 0.0294 | 0.375 ± 0.0343 |

- 原始结果：
  - `artifacts/qwen3_32b_vllm_tp2_gsm8k_limit200_think_finalnum_20251224/`

## 4. 结论（仅基于本次口径）

- 在 **flexible-extract（主指标）** 下：NVFP4 ≥ BF16 ≈ FP8（limit=50 与 limit=200 都呈现一致趋势）。
- 在 **strict-match** 下：NVFP4 显著更低，主要原因是它更常按指令输出“纯数字”（缺少 `####` 前缀）从而被 strict 解析判为 `[invalid]`；这更像是“格式/提取友好性”差异。

## 5. 复现

- 脚本：`tools/run_tp2_gsm8k_limit50_3prec_think_finalnum.sh`
- 用法示例：

```bash
# 默认 limit=50
bash tools/run_tp2_gsm8k_limit50_3prec_think_finalnum.sh

# rerun：limit=200
LIMIT=200 bash tools/run_tp2_gsm8k_limit50_3prec_think_finalnum.sh
```
