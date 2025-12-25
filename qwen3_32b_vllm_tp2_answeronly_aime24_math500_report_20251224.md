# Qwen3-32B vLLM TP=2（answer-only）AIME24 + MATH500 报告（2025-12-24）

## 设置
- 推理引擎：vLLM（lm-evaluation-harness `--model vllm`）
- 并行：2×GPU，`tensor_parallel_size=2`
- 生成：`temperature=0.0`，`max_gen_toks=256`
- 上下文：`max_model_len=4096`，`max_num_seqs=8`
- 口径：`--apply_chat_template` + `--system_instruction`（只输出最终答案），`enable_thinking=False`
- 稳定性：`enforce_eager=True`，`disable_custom_all_reduce=True`
- 采样规模：
  - `aime24`：`--limit 30`
  - `hendrycks_math500`：`--limit 100`

> 说明：`--limit` 仅用于快速抽样对比，不代表全量 benchmark 指标。

## 模型
- BF16：`Qwen/Qwen3-32B`
- FP8：`Qwen/Qwen3-32B-FP8`
- NVFP4：`RedHatAI/Qwen3-32B-NVFP4`

## 结果（exact_match）

| Precision | AIME24 (n=30) | MATH500 (n=100) |
|---|---:|---:|
| BF16 | 0.00 ± 0.00 | 0.27 ± 0.0446 |
| FP8  | 0.00 ± 0.00 | 0.25 ± 0.0435 |
| NVFP4 | 0.00 ± 0.00 | 0.28 ± 0.0451 |

## 产物
- 结果 JSON / samples JSONL：见 [artifacts/qwen3_32b_vllm_tp2_answeronly_20251224/](artifacts/qwen3_32b_vllm_tp2_answeronly_20251224/)

## AIME24 复测：更合理的评判口径（integer_match）

### 背景
`aime24` 的标准答案是一个十进制整数（AIME 口径），但模型输出经常包含 LaTeX、分数或 `<think>` 片段；这会导致 **strict exact_match** 严重低估。

本仓库新增了一个更宽松但更贴近 AIME 评分口径的指标：
- `integer_match`：从模型输出中提取“最终整数”（例如最后一个整数 token），再与标准答案整数比较。

脚本位置：
- [tools/score_lmeval_aime24_integer_match.py](tools/score_lmeval_aime24_integer_match.py)

### 复测 1：int-only（尝试 fewshot 参数，但任务配置强制 0-shot）
- 目标：强制模型只输出单个整数，减少格式干扰。
- 备注：lm-eval 内置 `aime24.yaml` 含 `num_fewshot: 0`，因此 `--num_fewshot 5` 会被忽略（日志里会提示）。

结果（n=30）：

| Precision | exact_match | integer_match |
|---|---:|---:|
| BF16 | 0.00 | 0.00 (0/30) |
| FP8  | 0.00 | 0.00 (0/30) |
| NVFP4 | 0.00 | 0.00 (0/30) |

产物：
- [artifacts/qwen3_32b_vllm_tp2_aime24_intonly_fewshotflag_20251224/](artifacts/qwen3_32b_vllm_tp2_aime24_intonly_fewshotflag_20251224/)

### 复测 2：打开 thinking（允许输出 `<think>`，再用 integer_match 抽取最终整数）
- 目标：提高推理正确率（代价是输出会包含 `<think>`，strict exact_match 基本不可用）。

结果（BF16，n=30）：

| Precision | exact_match | integer_match |
|---|---:|---:|
| BF16 | 0.00 | 0.0333 (1/30) |

产物：
- [artifacts/qwen3_32b_vllm_tp2_aime24_intonly_think1_20251224/](artifacts/qwen3_32b_vllm_tp2_aime24_intonly_think1_20251224/)

### 复测 3：B 方法（think_finalint_v1）：thinking 打开 + 强制 `</think>` 后输出单个整数
- 目标：保留模型推理链路（`enable_thinking=True`），但把“可评分的最终答案”固定到 `</think>` 之后。
- 评分口径：`integer_match[prefer_final]`（优先解析 `</think>` 之后的文本；失败再 fallback 到去 think / raw）。

结果（n=30）：

| Precision | exact_match | integer_match[prefer_final] |
|---|---:|---:|
| BF16 | 0.00 | 0.0000 (0/30) |
| FP8  | 0.00 | 0.0333 (1/30) |
| NVFP4 | 0.00 | 0.0333 (1/30) |

产物：
- [artifacts/qwen3_32b_vllm_tp2_aime24_think_finalint_20251224/](artifacts/qwen3_32b_vllm_tp2_aime24_think_finalint_20251224/)
