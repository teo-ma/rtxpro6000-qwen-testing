# Qwen3-32B（BF16 / FP8 / NVFP4）LiveBench 准确度对比（vLLM + TP=2，2× RTX Pro 6000 Blackwell / MIG）

## 1. 测试目的

在同一台 Azure VM（Standard_NC256ds_xl_RTXPRO6000BSE_v6，2× RTX Pro 6000 Blackwell，MIG 4g.96gb×2，TP=2）上，使用 vLLM OpenAI-compatible API 作为推理后端，对比 Qwen3-32B 三种精度（BF16 / FP8 / NVFP4）在 LiveBench（release `2024-11-25`）上的分数。

## 2. 运行方式（概要）

- 推理：vLLM OpenAI server（`/v1/chat/completions`）
- 评测：LiveBench 官方脚本通过 OpenAI API 调用本地 vLLM（`http://127.0.0.1:8000/v1`）
- 执行脚本：`tools/run_tp2_livebench_3prec_api.sh`
- 本次 run 的原始产物与日志：`artifacts/qwen3_32b_vllm_tp2_livebench_20251225/`
  - `results/livebench_summary.json`
  - `logs/`（包含 vLLM server 日志与 livebench/judgment 日志）

## 3. 口径（分数如何计算）

LiveBench 的 `gen_ground_truth_judgment.py` 会为每道题产生一条 JSONL judgment 记录（包含 `model`、`category`、`score` 等字段）。

本报告的 **overall 分数** 定义为：

- 对每个模型，汇总全部 1000 道题的 `score`（已对 `(model, question_id, turn)` 去重），取平均值。

同时给出按 `category` 聚合的均值（及该类题量 N）。

> 注：这等价于对 LiveBench ground-truth judge 输出做简单平均。若 LiveBench 上游有其它加权/汇总规则，需要以其官方 leaderboard 脚本为准。

## 4. 结果

### 4.1 Overall（1000 题均分）

| 精度 | Overall mean（越高越好） |
|---|---:|
| BF16 | 0.278556 |
| FP8 | 0.268326 |
| NVFP4 | 0.279269 |

以 BF16 为基准（100%）的相对分数与差距：

| 精度 | BF16=100% 相对分数 | 相对 BF16 差距 |
|---|---:|---:|
| BF16 | 100.00% | +0.00pp |
| FP8 | 96.33% | -3.67pp |
| NVFP4 | 100.26% | +0.26pp |

![LiveBench relative score bar](images/qwen3_32b_tp2_livebench_relative_bar_20251225.svg)

结论（按本口径）：**NVFP4 ≈ BF16 > FP8**（NVFP4 与 BF16 差距非常小）。

### 4.2 分项（按 category 的均分）

题量分布：coding 128 / data_analysis 150 / instruction_following 200 / language 140 / math 232 / reasoning 150。

| category | BF16 mean | FP8 mean | NVFP4 mean |
|---|---:|---:|---:|
| coding | 0.054688 | 0.054688 | 0.039062 |
| data_analysis | 0.301133 | 0.317133 | 0.314467 |
| instruction_following | 0.787542 | 0.774125 | 0.823833 |
| language | 0.316579 | 0.312172 | 0.333575 |
| math | 0.088605 | 0.051088 | 0.053908 |
| reasoning | 0.026667 | 0.022500 | 0.020833 |

## 5. 工程备注（本次跑分修复点）

本次三精度推理（1000 题/精度）已完成；最初失败发生在 “ground truth judgment” 阶段，原因是：

- LiveBench 脚本在处理 `old_instruction_following` 分组时对 dict 做 `set()` 去重（`TypeError: unhashable type: 'dict'`）。
- LiveBench venv 缺少 `lxml`，导致 `pandas.read_html()` 解析 HTML 表格失败。

处理方式：

- 在 VM 上对 LiveBench 的 `gen_ground_truth_judgment.py` 做了最小补丁（按 JSON key 去重，保留原 dict）。
- 在 LiveBench venv 内安装 `lxml`。
- 之后仅重跑 judgment 阶段（不再重新生成答案），产出最终分数。

对应的汇总文件见：`artifacts/qwen3_32b_vllm_tp2_livebench_20251225/results/livebench_summary.json`。
