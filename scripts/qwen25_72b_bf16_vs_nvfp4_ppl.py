import os

# Reduce noisy logs (vLLM/tok)
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
os.environ.setdefault("VLLM_LOG_LEVEL", "ERROR")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import math
import time
from typing import Any, Dict, List, Optional

import datasets
from datasets import load_dataset


def format_ultrachat_messages(messages: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for message in messages:
        role = (message.get("role") or "").strip()
        content = (message.get("content") or "").strip()
        if not content:
            continue
        lines.append("{}: {}".format(role, content))
    return "\n".join(lines)


def get_texts_wikitext(n: int) -> List[str]:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [(x.get("text") or "").strip() for x in dataset]
    texts = [t for t in texts if t]
    return texts[:n]


def get_texts_ultrachat(n: int) -> List[str]:
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
    texts: List[str] = []
    for row in dataset:
        messages = row.get("messages")
        if not messages:
            continue
        text = format_ultrachat_messages(messages)
        if text:
            texts.append(text)
        if len(texts) >= n:
            break
    return texts


def build_prompts(tokenizer, texts: List[str], max_model_len: int) -> List[str]:
    prompts: List[str] = []
    for text in texts:
        text = (text or "").strip()
        if not text:
            continue
        token_ids = tokenizer.encode(text)
        if len(token_ids) < 16:
            continue
        if len(token_ids) > (max_model_len - 8):
            token_ids = token_ids[: (max_model_len - 8)]
            text = tokenizer.decode(token_ids)
        prompts.append(text)
    return prompts


def compute_nll_ppl(llm, prompts: List[str], batch_size: int) -> Dict[str, Any]:
    from vllm import SamplingParams

    total_logprob = 0.0
    total_tokens = 0
    kept_texts = 0

    sp = SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=1)

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        outputs = llm.generate(batch, sp)
        for out in outputs:
            prompt_logprobs = out.prompt_logprobs
            if not prompt_logprobs:
                continue

            # Skip first token (no previous context).
            for entry in prompt_logprobs[1:]:
                if entry is None:
                    continue

                chosen_lp = None
                for lp in entry.values():
                    if getattr(lp, "rank", None) == 0:
                        chosen_lp = lp
                        break

                if chosen_lp is None:
                    chosen_lp = max(entry.values(), key=lambda x: x.logprob)

                total_logprob += float(chosen_lp.logprob)
                total_tokens += 1

            kept_texts += 1

    if total_tokens == 0:
        return {"nll": float("nan"), "ppl": float("nan"), "tokens": 0, "kept_texts": kept_texts}

    nll = -total_logprob / total_tokens
    ppl = math.exp(nll)
    return {"nll": nll, "ppl": ppl, "tokens": total_tokens, "kept_texts": kept_texts}


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return int(v)


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return float(v)


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v


def _optional_kwarg(env_name: str, cast_fn, target_key: str) -> Dict[str, Any]:
    raw = os.environ.get(env_name)
    if raw is None or raw == "":
        return {}
    return {target_key: cast_fn(raw)}


def _overall_from_datasets(ds_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    # Token-weighted NLL across datasets.
    total_tokens = 0
    total_nll_times_tokens = 0.0
    for m in ds_metrics.values():
        tokens = int(m.get("tokens") or 0)
        nll = float(m.get("nll") or 0.0)
        if tokens <= 0:
            continue
        total_tokens += tokens
        total_nll_times_tokens += nll * tokens

    if total_tokens <= 0:
        return {"nll": float("nan"), "ppl": float("nan"), "tokens": 0}

    nll = total_nll_times_tokens / total_tokens
    ppl = math.exp(nll)
    return {"nll": nll, "ppl": ppl, "tokens": total_tokens}


def main() -> None:
    datasets.logging.set_verbosity_error()

    max_model_len = _env_int("MAX_MODEL_LEN", 2048)
    n_wiki = _env_int("N_WIKITEXT", 50)
    n_ultra = _env_int("N_ULTRACHAT", 50)
    batch_size = _env_int("BATCH_SIZE", 1)

    bf16_model_id = _env_str("BF16_MODEL", "Qwen/Qwen2.5-72B-Instruct")
    nvfp4_model_path = _env_str("NVFP4_MODEL", "/mnt/data/models/Qwen2.5-72B-Instruct-NVFP4")

    out_json = _env_str("OUT_JSON", "/mnt/data/nvfp4_work/logs/qwen25_72b_bf16_vs_nvfp4_ppl_results.json")

    # vLLM knobs for BF16 baseline (large model) to try to fit on 96GB MIG.
    bf16_cpu_offload_gb = _env_int("BF16_CPU_OFFLOAD_GB", 0)
    bf16_gpu_mem_util = _env_float("BF16_GPU_MEMORY_UTILIZATION", 0.90)

    texts = {
        "wikitext2_test": get_texts_wikitext(n_wiki),
        "ultrachat_test_sft": get_texts_ultrachat(n_ultra),
    }

    models: Dict[str, Dict[str, Any]] = {
        "bf16": {
            "model": bf16_model_id,
            "dtype": "bfloat16",
        },
        "nvfp4": {
            "model": nvfp4_model_path,
            "quantization": "compressed-tensors",
            "dtype": "auto",
        },
    }

    results: Dict[str, Any] = {
        "config": {
            "max_model_len": max_model_len,
            "n_wikitext": n_wiki,
            "n_ultrachat": n_ultra,
            "batch_size": batch_size,
            "bf16_model": bf16_model_id,
            "nvfp4_model": nvfp4_model_path,
            "bf16_cpu_offload_gb": bf16_cpu_offload_gb,
            "bf16_gpu_memory_utilization": bf16_gpu_mem_util,
        },
        "results": {},
    }

    from vllm import LLM

    for model_name, spec in models.items():
        print("\n== Loading model: {} ==".format(model_name), flush=True)

        llm_kwargs: Dict[str, Any] = dict(
            disable_log_stats=True,
            enforce_eager=True,
            max_model_len=max_model_len,
            tensor_parallel_size=1,
            **spec,
        )

        if model_name == "bf16":
            if bf16_cpu_offload_gb > 0:
                llm_kwargs["cpu_offload_gb"] = bf16_cpu_offload_gb
            llm_kwargs["gpu_memory_utilization"] = bf16_gpu_mem_util

            # Optional overrides for BF16 only.
            llm_kwargs.update(_optional_kwarg("BF16_SWAP_SPACE_GB", int, "swap_space"))
            llm_kwargs.update(_optional_kwarg("BF16_MAX_NUM_SEQS", int, "max_num_seqs"))
            llm_kwargs.update(_optional_kwarg("BF16_MAX_NUM_BATCHED_TOKENS", int, "max_num_batched_tokens"))

        t0 = time.time()
        llm = LLM(**llm_kwargs)
        load_s = time.time() - t0

        tokenizer = llm.get_tokenizer()
        results["results"][model_name] = {"load_s": load_s, "datasets": {}}

        for dataset_name, dataset_texts in texts.items():
            prompts = build_prompts(tokenizer, dataset_texts, max_model_len)
            t1 = time.time()
            metrics = compute_nll_ppl(llm, prompts, batch_size)
            metrics["eval_s"] = time.time() - t1
            results["results"][model_name]["datasets"][dataset_name] = metrics

            print(
                "{}\t{}\tppl={:.6f}\tnll={:.6f}\ttokens={}\tkept_texts={}\teval_s={:.2f}".format(
                    model_name,
                    dataset_name,
                    metrics["ppl"],
                    metrics["nll"],
                    metrics["tokens"],
                    metrics["kept_texts"],
                    metrics["eval_s"],
                ),
                flush=True,
            )

        # Avoid EngineCore shutdown noise for short-lived scripts.
        llm.llm_engine.engine_core.shutdown()
        del llm

    # Overall (token-weighted) + relative score (BF16=100)
    for model_name in list(results["results"].keys()):
        ds_metrics = results["results"][model_name]["datasets"]
        results["results"][model_name]["overall"] = _overall_from_datasets(ds_metrics)

    bf16_overall_ppl = float(results["results"].get("bf16", {}).get("overall", {}).get("ppl", float("nan")))
    for model_name in list(results["results"].keys()):
        ppl = float(results["results"][model_name]["overall"]["ppl"])
        score = (bf16_overall_ppl / ppl * 100.0) if (ppl > 0 and not math.isnan(bf16_overall_ppl)) else float("nan")
        results["results"][model_name]["overall"]["score_bf16_100"] = score

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nWROTE_JSON {}".format(out_json), flush=True)


if __name__ == "__main__":
    main()
