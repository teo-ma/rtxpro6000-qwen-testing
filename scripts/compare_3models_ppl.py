import os

# Reduce noisy logs (vLLM/tok)
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
os.environ.setdefault("VLLM_LOG_LEVEL", "ERROR")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import math
import time
from typing import Any, Dict, List

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
    # Local import keeps import surface small.
    from vllm import SamplingParams

    total_logprob = 0.0
    total_tokens = 0
    kept_texts = 0

    # max_tokens=1 means we only need prompt scoring, no long generations.
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
                # Prefer rank==0 (chosen token)
                for lp in entry.values():
                    if getattr(lp, "rank", None) == 0:
                        chosen_lp = lp
                        break

                # Fallback: max logprob among candidates
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


def main() -> None:
    datasets.logging.set_verbosity_error()

    max_model_len = int(os.environ.get("MAX_MODEL_LEN", "2048"))
    n_wiki = int(os.environ.get("N_WIKITEXT", "200"))
    n_ultra = int(os.environ.get("N_ULTRACHAT", "200"))
    batch_size = int(os.environ.get("BATCH_SIZE", "16"))
    out_json = os.environ.get("OUT_JSON", "/mnt/data/nvfp4_work/logs/compare_3models_results.json")

    texts = {
        "wikitext2_test": get_texts_wikitext(n_wiki),
        "ultrachat_test_sft": get_texts_ultrachat(n_ultra),
    }

    models: Dict[str, Dict[str, Any]] = {
        "bf16": {
            "model": "Qwen/Qwen3-14B",
            "dtype": "bfloat16",
        },
        "ours_nvfp4": {
            "model": "/mnt/data/models/Qwen3-14B-NVFP4",
            "quantization": "compressed-tensors",
            "dtype": "auto",
        },
        "nvidia_nvfp4": {
            "model": "/mnt/data/models/nvidia-Qwen3-14B-NVFP4",
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
        },
        "results": {},
    }

    from vllm import LLM

    for model_name, spec in models.items():
        print("\n== Loading model: {} ==".format(model_name), flush=True)
        t0 = time.time()
        llm = LLM(
            disable_log_stats=True,
            enforce_eager=True,
            max_model_len=max_model_len,
            tensor_parallel_size=1,
            **spec,
        )
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

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nWROTE_JSON {}".format(out_json), flush=True)


if __name__ == "__main__":
    main()
