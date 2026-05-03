#!/usr/bin/env python3
"""RQ2 Exp3 quantization experiment for local Transformers alert classification."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


VALID_LABELS = ("Attack", "Non-Attack")


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON list")

    out: list[dict[str, Any]] = []
    for i, rec in enumerate(data):
        if not isinstance(rec, dict):
            raise ValueError(f"{path} record[{i}] is not an object")
        out.append(rec)
    return out


def safe_model_name(model: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", model.strip())
    return name or "model"


def build_user_content(record: dict[str, Any]) -> str:
    fields = dict(record)
    return "Alert fields (JSON):\n" + json.dumps(fields, ensure_ascii=False, indent=2)


def parse_label(content: str | None) -> str | None:
    if content is None:
        return None
    text = str(content).strip()
    if text in VALID_LABELS:
        return text
    if "Non-Attack" in text:
        return "Non-Attack"
    if "Attack" in text:
        return "Attack"
    return None


def atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def compute_binary_metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> dict[str, Any]:
    valid = tp + fp + tn + fn
    accuracy = safe_div(tp + tn, valid)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    fpr = safe_div(fp, fp + tn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "positive_class": "Attack",
        "valid_eval_count": valid,
        "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall_tpr": round(recall, 6),
        "fpr": round(fpr, 6),
        "f1_score": round(f1, 6),
    }


def build_summary(
    args: argparse.Namespace,
    total_jobs: int,
    completed: int,
    results: list[dict[str, Any]],
    failed: list[dict[str, Any]],
    attack_true_stats: dict[str, int],
    non_attack_true_stats: dict[str, int],
    tp: int,
    fp: int,
    tn: int,
    fn: int,
    status: str,
) -> dict[str, Any]:
    attack_output = sum(1 for x in results if x.get("llm_label") == "Attack")
    non_attack_output = sum(1 for x in results if x.get("llm_label") == "Non-Attack")
    return {
        "model": args.model_path,
        "quantization": args.quantization,
        "experiment": "RQ2_Exp3_quantization_model_comparison",
        "total": total_jobs,
        "completed": completed,
        "success": len(results) - len(failed),
        "failed": len(failed),
        "sample_per_class": args.sample_per_class,
        "attack_samples": args.sample_per_class,
        "non_attack_samples": args.sample_per_class,
        "retry_times": args.retry_times,
        "attack_output": attack_output,
        "non_attack_output": non_attack_output,
        "by_true_class": {
            "Attack": attack_true_stats,
            "Non-Attack": non_attack_true_stats,
        },
        "metrics": {
            **compute_binary_metrics_from_counts(tp, fp, tn, fn),
            "strict_accuracy_overall": round(safe_div(tp + tn, completed), 6),
        },
        "status": status,
    }


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def resolve_dtype(dtype: str, device: str) -> torch.dtype | None:
    if dtype == "auto":
        if device == "cuda":
            return torch.float16
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def build_quantization_config(quantization: str, device: str) -> Any | None:
    if quantization == "none":
        return None
    if device != "cuda":
        raise ValueError("bitsandbytes quantization requires CUDA device.")
    try:
        from transformers import BitsAndBytesConfig
    except Exception as e:
        raise RuntimeError(
            "BitsAndBytesConfig is unavailable. Please install bitsandbytes-compatible packages."
        ) from e
    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=16.0)
    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    raise ValueError(f"Unsupported quantization: {quantization}")


def build_prompt(tokenizer: AutoTokenizer, system_prompt: str, user_content: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return f"System: {system_prompt}\nUser: {user_content}\nAssistant:"


def generate_once(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    system_prompt: str,
    user_content: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> tuple[str | None, str | None]:
    prompt = build_prompt(tokenizer, system_prompt, user_content)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    new_ids = outputs[:, prompt_len:]
    text = tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()
    parsed = parse_label(text)
    return text, parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ2 Exp3 local Transformers quantization evaluation")
    parser.add_argument("--model", "--model-path", dest="model_path", default="", help="Model name or local/Hugging Face model path")
    parser.add_argument("--attack-data", default="", help="Path to the Attack JSON file")
    parser.add_argument("--fp-data", default="", help="Path to the Non-Attack JSON file")
    parser.add_argument("--sample-per-class", type=int, default=250)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda", help="auto/cpu/cuda")
    parser.add_argument("--dtype", default="auto", help="auto/float16/bfloat16/float32")
    parser.add_argument("--quantization", choices=["none", "8bit", "4bit"], default="8bit", help="bitsandbytes quantization mode")
    parser.add_argument("--timeout", type=int, default=0, help="Reserved for compatibility; not used in local mode")
    parser.add_argument("--retry-times", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling generation")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--out-dir", default="", help="Directory for quantization experiment result files")
    parser.add_argument("--log-dir", default="", help="Directory for failed-case logs")
    args = parser.parse_args()

    required_args = {
        "--model": args.model_path,
        "--attack-data": args.attack_data,
        "--fp-data": args.fp_data,
        "--out-dir": args.out_dir,
        "--log-dir": args.log_dir,
    }
    missing = [name for name, value in required_args.items() if not value]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")

    device = resolve_device(args.device)
    torch_dtype = resolve_dtype(args.dtype, device)
    quantization_config = build_quantization_config(args.quantization, device)

    attack_records = load_json_list(Path(args.attack_data))
    fp_records = load_json_list(Path(args.fp_data))
    if len(attack_records) < args.sample_per_class:
        raise ValueError(f"attack records not enough: {len(attack_records)} < {args.sample_per_class}")
    if len(fp_records) < args.sample_per_class:
        raise ValueError(f"fp records not enough: {len(fp_records)} < {args.sample_per_class}")

    rng = random.Random(args.seed)
    sampled_attack = rng.sample(attack_records, args.sample_per_class)
    sampled_fp = rng.sample(fp_records, args.sample_per_class)

    tasks: list[dict[str, Any]] = []
    uid = 1
    for rec in sampled_attack:
        tasks.append({"id": uid, "source": "attack_clean", "true_label": "Attack", "record": rec})
        uid += 1
    for rec in sampled_fp:
        tasks.append({"id": uid, "source": "fp_clean", "true_label": "Non-Attack", "record": rec})
        uid += 1
    rng.shuffle(tasks)

    system_prompt = (
"""
<task description>
You will be given one network alert record and your task is to classify the alert into one of two labels:
- Attack: The alert indicates a real malicious activity, an attack attempt, or other security-threatening actions.
- Non-Attack: The alert does not indicate a real attack. It may correspond to benign traffic, normal operations, false positives, or informational events.
<constraints>
Do not output explanations or any additional text. Return only "Attack" or "Non-Attack"./no_think
"""
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
    }
    if quantization_config is not None:
        model_load_kwargs["quantization_config"] = quantization_config
        model_load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_load_kwargs)
    if quantization_config is None:
        model.to(device)
    model.eval()

    out_dir = Path(args.out_dir)
    log_dir = Path(args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    model_file = f"{safe_model_name(args.model_path)}__quant_{args.quantization}.json"
    result_path = out_dir / model_file
    failed_path = log_dir / f"{safe_model_name(args.model_path)}__quant_{args.quantization}_failed.json"

    results: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    completed = 0
    total_jobs = len(tasks)
    max_attempts = max(1, args.retry_times + 1)

    attack_true_stats = {"llm_attack": 0, "llm_non_attack": 0, "llm_other": 0, "failed": 0}
    non_attack_true_stats = {"llm_attack": 0, "llm_non_attack": 0, "llm_other": 0, "failed": 0}
    tp = fp = tn = fn = 0

    iterable = tasks
    if tqdm is not None:
        iterable = tqdm(tasks, total=total_jobs, desc=f"RQ2 Quantization Evaluating ({args.quantization})", unit="alert")

    for sample in iterable:
        attempts = []
        llm_label = None
        llm_output_raw = None
        ok = False

        for i in range(1, max_attempts + 1):
            try:
                raw_out, parsed = generate_once(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    system_prompt=system_prompt,
                    user_content=build_user_content(sample["record"]),
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                attempts.append(
                    {
                        "attempt": i,
                        "model_output": raw_out,
                        "parsed_label": parsed,
                        "error": None,
                    }
                )
                llm_output_raw = raw_out
                llm_label = parsed
                if parsed in VALID_LABELS:
                    ok = True
                    break
            except Exception as e:
                attempts.append(
                    {
                        "attempt": i,
                        "model_output": None,
                        "parsed_label": None,
                        "error": f"generate_error: {e}",
                    }
                )

        record = {
            "id": sample["id"],
            "source": sample["source"],
            "true_label": sample["true_label"],
            "llm_label": llm_label if ok else None,
            "llm_output_raw": llm_output_raw,
            "record": sample["record"],
            "ok": ok,
            "attempts": attempts,
        }
        results.append(record)
        completed += 1

        cls_stats = attack_true_stats if sample["true_label"] == "Attack" else non_attack_true_stats
        if not ok:
            failed.append(record)
            cls_stats["failed"] += 1
        else:
            if record["llm_label"] == "Attack":
                cls_stats["llm_attack"] += 1
            elif record["llm_label"] == "Non-Attack":
                cls_stats["llm_non_attack"] += 1
            else:
                cls_stats["llm_other"] += 1

            if sample["true_label"] == "Attack":
                if record["llm_label"] == "Attack":
                    tp += 1
                elif record["llm_label"] == "Non-Attack":
                    fn += 1
            else:
                if record["llm_label"] == "Attack":
                    fp += 1
                elif record["llm_label"] == "Non-Attack":
                    tn += 1

        summary_partial = build_summary(
            args=args,
            total_jobs=total_jobs,
            completed=completed,
            results=results,
            failed=failed,
            attack_true_stats=attack_true_stats,
            non_attack_true_stats=non_attack_true_stats,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            status="running" if completed < total_jobs else "finished",
        )
        atomic_write_json(result_path, {"summary": summary_partial, "results": results})

    results.sort(key=lambda x: x["id"])
    failed.sort(key=lambda x: x["id"])

    summary = build_summary(
        args=args,
        total_jobs=total_jobs,
        completed=completed,
        results=results,
        failed=failed,
        attack_true_stats=attack_true_stats,
        non_attack_true_stats=non_attack_true_stats,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        status="finished",
    )
    atomic_write_json(result_path, {"summary": summary, "results": results})

    with failed_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model_path,
                "quantization": args.quantization,
                "experiment": "RQ2_Exp3_quantization_model_comparison",
                "failed_count": len(failed),
                "note": "Failed after max_attempts local generation attempts.",
                "max_attempts": max_attempts,
                "failed_alerts": failed,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"result_file={result_path}")
    print(f"failed_log_file={failed_path}")


if __name__ == "__main__":
    main()
