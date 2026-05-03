#!/usr/bin/env python3
"""RQ2: Configuration Experiment (User-baseline) transformers-based single-thread evaluation for Attack vs Non-Attack alerts."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


VALID_LABELS = ("Attack", "Non-Attack")
EXPERIMENT_NAME = "RQ2: Configuration Experiment (User-baseline)"


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


def extract_model_short_name(model: str) -> str:
    cleaned = model.strip().rstrip("/\\")
    if not cleaned:
        return "model"
    # For local paths use final path segment; for hub ids use the last "/" segment.
    return Path(cleaned).name or cleaned.split("/")[-1] or "model"


def build_user_content(record: dict[str, Any]) -> str:
    fields = dict(record)
    return "Alert fields (JSON):\n" + json.dumps(fields, ensure_ascii=False, indent=2)


def build_multiturn_messages(
    role_setting_prompt: str,
    task_definition_prompt: str,
    constraint_prompt: str,
    user_content: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": role_setting_prompt},
        {"role": "user", "content": task_definition_prompt},
        {"role": "assistant", "content": "Confirmed. I understand the task definition."},
        {"role": "user", "content": constraint_prompt},
        {
            "role": "assistant",
            "content": "Confirmed. I will strictly follow all constraints and only output one final label.",
        },
        {
            "role": "user",
            "content": (
                "Task confirmation: Please confirm you are ready.\n"
                "Positive start: Great, start now and classify the following alert.\n\n"
                f"{user_content}"
            ),
        },
    ]


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


def resolve_device(device_name: str) -> str:
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_name


def resolve_torch_dtype(dtype_name: str, device: str) -> torch.dtype | None:
    name = dtype_name.lower().strip()
    if name == "auto":
        return torch.float16 if device == "cuda" else torch.float32
    if name in {"float16", "fp16"}:
        return torch.float16
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if name in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def render_prompt(tokenizer: AutoTokenizer, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        role_map = {"system": "System", "user": "User", "assistant": "Assistant"}
        parts: list[str] = []
        for msg in messages:
            role = role_map.get(msg.get("role", ""), msg.get("role", "User").capitalize())
            parts.append(f"{role}:\n{msg.get('content', '')}\n")
        parts.append("Assistant:\n")
        return "\n".join(parts)


def call_llm_once(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    messages: list[dict[str, str]],
    max_input_tokens: int,
    max_new_tokens: int,
    temperature: float,
) -> tuple[str | None, str | None]:
    prompt = render_prompt(tokenizer, messages)
    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": pad_token_id,
        }
        if tokenizer.eos_token_id is not None:
            gen_kwargs["eos_token_id"] = tokenizer.eos_token_id
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            output_ids = model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][input_len:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return text, None
    except Exception as e:
        return None, f"inference_error: {e}"


def run_single_attempt(
    sample: dict[str, Any],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    role_setting_prompt: str,
    task_definition_prompt: str,
    constraint_prompt: str,
    max_input_tokens: int,
    max_new_tokens: int,
    temperature: float,
    sleep_before: float,
) -> dict[str, Any]:
    if sleep_before > 0:
        time.sleep(sleep_before)

    user_content = build_user_content(sample["record"])
    messages = build_multiturn_messages(
        role_setting_prompt=role_setting_prompt,
        task_definition_prompt=task_definition_prompt,
        constraint_prompt=constraint_prompt,
        user_content=user_content,
    )

    content, err = call_llm_once(
        model=model,
        tokenizer=tokenizer,
        device=device,
        messages=messages,
        max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    parsed = parse_label(content)
    return {
        "model_output": content,
        "parsed_label": parsed,
        "error": err,
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
        "experiment": EXPERIMENT_NAME,
        "model": args.model,
        "prompt_tag": args.prompt_tag,
        "total": total_jobs,
        "completed": completed,
        "success": len(results) - len(failed),
        "failed": len(failed),
        "sample_per_class": args.sample_per_class,
        "attack_samples": args.sample_per_class,
        "non_attack_samples": args.sample_per_class,
        "threads": 1,
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


def load_local_model(args: argparse.Namespace) -> tuple[AutoTokenizer, AutoModelForCausalLM, str]:
    device = resolve_device(args.device)
    torch_dtype = resolve_torch_dtype(args.dtype, device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
    }
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.to(device)

    model.eval()
    return tokenizer, model, device


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{EXPERIMENT_NAME} local-transformers single-thread evaluation")
    parser.add_argument("--model", default="", help="Model name or local/Hugging Face model path")
    parser.add_argument("--attack-data", default="", help="Path to the Attack JSON file")
    parser.add_argument("--fp-data", default="", help="Path to the Non-Attack JSON file")
    parser.add_argument("--sample-per-class", type=int, default=250)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--retry-times", type=int, default=3, help="extra retries after first attempt")
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true")
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.set_defaults(trust_remote_code=True)
    parser.add_argument("--prompt-tag", default="user-baseline", help="Prompt identifier appended to output file names")
    parser.add_argument("--out-dir", default="", help="Directory for prompting experiment result files")
    parser.add_argument("--log-dir", default="", help="Directory for failed-case logs")
    args = parser.parse_args()

    required_args = {
        "--model": args.model,
        "--attack-data": args.attack_data,
        "--fp-data": args.fp_data,
        "--out-dir": args.out_dir,
        "--log-dir": args.log_dir,
    }
    missing = [name for name, value in required_args.items() if not value]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")

    attack_records = load_json_list(Path(args.attack_data))
    fp_records = load_json_list(Path(args.fp_data))
    if len(attack_records) < args.sample_per_class:
        raise ValueError(f"attack records not enough: {len(attack_records)} < {args.sample_per_class}")
    if len(fp_records) < args.sample_per_class:
        raise ValueError(f"fp records not enough: {len(fp_records)} < {args.sample_per_class}")

    tokenizer, model, device = load_local_model(args)
    print(
        f"[config] device={device}, dtype={args.dtype}, trust_remote_code={args.trust_remote_code}, "
        f"max_input_tokens={args.max_input_tokens}, max_new_tokens={args.max_new_tokens}, retry_times={args.retry_times}",
        flush=True,
    )

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

    role_setting_prompt = (
"""
You are an experienced Tier-1 SOC alert triage analyst.
Your responsibility is to make accurate and conservative alert classification decisions.
"""
    )
    task_definition_prompt = (
"""
Task definition:
You will be given one network alert record and must classify it into exactly one of two labels:
- Attack: real malicious activity, attack attempt, exploitation behavior, or other security-threatening actions.
- Non-Attack: benign traffic, normal business activity, internal testing, false positives, or informational events.
"""
    )
    constraint_prompt = (
"""
Strengthened constraints:
1) Base your decision only on the provided alert fields.
2) Do not output explanations, analysis, or any extra text.
3) Output must be exactly one label: "Attack" or "Non-Attack".
4) If evidence is insufficient for a real attack, choose "Non-Attack".
"""
    )

    out_dir = Path(args.out_dir)
    log_dir = Path(args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    model_name = safe_model_name(extract_model_short_name(args.model))
    prompt_tag = safe_model_name(args.prompt_tag or "prompt")
    result_path = out_dir / f"{model_name}_{prompt_tag}.json"
    failed_path = log_dir / f"{model_name}_failed_{prompt_tag}.json"

    total_jobs = len(tasks)
    max_attempts = max(1, args.retry_times + 1)

    results: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    completed = 0

    attack_true_stats = {"llm_attack": 0, "llm_non_attack": 0, "llm_other": 0, "failed": 0}
    non_attack_true_stats = {"llm_attack": 0, "llm_non_attack": 0, "llm_other": 0, "failed": 0}
    tp = fp = tn = fn = 0

    iterator = tqdm(tasks, desc=f"{EXPERIMENT_NAME} Evaluating", unit="alert") if tqdm is not None else tasks
    for sample in iterator:
        attempts: list[dict[str, Any]] = []
        final_attempt: dict[str, Any] | None = None
        is_success = False

        for attempt_idx in range(1, max_attempts + 1):
            sleep_before = args.retry_delay if attempt_idx > 1 else 0.0
            attempt_res = run_single_attempt(
                sample=sample,
                model=model,
                tokenizer=tokenizer,
                device=device,
                role_setting_prompt=role_setting_prompt,
                task_definition_prompt=task_definition_prompt,
                constraint_prompt=constraint_prompt,
                max_input_tokens=args.max_input_tokens,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                sleep_before=sleep_before,
            )
            attempt_res["attempt"] = attempt_idx
            attempts.append(attempt_res)
            final_attempt = attempt_res
            if attempt_res.get("parsed_label") in VALID_LABELS:
                is_success = True
                break

        llm_label = final_attempt.get("parsed_label") if (is_success and final_attempt) else None
        llm_output_raw = final_attempt.get("model_output") if final_attempt else None

        record = {
            "id": sample["id"],
            "source": sample["source"],
            "true_label": sample["true_label"],
            "llm_label": llm_label,
            "llm_output_raw": llm_output_raw,
            "record": sample["record"],
            "ok": is_success,
            "attempts": attempts,
        }
        results.append(record)
        completed += 1

        cls_stats = attack_true_stats if sample["true_label"] == "Attack" else non_attack_true_stats
        if not is_success:
            failed.append(record)
            cls_stats["failed"] += 1
        else:
            if llm_label == "Attack":
                cls_stats["llm_attack"] += 1
            elif llm_label == "Non-Attack":
                cls_stats["llm_non_attack"] += 1
            else:
                cls_stats["llm_other"] += 1

            if sample["true_label"] == "Attack":
                if llm_label == "Attack":
                    tp += 1
                elif llm_label == "Non-Attack":
                    fn += 1
            else:
                if llm_label == "Attack":
                    fp += 1
                elif llm_label == "Non-Attack":
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
                "experiment": EXPERIMENT_NAME,
                "model": args.model,
                "prompt_tag": args.prompt_tag,
                "failed_count": len(failed),
                "note": "Failed after max_attempts attempts.",
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
