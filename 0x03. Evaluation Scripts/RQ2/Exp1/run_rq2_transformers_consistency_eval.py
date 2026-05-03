#!/usr/bin/env python3
"""RQ2 Transformers consistency evaluation on 500 sampled alerts."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter
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


def build_user_content(record: dict[str, Any]) -> str:
    fields = dict(record)
    return "Alert fields (JSON):\n" + json.dumps(fields, ensure_ascii=False, indent=2)


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

    gen_kwargs: dict[str, Any] = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.inference_mode():
        outputs = model.generate(**gen_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    new_ids = outputs[:, prompt_len:]
    text = tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()
    parsed = parse_label(text)
    return text, parsed


def atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def extract_model_name(model_path: str) -> str:
    cleaned = model_path.strip().rstrip("/\\")
    if not cleaned:
        return "model"
    model_name = os.path.basename(cleaned) or cleaned.split("/")[-1]
    model_name = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)
    return model_name or "model"


def normalize_attempt_value(parsed_label: str | None, raw_output: str | None, error: str | None) -> str:
    if parsed_label in VALID_LABELS:
        return parsed_label
    if error is not None:
        return "__ERROR__"
    text = (raw_output or "").strip()
    return text if text else "__EMPTY__"


def str2bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_summary(
    args: argparse.Namespace,
    model_name: str,
    total_jobs: int,
    completed: int,
    results: list[dict[str, Any]],
    status: str,
) -> dict[str, Any]:
    consistent_all_10_count = sum(1 for x in results if x.get("is_consistent_all_repeats") is True)
    avg_max_same_count = safe_div(sum(int(x.get("max_same_count", 0)) for x in results), len(results))
    return {
        "model_path": args.model_path,
        "model_name": model_name,
        "temperature": args.temperature,
        "do_sample": args.do_sample,
        "top_p": args.top_p,
        "sample_per_class": args.sample_per_class,
        "total_samples": total_jobs,
        "repeat_times": args.repeat_times,
        "completed": completed,
        "consistency_metrics": {
            "all_10_consistent_count": consistent_all_10_count,
            "all_10_consistent_ratio": round(safe_div(consistent_all_10_count, total_jobs), 6),
            "avg_max_same_count": round(avg_max_same_count, 6),
        },
        "status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ2 local Transformers consistency evaluation")
    parser.add_argument("--model", "--model-path", dest="model_path", default="", help="Model name or local/Hugging Face model path")
    parser.add_argument("--attack-data", default="", help="Path to the Attack JSON file")
    parser.add_argument("--fp-data", default="", help="Path to the Non-Attack JSON file")
    parser.add_argument("--sample-per-class", type=int, default=250)
    parser.add_argument("--repeat-times", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda", help="auto/cpu/cuda")
    parser.add_argument("--dtype", default="auto", help="auto/float16/bfloat16/float32")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--do-sample", type=str2bool, default=True, help="true/false")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--out-dir", default="", help="Directory for consistency evaluation result files")
    args = parser.parse_args()

    required_args = {
        "--model": args.model_path,
        "--attack-data": args.attack_data,
        "--fp-data": args.fp_data,
        "--out-dir": args.out_dir,
    }
    missing = [name for name, value in required_args.items() if not value]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")

    if args.sample_per_class <= 0:
        raise ValueError("sample-per-class must be > 0")
    if args.repeat_times <= 0:
        raise ValueError("repeat-times must be > 0")
    if args.do_sample and args.temperature <= 0:
        print(
            "warning: do_sample=True but temperature<=0; "
            "auto-switch to do_sample=False (greedy decoding)."
        )
        args.do_sample = False

    device = resolve_device(args.device)
    torch_dtype = resolve_dtype(args.dtype, device)

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
Do not output explanations or any additional text. Return only "Attack" or "Non-Attack".
"""
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    model.to(device)
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = extract_model_name(args.model_path)
    temp_str = f"{args.temperature:g}"
    result_path = out_dir / f"实验1+{model_name}+{temp_str}.json"

    results: list[dict[str, Any]] = []
    completed = 0
    total_jobs = len(tasks)

    iterable = tasks
    if tqdm is not None:
        iterable = tqdm(tasks, total=total_jobs, desc="RQ2 Consistency Evaluating", unit="alert")

    for sample in iterable:
        attempts: list[dict[str, Any]] = []
        normalized_values: list[str] = []

        for i in range(1, args.repeat_times + 1):
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
                error = None
            except Exception as e:
                raw_out = None
                parsed = None
                error = f"generate_error: {e}"

            normalized = normalize_attempt_value(parsed, raw_out, error)
            normalized_values.append(normalized)
            attempts.append(
                {
                    "attempt": i,
                    "model_output": raw_out,
                    "parsed_label": parsed,
                    "normalized_value": normalized,
                    "error": error,
                }
            )

        counter = Counter(normalized_values)
        max_same_value, max_same_count = max(counter.items(), key=lambda kv: kv[1])
        is_consistent_all_repeats = max_same_count == args.repeat_times

        results.append(
            {
                "id": sample["id"],
                "source": sample["source"],
                "true_label": sample["true_label"],
                "record": sample["record"],
                "attempts": attempts,
                "normalized_values": normalized_values,
                "max_same_value": max_same_value,
                "max_same_count": max_same_count,
                "is_consistent_all_repeats": is_consistent_all_repeats,
            }
        )
        completed += 1

        summary_partial = build_summary(
            args=args,
            model_name=model_name,
            total_jobs=total_jobs,
            completed=completed,
            results=results,
            status="running" if completed < total_jobs else "finished",
        )
        atomic_write_json(result_path, {"summary": summary_partial, "results": results})

    results.sort(key=lambda x: x["id"])
    summary = build_summary(
        args=args,
        model_name=model_name,
        total_jobs=total_jobs,
        completed=completed,
        results=results,
        status="finished",
    )
    atomic_write_json(result_path, {"summary": summary, "results": results})

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"result_file={result_path}")


if __name__ == "__main__":
    main()
