#!/usr/bin/env python3
"""RQ3 local Transformers latency and GPU-memory profiling for alert handling."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import re
import statistics
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
PROMPT_MODES = ("zero", "few", "cot")
PROMPT_MODE_LABELS = {
    "zero": "Zero-shot",
    "few": "Few-shot",
    "cot": "CoT",
}
SECONDS_PER_DAY = 24 * 60 * 60



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


def sample_reference_records(
    records: list[dict[str, Any]],
    k: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if not records:
        raise ValueError("reference records are empty")
    if len(records) >= k:
        return rng.sample(records, k)
    return [rng.choice(records) for _ in range(k)]


def build_few_shot_user_content(
    target_record: dict[str, Any],
    ref_attack_records: list[dict[str, Any]],
    ref_non_attack_records: list[dict[str, Any]],
) -> str:
    parts: list[str] = []
    for i, rec in enumerate(ref_attack_records, start=1):
        parts.append(
            f"Reference Example A{i} (Attack):\n"
            + json.dumps(dict(rec), ensure_ascii=False, indent=2)
            + f"\nReference Label A{i}: Attack\n"
        )
    for i, rec in enumerate(ref_non_attack_records, start=1):
        parts.append(
            f"Reference Example N{i} (Non-Attack):\n"
            + json.dumps(dict(rec), ensure_ascii=False, indent=2)
            + f"\nReference Label N{i}: Non-Attack\n"
        )
    parts.append(
        "Now classify the following target alert.\n"
        + "Target Alert fields (JSON):\n"
        + json.dumps(dict(target_record), ensure_ascii=False, indent=2)
    )
    return "\n".join(parts)


def build_prompt_mode_user_content(
    prompt_mode: str,
    target_record: dict[str, Any],
    attack_records: list[dict[str, Any]],
    fp_records: list[dict[str, Any]],
    rng: random.Random,
) -> str:
    if prompt_mode == "few":
        return build_few_shot_user_content(
            target_record=target_record,
            ref_attack_records=sample_reference_records(attack_records, 3, rng),
            ref_non_attack_records=sample_reference_records(fp_records, 3, rng),
        )
    return build_user_content(target_record)


def get_system_prompt(prompt_mode: str) -> str:
    if prompt_mode == "few":
        return """
<task description>
You will be given six labeled reference alerts (three Attack and three Non-Attack) and one target alert.
Use the references as guidance, then classify the target alert into one of two labels:
- Attack: The alert indicates a real malicious activity, an attack attempt, or other security-threatening actions.
- Non-Attack: The alert does not indicate a real attack. It may correspond to benign traffic, normal operations, false positives, or informational events.
<constraints>
Do not output explanations or any additional text. Return only "Attack" or "Non-Attack"./no_think
"""
    if prompt_mode == "cot":
        return """
<task description>
You will be given one network alert record and your task is to classify the alert into one of two labels:
- Attack: The alert indicates a real malicious activity, an attack attempt, or other security-threatening actions.
- Non-Attack: The alert does not indicate a real attack. It may correspond to benign traffic, normal operations, false positives, or informational events.
<constraints>
Before giving the final label, first output a concise reasoning process in no more than 200 words.
Then output the final label.
The final label must be exactly one of: "Attack" or "Non-Attack".
Do not output any additional content beyond the required format./no_think
<output format>
Reasoning: <your reasoning process, <=200 words>
Label: <Attack or Non-Attack>
"""
    return """
<task description>
You will be given one network alert record and your task is to classify the alert into one of two labels:
- Attack: The alert indicates a real malicious activity, an attack attempt, or other security-threatening actions.
- Non-Attack: The alert does not indicate a real attack. It may correspond to benign traffic, normal operations, false positives, or informational events.
<constraints>
Do not output explanations or any additional text. Return only "Attack" or "Non-Attack"./no_think
"""


def parse_label(content: str | None) -> str | None:
    if content is None:
        return None
    text = str(content).strip()
    if not text:
        return None

    label_line = re.search(
        r"(?im)^\s*(?:final\s*)?(?:label|result|classification)\s*[:：]\s*(.+?)\s*$",
        text,
    )
    if label_line:
        val = label_line.group(1).strip()
        if val in VALID_LABELS:
            return val
        if "Non-Attack" in val:
            return "Non-Attack"
        if "Attack" in val:
            return "Attack"

    if text in VALID_LABELS:
        return text

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        last = lines[-1]
        if last in VALID_LABELS:
            return last
        if "Non-Attack" in last:
            return "Non-Attack"
        if "Attack" in last:
            return "Attack"

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


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def round_float(value: float, ndigits: int = 6) -> float:
    return round(float(value), ndigits)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def resolve_dtype(dtype: str, device: str) -> torch.dtype | None:
    if dtype == "auto":
        if device.startswith("cuda"):
            return torch.float16
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def is_cuda_device(device: str) -> bool:
    return device.startswith("cuda") and torch.cuda.is_available()


def cuda_sync(device: str) -> None:
    if is_cuda_device(device):
        torch.cuda.synchronize(torch.device(device))


def reset_cuda_peak_memory(device: str) -> None:
    if is_cuda_device(device):
        torch.cuda.reset_peak_memory_stats(torch.device(device))


def gpu_memory_snapshot(device: str) -> dict[str, float | None]:
    if not is_cuda_device(device):
        return {
            "allocated_mb": None,
            "reserved_mb": None,
            "max_allocated_mb": None,
            "max_reserved_mb": None,
        }

    dev = torch.device(device)
    mb = 1024 * 1024
    return {
        "allocated_mb": round_float(torch.cuda.memory_allocated(dev) / mb, 3),
        "reserved_mb": round_float(torch.cuda.memory_reserved(dev) / mb, 3),
        "max_allocated_mb": round_float(torch.cuda.max_memory_allocated(dev) / mb, 3),
        "max_reserved_mb": round_float(torch.cuda.max_memory_reserved(dev) / mb, 3),
    }


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
    max_input_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    prompt = build_prompt(tokenizer, system_prompt, user_content)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )
    prompt_tokens = int(inputs["input_ids"].shape[1])
    inputs = {k: v.to(device) for k, v in inputs.items()}

    cuda_sync(device)
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    cuda_sync(device)
    latency_seconds = time.perf_counter() - start

    new_ids = outputs[:, prompt_tokens:]
    output_tokens = int(new_ids.shape[1])
    text = tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()
    return {
        "latency_seconds": latency_seconds,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "output_raw": text,
        "parsed_label": parse_label(text),
    }


def build_latency_summary(latencies: list[float]) -> dict[str, float | int]:
    if not latencies:
        return {
            "count": 0,
            "mean_seconds": 0.0,
            "median_seconds": 0.0,
            "p90_seconds": 0.0,
            "p95_seconds": 0.0,
            "p99_seconds": 0.0,
            "min_seconds": 0.0,
            "max_seconds": 0.0,
            "std_seconds": 0.0,
            "alerts_per_second_by_mean": 0.0,
            "alerts_per_day_by_mean": 0.0,
            "alerts_per_day_by_p95": 0.0,
        }

    mean_latency = statistics.mean(latencies)
    p95_latency = percentile(latencies, 0.95)
    std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    return {
        "count": len(latencies),
        "mean_seconds": round_float(mean_latency),
        "median_seconds": round_float(statistics.median(latencies)),
        "p90_seconds": round_float(percentile(latencies, 0.90)),
        "p95_seconds": round_float(p95_latency),
        "p99_seconds": round_float(percentile(latencies, 0.99)),
        "min_seconds": round_float(min(latencies)),
        "max_seconds": round_float(max(latencies)),
        "std_seconds": round_float(std_latency),
        "alerts_per_second_by_mean": round_float(safe_div(1.0, mean_latency)),
        "alerts_per_day_by_mean": round_float(safe_div(SECONDS_PER_DAY, mean_latency), 2),
        "alerts_per_day_by_p95": round_float(safe_div(SECONDS_PER_DAY, p95_latency), 2),
    }


def write_latency_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "source",
        "ok",
        "latency_seconds",
        "prompt_tokens",
        "output_tokens",
        "parsed_label",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k) for k in fieldnames})


def build_summary(
    args: argparse.Namespace,
    model_path: str,
    prompt_mode: str,
    device: str,
    selected_total: int,
    measured_total: int,
    completed: int,
    failed_count: int,
    warmup_count: int,
    measured_latencies: list[float],
    load_seconds: float,
    model_loaded_memory: dict[str, float | None],
    eval_memory: dict[str, float | None],
    status: str,
) -> dict[str, Any]:
    return {
        "model": model_path,
        "device": device,
        "dtype": args.dtype,
        "total": measured_total,
        "selected_total": selected_total,
        "measured_total": measured_total,
        "completed": completed,
        "success": completed - failed_count,
        "failed": failed_count,
        "sample_per_class": args.sample_per_class,
        "prompt_mode": prompt_mode,
        "warmup_samples_requested": args.warmup_samples,
        "warmup_samples_used": warmup_count,
        "generation": {
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "model_load_seconds": round_float(load_seconds),
        "latency": build_latency_summary(measured_latencies),
        "gpu_memory_after_model_load": model_loaded_memory,
        "gpu_memory_eval_peak": eval_memory,
        "status": status,
    }


def build_tasks(
    attack_records: list[dict[str, Any]],
    fp_records: list[dict[str, Any]],
    sample_per_class: int,
    seed: int | None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    sampled_attack = rng.sample(attack_records, sample_per_class)
    sampled_fp = rng.sample(fp_records, sample_per_class)

    tasks: list[dict[str, Any]] = []
    uid = 1
    for rec in sampled_attack:
        tasks.append({"id": uid, "source": "attack_clean", "record": rec})
        uid += 1
    for rec in sampled_fp:
        tasks.append({"id": uid, "source": "fp_clean", "record": rec})
        uid += 1
    rng.shuffle(tasks)
    return tasks


def make_reference_rng(seed: int | None, prompt_mode: str, sample_id: int) -> random.Random:
    return random.Random(f"rq3:{seed}:{prompt_mode}:{sample_id}")


def release_device_memory(device: str) -> None:
    gc.collect()
    if is_cuda_device(device):
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        cuda_sync(device)


def run_latency_for_prompt(
    args: argparse.Namespace,
    model_path: str,
    prompt_mode: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: str,
    attack_records: list[dict[str, Any]],
    fp_records: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    load_seconds: float,
    model_loaded_memory: dict[str, float | None],
) -> dict[str, Any]:
    system_prompt = get_system_prompt(prompt_mode)

    out_dir = Path(args.out_dir)
    log_dir = Path(args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    model_tag = safe_model_name(model_path)
    prompt_tag = safe_model_name(prompt_mode)
    result_path = out_dir / f"{model_tag}_{prompt_tag}_latency.json"
    csv_path = out_dir / f"{model_tag}_{prompt_tag}_latency_samples.csv"
    failed_path = log_dir / f"{model_tag}_{prompt_tag}_latency_failed.json"

    warmup_count = min(max(0, args.warmup_samples), len(tasks))
    warmup_tasks = tasks[:warmup_count]
    measure_tasks = tasks[warmup_count:]
    selected_total = len(tasks)

    prompt_mode_label = PROMPT_MODE_LABELS.get(prompt_mode, prompt_mode)
    model_label = Path(model_path.rstrip("/\\")).name or safe_model_name(model_path)
    warmup_iterable = warmup_tasks
    if tqdm is not None and warmup_tasks:
        warmup_iterable = tqdm(
            warmup_tasks,
            total=len(warmup_tasks),
            desc=f"RQ3 {model_label} {prompt_mode_label} Warmup",
            unit="alert",
        )

    for sample in warmup_iterable:
        generate_once(
            model=model,
            tokenizer=tokenizer,
            device=device,
            system_prompt=system_prompt,
            user_content=build_prompt_mode_user_content(
                prompt_mode=prompt_mode,
                target_record=sample["record"],
                attack_records=attack_records,
                fp_records=fp_records,
                rng=make_reference_rng(args.seed, prompt_mode, sample["id"]),
            ),
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    reset_cuda_peak_memory(device)

    records: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    latencies: list[float] = []
    total_jobs = len(measure_tasks)

    iterable = measure_tasks
    if tqdm is not None:
        iterable = tqdm(
            measure_tasks,
            total=total_jobs,
            desc=f"RQ3 {model_label} {prompt_mode_label} Latency Profiling",
            unit="alert",
        )

    for idx, sample in enumerate(iterable, start=1):
        record = {
            "id": sample["id"],
            "source": sample["source"],
            "ok": False,
            "latency_seconds": None,
            "prompt_tokens": None,
            "output_tokens": None,
            "parsed_label": None,
            "output_raw": None,
            "error": None,
        }
        try:
            generated = generate_once(
                model=model,
                tokenizer=tokenizer,
                device=device,
                system_prompt=system_prompt,
                user_content=build_prompt_mode_user_content(
                    prompt_mode=prompt_mode,
                    target_record=sample["record"],
                    attack_records=attack_records,
                    fp_records=fp_records,
                    rng=make_reference_rng(args.seed, prompt_mode, sample["id"]),
                ),
                max_input_tokens=args.max_input_tokens,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            latency = float(generated["latency_seconds"])
            record.update(
                {
                    "ok": True,
                    "latency_seconds": round_float(latency),
                    "prompt_tokens": generated["prompt_tokens"],
                    "output_tokens": generated["output_tokens"],
                    "parsed_label": generated["parsed_label"],
                    "output_raw": generated["output_raw"],
                }
            )
            latencies.append(latency)
        except Exception as e:
            record["error"] = f"generate_error: {e}"
            failed.append(record)

        records.append(record)

        summary_partial = build_summary(
            args=args,
            model_path=model_path,
            prompt_mode=prompt_mode,
            device=device,
            selected_total=selected_total,
            measured_total=total_jobs,
            completed=idx,
            failed_count=len(failed),
            warmup_count=warmup_count,
            measured_latencies=latencies,
            load_seconds=load_seconds,
            model_loaded_memory=model_loaded_memory,
            eval_memory=gpu_memory_snapshot(device),
            status="running" if idx < total_jobs else "finished",
        )
        atomic_write_json(result_path, {"summary": summary_partial, "results": records})

    records.sort(key=lambda x: x["id"])
    failed.sort(key=lambda x: x["id"])
    eval_memory = gpu_memory_snapshot(device)

    summary = build_summary(
        args=args,
        model_path=model_path,
        prompt_mode=prompt_mode,
        device=device,
        selected_total=selected_total,
        measured_total=total_jobs,
        completed=len(records),
        failed_count=len(failed),
        warmup_count=warmup_count,
        measured_latencies=latencies,
        load_seconds=load_seconds,
        model_loaded_memory=model_loaded_memory,
        eval_memory=eval_memory,
        status="finished",
    )
    atomic_write_json(result_path, {"summary": summary, "results": records})
    write_latency_csv(csv_path, records)

    with failed_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model_path,
                "prompt_mode": prompt_mode,
                "failed_count": len(failed),
                "note": "Failed local generation attempts during RQ3 latency profiling.",
                "failed_alerts": failed,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"result_file={result_path}")
    print(f"latency_csv_file={csv_path}")
    print(f"failed_log_file={failed_path}")

    return {
        "summary": summary,
        "result_file": str(result_path),
        "latency_csv_file": str(csv_path),
        "failed_log_file": str(failed_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ3 batch local Transformers latency and GPU-memory profiling")
    parser.add_argument("--model", "--model-path", dest="model_path", default="", help="Model name or local/Hugging Face model path")
    parser.add_argument(
        "--model-paths",
        nargs="+",
        default=None,
        help="Batch model paths. Models are loaded one by one.",
    )
    parser.add_argument("--attack-data", default="", help="Path to the Attack JSON file")
    parser.add_argument("--fp-data", default="", help="Path to the Non-Attack JSON file")
    parser.add_argument("--sample-per-class", type=int, default=250)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda", help="auto/cpu/cuda/cuda:0")
    parser.add_argument("--dtype", default="auto", help="auto/float16/bfloat16/float32")
    parser.add_argument("--prompt-mode", choices=PROMPT_MODES, default=None, help="Run a single prompt style instead of all prompt modes")
    parser.add_argument(
        "--prompt-modes",
        nargs="+",
        choices=PROMPT_MODES,
        default=list(PROMPT_MODES),
        help="Prompt styles to run for each model",
    )
    parser.add_argument("--max-input-tokens", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling generation")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--warmup-samples", type=int, default=3, help="Warmup alerts excluded from latency statistics")
    parser.add_argument("--out-dir", default="", help="Directory for RQ3 latency result files")
    parser.add_argument("--log-dir", default="", help="Directory for failed-case logs")
    args = parser.parse_args()

    required_args = {
        "--attack-data": args.attack_data,
        "--fp-data": args.fp_data,
        "--out-dir": args.out_dir,
        "--log-dir": args.log_dir,
    }
    missing = [name for name, value in required_args.items() if not value]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")
    if not args.model_path and not args.model_paths:
        raise ValueError("Missing required argument: --model/--model-path or --model-paths")

    model_paths = [args.model_path] if args.model_path else list(args.model_paths)
    prompt_modes = [args.prompt_mode] if args.prompt_mode else list(args.prompt_modes)

    device = resolve_device(args.device)
    torch_dtype = resolve_dtype(args.dtype, device)

    attack_records = load_json_list(Path(args.attack_data))
    fp_records = load_json_list(Path(args.fp_data))
    if len(attack_records) < args.sample_per_class:
        raise ValueError(f"attack records not enough: {len(attack_records)} < {args.sample_per_class}")
    if len(fp_records) < args.sample_per_class:
        raise ValueError(f"fp records not enough: {len(fp_records)} < {args.sample_per_class}")

    tasks = build_tasks(
        attack_records=attack_records,
        fp_records=fp_records,
        sample_per_class=args.sample_per_class,
        seed=args.seed,
    )

    out_dir = Path(args.out_dir)
    log_dir = Path(args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    batch_summary_path = out_dir / "rq3_batch_latency_summary.json"
    batch_results: list[dict[str, Any]] = []

    print(
        json.dumps(
            {
                "batch_models": model_paths,
                "batch_prompt_modes": prompt_modes,
                "device": device,
                "dtype": args.dtype,
                "sample_per_class": args.sample_per_class,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    for model_index, model_path in enumerate(model_paths, start=1):
        tokenizer = None
        model = None
        try:
            print(
                f"[batch] loading model {model_index}/{len(model_paths)}: {model_path}",
                flush=True,
            )
            reset_cuda_peak_memory(device)
            cuda_sync(device)
            load_start = time.perf_counter()
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            )
            model.to(device)
            model.eval()
            cuda_sync(device)
            load_seconds = time.perf_counter() - load_start

            model_loaded_memory = gpu_memory_snapshot(device)
            reset_cuda_peak_memory(device)

            for prompt_index, prompt_mode in enumerate(prompt_modes, start=1):
                print(
                    f"[batch] model {model_index}/{len(model_paths)} prompt {prompt_index}/{len(prompt_modes)}: {prompt_mode}",
                    flush=True,
                )
                run_result = run_latency_for_prompt(
                    args=args,
                    model_path=model_path,
                    prompt_mode=prompt_mode,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    attack_records=attack_records,
                    fp_records=fp_records,
                    tasks=tasks,
                    load_seconds=load_seconds,
                    model_loaded_memory=model_loaded_memory,
                )
                batch_results.append(run_result)
                atomic_write_json(
                    batch_summary_path,
                    {
                        "status": "running",
                        "model_paths": model_paths,
                        "prompt_modes": prompt_modes,
                        "finished_runs": len(batch_results),
                        "total_runs": len(model_paths) * len(prompt_modes),
                        "runs": batch_results,
                    },
                )
        finally:
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            release_device_memory(device)
            print(f"[batch] unloaded model: {model_path}", flush=True)

    atomic_write_json(
        batch_summary_path,
        {
            "status": "finished",
            "model_paths": model_paths,
            "prompt_modes": prompt_modes,
            "finished_runs": len(batch_results),
            "total_runs": len(model_paths) * len(prompt_modes),
            "runs": batch_results,
        },
    )
    print(f"batch_summary_file={batch_summary_path}")


if __name__ == "__main__":
    main()
