#!/usr/bin/env python3
"""RQ2 API-based consistency evaluation on 500 sampled alerts."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import requests

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


def atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def extract_model_name(model: str) -> str:
    cleaned = model.strip().rstrip("/\\")
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


def call_llm_once(
    url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_content: str,
    timeout: int,
    temperature: float,
    top_p: float,
) -> tuple[str | None, int | None, str | None]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "temperature": temperature,
        "top_p": top_p,
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "AlertMeasurement-RQ2/1.0",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        resp = requests.post(
            url,
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False),
            timeout=timeout,
        )
    except Exception as e:
        return None, None, f"request_error: {e}"

    status = resp.status_code
    try:
        resp_json = resp.json()
    except Exception:
        body = resp.text[:500]
        body_l = body.lower()
        if "cloudflare" in body_l or "attention required" in body_l:
            return None, status, f"cloudflare_blocked: {body}"
        return None, status, f"non_json_response: {body}"

    if not resp.ok:
        return None, status, f"http_error: {resp_json}"

    raw_content = (
        resp_json.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    return str(raw_content).strip(), status, None


def run_single_sample(
    sample: dict[str, Any],
    url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    timeout: int,
    repeat_times: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    normalized_values: list[str] = []

    for i in range(1, repeat_times + 1):
        content, status, err = call_llm_once(
            url=url,
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            user_content=build_user_content(sample["record"]),
            timeout=timeout,
            temperature=temperature,
            top_p=top_p,
        )
        parsed = parse_label(content)
        normalized = normalize_attempt_value(parsed, content, err)
        normalized_values.append(normalized)
        attempts.append(
            {
                "attempt": i,
                "http_status": status,
                "model_output": content,
                "parsed_label": parsed,
                "normalized_value": normalized,
                "error": err,
            }
        )

    counter = Counter(normalized_values)
    max_same_value, max_same_count = max(counter.items(), key=lambda kv: kv[1])
    is_consistent_all_repeats = max_same_count == repeat_times

    return {
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
        "model": args.model,
        "model_name": model_name,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "sample_per_class": args.sample_per_class,
        "total_samples": total_jobs,
        "repeat_times": args.repeat_times,
        "threads": args.threads,
        "completed": completed,
        "consistency_metrics": {
            "all_10_consistent_count": consistent_all_10_count,
            "all_10_consistent_ratio": round(safe_div(consistent_all_10_count, total_jobs), 6),
            "avg_max_same_count": round(avg_max_same_count, 6),
        },
        "status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ2 API consistency evaluation")
    parser.add_argument("--url", default="", help="LLM API endpoint, e.g., https://api.example.com/v1/chat/completions")
    parser.add_argument("--api-key", default="", help="API key for the LLM service")
    parser.add_argument("--model", default="", help="Model name or path")
    parser.add_argument("--attack-data", default="", help="Path to the Attack JSON file")
    parser.add_argument("--fp-data", default="", help="Path to the Non-Attack JSON file")
    parser.add_argument("--sample-per-class", type=int, default=250)
    parser.add_argument("--repeat-times", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--threads", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--out-dir", default="", help="Directory for consistency evaluation result files")
    args = parser.parse_args()

    required_args = {
        "--url": args.url,
        "--api-key": args.api_key,
        "--model": args.model,
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
    if args.threads <= 0:
        raise ValueError("threads must be > 0")

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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = extract_model_name(args.model)
    temp_str = f"{args.temperature:g}"
    result_path = out_dir / f"实验1+{model_name}+{temp_str}.json"

    total_jobs = len(tasks)
    results: list[dict[str, Any]] = []
    completed = 0

    pending_ids = [t["id"] for t in tasks]
    samples_by_id = {t["id"]: t for t in tasks}

    pbar = tqdm(total=total_jobs, desc="RQ2 API Consistency Evaluating", unit="alert") if tqdm is not None else None

    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        inflight: dict[Any, int] = {}

        def submit_more() -> None:
            while pending_ids and len(inflight) < args.threads:
                sid = pending_ids.pop()
                sample = samples_by_id[sid]
                fut = ex.submit(
                    run_single_sample,
                    sample=sample,
                    url=args.url,
                    api_key=args.api_key,
                    model=args.model,
                    system_prompt=system_prompt,
                    timeout=args.timeout,
                    repeat_times=args.repeat_times,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                inflight[fut] = sid

        submit_more()

        while inflight:
            done, _ = wait(set(inflight.keys()), return_when=FIRST_COMPLETED)
            for fut in done:
                inflight.pop(fut)
                record = fut.result()
                results.append(record)
                completed += 1

                if pbar is not None:
                    pbar.update(1)
                else:
                    pct = (completed / total_jobs) * 100
                    print(f"\rRQ2 API Consistency Evaluating: {completed}/{total_jobs} ({pct:.1f}%)", end="", flush=True)

                summary_partial = build_summary(
                    args=args,
                    model_name=model_name,
                    total_jobs=total_jobs,
                    completed=completed,
                    results=results,
                    status="running" if completed < total_jobs else "finished",
                )
                atomic_write_json(result_path, {"summary": summary_partial, "results": results})

            submit_more()

    if pbar is not None:
        pbar.close()
    else:
        print()

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
