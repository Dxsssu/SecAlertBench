#!/usr/bin/env python3
"""RQ2: Configuration Experiment (Zero-shot) API-based multi-thread evaluation for Attack vs Non-Attack alerts."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import requests

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


VALID_LABELS = ("Attack", "Non-Attack")
EXPERIMENT_NAME = "RQ2: Configuration Experiment (Zero-shot)"


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


def call_llm_once(
    url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_content: str,
    timeout: int,
) -> tuple[str | None, int | None, str | None, dict[str, Any] | None]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "temperature": 0,
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "AlertMeasurement-RQ2-ConfigExperiment/1.0",
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
        return None, None, f"request_error: {e}", None

    status = resp.status_code
    try:
        resp_json = resp.json()
    except Exception:
        body = resp.text[:500]
        body_l = body.lower()
        if "cloudflare" in body_l or "attention required" in body_l:
            return None, status, f"cloudflare_blocked: {body}", None
        return None, status, f"non_json_response: {body}", None

    if not resp.ok:
        return None, status, f"http_error: {resp_json}", resp_json

    raw_content = (
        resp_json.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    return str(raw_content).strip(), status, None, resp_json


def run_single_attempt(
    sample: dict[str, Any],
    url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    timeout: int,
    sleep_before: float,
) -> dict[str, Any]:
    if sleep_before > 0:
        time.sleep(sleep_before)

    content, status, err, _ = call_llm_once(
        url=url,
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_content=build_user_content(sample["record"]),
        timeout=timeout,
    )
    parsed = parse_label(content)
    return {
        "http_status": status,
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
    worker_restart_counts: list[int],
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
        "threads": args.threads,
        "worker_restart_total": sum(worker_restart_counts),
        "worker_restart_by_index": {str(i): c for i, c in enumerate(worker_restart_counts) if c > 0},
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


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{EXPERIMENT_NAME} large-scale multi-thread LLM evaluation")
    parser.add_argument("--url", default="", help="LLM API endpoint, e.g., https://api.example.com/v1/chat/completions")
    parser.add_argument("--api-key", default="", help="API key for the LLM service")
    parser.add_argument("--model", default="", help="Model name or path")
    parser.add_argument("--attack-data", default="", help="Path to the Attack JSON file")
    parser.add_argument("--fp-data", default="", help="Path to the Non-Attack JSON file")
    parser.add_argument("--sample-per-class", type=int, default=250)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--threads", type=int, default=40)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retry-times", type=int, default=10, help="extra retries after first attempt")
    parser.add_argument("--retry-delay", type=float, default=5.0)
    parser.add_argument("--prompt-tag", default="zero_shot", help="Prompt identifier appended to output file names")
    parser.add_argument("--out-dir", default="", help="Directory for prompting experiment result files")
    parser.add_argument("--log-dir", default="", help="Directory for failed-case logs")
    args = parser.parse_args()

    required_args = {
        "--url": args.url,
        "--api-key": args.api_key,
        "--model": args.model,
        "--attack-data": args.attack_data,
        "--fp-data": args.fp_data,
        "--out-dir": args.out_dir,
        "--log-dir": args.log_dir,
    }
    missing = [name for name, value in required_args.items() if not value]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")

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
    log_dir = Path(args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    model_name = safe_model_name(args.model)
    prompt_tag = safe_model_name(args.prompt_tag or "prompt")
    result_path = out_dir / f"{model_name}_{prompt_tag}.json"
    failed_path = log_dir / f"{model_name}_failed_{prompt_tag}.json"

    total_jobs = len(tasks)
    max_attempts = max(1, args.retry_times + 1)

    # Per-sample mutable state; retries are re-submitted to the thread pool.
    states: dict[int, dict[str, Any]] = {
        t["id"]: {
            "sample": t,
            "attempt_count": 0,
            "attempts": [],
            "finalized": False,
        }
        for t in tasks
    }
    pending_ids = [t["id"] for t in tasks]

    results: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    completed = 0

    attack_true_stats = {"llm_attack": 0, "llm_non_attack": 0, "llm_other": 0, "failed": 0}
    non_attack_true_stats = {"llm_attack": 0, "llm_non_attack": 0, "llm_other": 0, "failed": 0}
    tp = fp = tn = fn = 0
    worker_restart_counts = [0 for _ in range(args.threads)]

    pbar = tqdm(total=total_jobs, desc=f"{EXPERIMENT_NAME} Evaluating", unit="alert") if tqdm is not None else None

    worker_executors = [ThreadPoolExecutor(max_workers=1) for _ in range(args.threads)]
    idle_workers = set(range(args.threads))
    inflight: dict[Any, tuple[int, int]] = {}

    def restart_worker(worker_idx: int) -> None:
        worker_executors[worker_idx].shutdown(wait=False, cancel_futures=True)
        worker_executors[worker_idx] = ThreadPoolExecutor(max_workers=1)
        worker_restart_counts[worker_idx] += 1

    def submit_to_worker(worker_idx: int, sid: int) -> None:
        st = states[sid]
        sleep_before = args.retry_delay if st["attempt_count"] > 0 else 0.0
        fut = worker_executors[worker_idx].submit(
            run_single_attempt,
            sample=st["sample"],
            url=args.url,
            api_key=args.api_key,
            model=args.model,
            system_prompt=system_prompt,
            timeout=args.timeout,
            sleep_before=sleep_before,
        )
        inflight[fut] = (sid, worker_idx)

    def submit_more() -> None:
        while pending_ids and idle_workers:
            worker_idx = idle_workers.pop()
            sid = pending_ids.pop()
            submit_to_worker(worker_idx, sid)

    try:
        submit_more()
        while inflight or pending_ids:
            if not inflight:
                submit_more()
                continue

            done, _ = wait(set(inflight.keys()), return_when=FIRST_COMPLETED)
            for fut in done:
                sid, worker_idx = inflight.pop(fut)
                st = states[sid]
                sample = st["sample"]

                attempt_res = fut.result()
                st["attempt_count"] += 1
                st["attempts"].append(
                    {
                        "attempt": st["attempt_count"],
                        **attempt_res,
                    }
                )

                parsed = attempt_res.get("parsed_label")
                is_success = parsed in VALID_LABELS

                if is_success or st["attempt_count"] >= max_attempts:
                    llm_label = parsed if is_success else None
                    llm_output_raw = attempt_res.get("model_output")
                    record = {
                        "id": sample["id"],
                        "source": sample["source"],
                        "true_label": sample["true_label"],
                        "llm_label": llm_label,
                        "llm_output_raw": llm_output_raw,
                        "record": sample["record"],
                        "ok": is_success,
                        "attempts": st["attempts"],
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

                    if pbar is not None:
                        pbar.update(1)
                    else:
                        pct = (completed / total_jobs) * 100
                        print(f"\r{EXPERIMENT_NAME} Evaluating: {completed}/{total_jobs} ({pct:.1f}%)", end="", flush=True)

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
                        worker_restart_counts=worker_restart_counts,
                        status="running" if completed < total_jobs else "finished",
                    )
                    atomic_write_json(result_path, {"summary": summary_partial, "results": results})
                    idle_workers.add(worker_idx)
                else:
                    # Restart only this failed worker and immediately re-submit this same sample on it.
                    restart_worker(worker_idx)
                    submit_to_worker(worker_idx, sid)

            submit_more()
    finally:
        for ex in worker_executors:
            ex.shutdown(wait=True, cancel_futures=True)

    if pbar is not None:
        pbar.close()
    else:
        print()

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
        worker_restart_counts=worker_restart_counts,
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
                "note": "Failed after max_attempts attempts."
                        " If an attempt fails before max_attempts, only the failed worker thread is restarted"
                        " and the same sample is immediately resubmitted on that restarted worker.",
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
