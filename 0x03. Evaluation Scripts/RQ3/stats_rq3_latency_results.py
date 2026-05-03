#!/usr/bin/env python3
"""Summarize RQ3 latency results for selected Qwen3 models."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


PROMPT_ORDER = ("zero", "few", "cot")

MODEL_ALIASES = {
    "Qwen3-32B": (
        "Qwen3-32B",
    ),
    "Qwen3-30B-A3B": (
        "Qwen3-30B-A3B-Instruct-2507",
        "Qwen3-30B-A3B",
    ),
    "Qwen3-14B": (
        "Qwen3-14B",
    ),
}


def safe_model_name(model: str) -> str:
    import re

    name = re.sub(r"[^A-Za-z0-9._-]+", "_", model.strip())
    return name or "model"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def normalize_model_name(model: str) -> str | None:
    model_text = str(model or "")
    for canonical, aliases in MODEL_ALIASES.items():
        if any(alias in model_text for alias in aliases):
            return canonical
    return None


def prompt_sort_key(prompt_mode: str) -> int:
    try:
        return PROMPT_ORDER.index(prompt_mode)
    except ValueError:
        return len(PROMPT_ORDER)


def find_result_files(result_dir: Path) -> list[Path]:
    files = []
    for path in result_dir.glob("*_latency.json"):
        if path.name == "rq3_batch_latency_summary.json":
            continue
        files.append(path)
    return sorted(files)


def get_nested(obj: dict[str, Any], path: tuple[str, ...], default: Any = "") -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def round_value(value: Any, ndigits: int = 4) -> Any:
    if isinstance(value, (int, float)):
        return round(float(value), ndigits)
    return value


def build_row(path: Path, summary: dict[str, Any]) -> dict[str, Any] | None:
    canonical_model = normalize_model_name(str(summary.get("model", "")))
    if canonical_model is None:
        return None

    prompt_mode = str(summary.get("prompt_mode", "")).strip()
    latency = summary.get("latency", {}) if isinstance(summary.get("latency"), dict) else {}
    load_memory = summary.get("gpu_memory_after_model_load", {})
    peak_memory = summary.get("gpu_memory_eval_peak", {})

    completed = int(summary.get("completed") or 0)
    success = int(summary.get("success") or 0)
    failed = int(summary.get("failed") or 0)
    success_rate = success / completed if completed else 0.0

    return {
        "model": canonical_model,
        "prompt_mode": prompt_mode,
        "status": summary.get("status", ""),
        "completed": completed,
        "success": success,
        "failed": failed,
        "success_rate": round(success_rate, 6),
        "mean_latency_s": round_value(latency.get("mean_seconds", ""), 6),
        "median_latency_s": round_value(latency.get("median_seconds", ""), 6),
        "p90_latency_s": round_value(latency.get("p90_seconds", ""), 6),
        "p95_latency_s": round_value(latency.get("p95_seconds", ""), 6),
        "p99_latency_s": round_value(latency.get("p99_seconds", ""), 6),
        "min_latency_s": round_value(latency.get("min_seconds", ""), 6),
        "max_latency_s": round_value(latency.get("max_seconds", ""), 6),
        "std_latency_s": round_value(latency.get("std_seconds", ""), 6),
        "alerts_per_second_by_mean": round_value(latency.get("alerts_per_second_by_mean", ""), 6),
        "alerts_per_day_by_mean": round_value(latency.get("alerts_per_day_by_mean", ""), 2),
        "alerts_per_day_by_p95": round_value(latency.get("alerts_per_day_by_p95", ""), 2),
        "model_load_seconds": round_value(summary.get("model_load_seconds", ""), 6),
        "gpu_after_load_allocated_mb": round_value(get_nested(load_memory, ("allocated_mb",)), 3),
        "gpu_after_load_reserved_mb": round_value(get_nested(load_memory, ("reserved_mb",)), 3),
        "gpu_eval_peak_allocated_mb": round_value(get_nested(peak_memory, ("max_allocated_mb",)), 3),
        "gpu_eval_peak_reserved_mb": round_value(get_nested(peak_memory, ("max_reserved_mb",)), 3),
        "max_input_tokens": get_nested(summary, ("generation", "max_input_tokens")),
        "max_new_tokens": get_nested(summary, ("generation", "max_new_tokens")),
        "source_file": str(path),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("no rows to write")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "model",
        "prompt",
        "mean(s)",
        "p95(s)",
        "alerts/day",
        "peak_mem(MB)",
        "status",
    ]
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row["model"],
                row["prompt_mode"],
                row["mean_latency_s"],
                row["p95_latency_s"],
                row["alerts_per_day_by_mean"],
                row["gpu_eval_peak_allocated_mb"],
                row["status"],
            ]
        )

    widths = [len(h) for h in headers]
    for tr in table_rows:
        for i, val in enumerate(tr):
            widths[i] = max(widths[i], len(str(val)))

    print(" | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("-+-".join("-" * w for w in widths))
    for tr in table_rows:
        print(" | ".join(str(val).ljust(widths[i]) for i, val in enumerate(tr)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize RQ3 latency results for selected Qwen3 models")
    parser.add_argument("--result-dir", type=Path, default=None, help="Directory containing RQ3 latency result JSON files")
    parser.add_argument("--out-dir", type=Path, default=None, help="Directory for summary CSV and JSON outputs")
    parser.add_argument("--out-prefix", default="rq3_selected_qwen_latency_stats")
    args = parser.parse_args()

    required_args = {
        "--result-dir": args.result_dir,
        "--out-dir": args.out_dir,
    }
    missing = [name for name, value in required_args.items() if value is None]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")

    rows: list[dict[str, Any]] = []
    for path in find_result_files(args.result_dir):
        data = load_json(path)
        summary = data.get("summary", {})
        if not isinstance(summary, dict):
            continue
        row = build_row(path, summary)
        if row is not None:
            rows.append(row)

    rows.sort(key=lambda r: (r["model"], prompt_sort_key(str(r["prompt_mode"]))))

    if not rows:
        raise SystemExit(f"No selected Qwen3 latency result files found in {args.result_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / f"{args.out_prefix}.csv"
    out_json = args.out_dir / f"{args.out_prefix}.json"

    write_csv(out_csv, rows)
    atomic_write_json(
        out_json,
        {
            "result_dir": str(args.result_dir),
            "selected_models": list(MODEL_ALIASES.keys()),
            "row_count": len(rows),
            "rows": rows,
        },
    )

    print_table(rows)
    print(f"csv_file={out_csv}")
    print(f"json_file={out_json}")


if __name__ == "__main__":
    main()
