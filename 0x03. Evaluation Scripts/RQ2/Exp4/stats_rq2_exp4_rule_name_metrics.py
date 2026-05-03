#!/usr/bin/env python3
"""Compute per-attack-type metrics for selected RQ1 model results.

The script first counts attack_type frequencies across the selected result files,
keeps the top-K attack types, and then computes TPR/FPR/F1 for each model on each
selected attack_type.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


VALID_LABELS = ("Attack", "Non-Attack")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"{path} must be a JSON object")
    return obj


def safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def get_attack_type(item: dict[str, Any]) -> str:
    record = item.get("record", {})
    if not isinstance(record, dict):
        return ""
    return str(record.get("attack_type") or "").strip() or "<EMPTY_ATTACK_TYPE>"


def compute_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for item in items:
        true_label = item.get("true_label")
        pred_label = item.get("llm_label")
        if true_label not in VALID_LABELS or pred_label not in VALID_LABELS:
            continue

        if true_label == "Attack":
            if pred_label == "Attack":
                tp += 1
            else:
                fn += 1
        else:
            if pred_label == "Attack":
                fp += 1
            else:
                tn += 1

    precision = safe_div(tp, tp + fp)
    tpr = safe_div(tp, tp + fn)
    fpr = safe_div(fp, fp + tn)
    f1 = safe_div(2 * precision * tpr, precision + tpr)
    return {
        "TPR": round(tpr, 6),
        "FPR": round(fpr, 6),
        "F1": round(f1, 6),
    }




def load_model_file_map(value: str) -> dict[str, str]:
    path = Path(value)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    else:
        obj = json.loads(value)

    if not isinstance(obj, dict):
        raise ValueError("model-files must be a JSON object mapping model names to result filenames")
    out: dict[str, str] = {}
    for model, filename in obj.items():
        if not isinstance(model, str) or not isinstance(filename, str):
            raise ValueError("model-files keys and values must be strings")
        out[model] = filename
    if not out:
        raise ValueError("model-files mapping is empty")
    return out

def load_model_results(rq1_dir: Path, model_files: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for model, filename in model_files.items():
        path = rq1_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing result file for {model}: {path}")
        obj = load_json(path)
        results = obj.get("results", [])
        if not isinstance(results, list):
            raise ValueError(f"{path}: results must be a list")
        out[model] = [item for item in results if isinstance(item, dict)]
    return out


def build_summary(model_results: dict[str, list[dict[str, Any]]], top_k: int) -> dict[str, Any]:
    attack_type_counter: Counter[str] = Counter()
    attack_type_label_counter: dict[str, Counter[str]] = {}

    for results in model_results.values():
        counter = Counter(get_attack_type(item) for item in results)
        attack_type_counter.update(counter)
        for item in results:
            attack_type = get_attack_type(item)
            true_label = item.get("true_label")
            if true_label in VALID_LABELS:
                attack_type_label_counter.setdefault(attack_type, Counter())[true_label] += 1

    attack_type_infos = []
    for attack_type, count in attack_type_counter.items():
        label_counts = attack_type_label_counter.get(attack_type, Counter())
        attack_count = label_counts.get("Attack", 0)
        non_attack_count = label_counts.get("Non-Attack", 0)
        attack_type_infos.append(
            {
                "attack_type": attack_type,
                "total_count_across_models": count,
                "attack_count": attack_count,
                "non_attack_count": non_attack_count,
                "balanced_count": min(attack_count, non_attack_count),
            }
        )

    top_attack_types = sorted(
        attack_type_infos,
        key=lambda x: (
            x["balanced_count"],
            x["total_count_across_models"],
            x["attack_count"] + x["non_attack_count"],
        ),
        reverse=True,
    )[:top_k]

    rows: list[dict[str, Any]] = []
    for attack_type_info in top_attack_types:
        attack_type = attack_type_info["attack_type"]
        for model, results in model_results.items():
            items = [item for item in results if get_attack_type(item) == attack_type]
            metrics = compute_metrics(items)
            rows.append(
                {
                    "attack_type": attack_type,
                    "balanced_count": attack_type_info["balanced_count"],
                    "top_attack_type_total_count_across_models": attack_type_info["total_count_across_models"],
                    "attack_count": attack_type_info["attack_count"],
                    "non_attack_count": attack_type_info["non_attack_count"],
                    "model": model,
                    **metrics,
                }
            )

    return {
        "top_k": top_k,
        "models": list(model_results.keys()),
        "top_attack_types": top_attack_types,
        "rows": rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "attack_type",
        "balanced_count",
        "top_attack_type_total_count_across_models",
        "attack_count",
        "non_attack_count",
        "model",
        "TPR",
        "FPR",
        "F1",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ2 Exp4 attack_type Top-K metrics by selected RQ1 models")
    parser.add_argument("--rq1-dir", default="", help="Directory containing RQ1 model result JSON files")
    parser.add_argument(
        "--model-files",
        default="",
        help="JSON file path or inline JSON object mapping display model names to RQ1 result filenames",
    )
    parser.add_argument("--out-dir", default="", help="Directory for RQ2 Exp4 attack-type metric outputs")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    required_args = {
        "--rq1-dir": args.rq1_dir,
        "--model-files": args.model_files,
        "--out-dir": args.out_dir,
    }
    missing = [name for name, value in required_args.items() if not value]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")
    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")

    rq1_dir = Path(args.rq1_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_files = load_model_file_map(args.model_files)

    model_results = load_model_results(rq1_dir, model_files)
    summary = build_summary(model_results, args.top_k)

    out_json = out_dir / f"top{args.top_k}_attack_type_metrics.json"
    out_csv = out_dir / f"top{args.top_k}_attack_type_metrics.csv"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_csv(out_csv, summary["rows"])

    print(json.dumps({"out_json": str(out_json), "out_csv": str(out_csv), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
