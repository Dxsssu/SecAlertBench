#!/usr/bin/env python3
"""Summarize TPR/FPR/F1 across quantization modes for RQ2 Exp3."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


QUANTIZATION_ORDER = {"none": 0, "8bit": 1, "4bit": 2}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"{path} must be a JSON object")
    return obj


def short_model_name(model_path: str) -> str:
    return Path(model_path.rstrip("/")).name or model_path


def collect_rows(result_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(result_dir.glob("*__quant_*.json")):
        if path.name.endswith(".bak"):
            continue

        obj = load_json(path)
        summary = obj.get("summary", {})
        if not isinstance(summary, dict):
            continue
        metrics = summary.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        cm = metrics.get("confusion_matrix", {})
        if not isinstance(cm, dict):
            cm = {}

        model_path = str(summary.get("model", ""))
        quantization = str(summary.get("quantization", "unknown"))
        rows.append(
            {
                "model": short_model_name(model_path),
                "model_path": model_path,
                "quantization": quantization,
                "ACC": metrics.get("accuracy"),
                "PRE": metrics.get("precision"),
                "TPR": metrics.get("recall_tpr"),
                "FPR": metrics.get("fpr"),
                "F1": metrics.get("f1_score"),
                "TP": cm.get("TP"),
                "FP": cm.get("FP"),
                "TN": cm.get("TN"),
                "FN": cm.get("FN"),
                "valid_eval_count": metrics.get("valid_eval_count"),
                "result_file": str(path),
            }
        )

    return sorted(
        rows,
        key=lambda x: (
            str(x["model"]).lower(),
            QUANTIZATION_ORDER.get(str(x["quantization"]), 99),
        ),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "model",
        "quantization",
        "ACC",
        "PRE",
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
    parser = argparse.ArgumentParser(description="RQ2 Exp3 quantization metrics summary")
    parser.add_argument("--result-dir", default="", help="Directory containing quantization experiment result JSON files")
    parser.add_argument("--out-json", default="", help="Output JSON file path for the metrics summary")
    parser.add_argument("--out-csv", default="", help="Output CSV file path for the metrics summary")
    args = parser.parse_args()

    required_args = {
        "--result-dir": args.result_dir,
        "--out-json": args.out_json,
        "--out-csv": args.out_csv,
    }
    missing = [name for name, value in required_args.items() if not value]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")

    rows = collect_rows(Path(args.result_dir))
    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_json.open("w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, ensure_ascii=False, indent=2)
    write_csv(out_csv, rows)

    print(json.dumps({"out_json": str(out_json), "out_csv": str(out_csv), "rows": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
