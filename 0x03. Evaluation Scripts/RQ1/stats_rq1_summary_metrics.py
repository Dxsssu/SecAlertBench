#!/usr/bin/env python3
"""Summarize RQ1 model metrics from result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


COLUMNS = [
    "model",
    "accuracy",
    "precision",
    "recall_tpr",
    "fpr",
    "f1_score",
]


def fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [r.get(key) for r in rows if isinstance(r.get(key), (int, float))]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def build_avg_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model": "AVG",
        "accuracy": mean_metric(rows, "accuracy"),
        "precision": mean_metric(rows, "precision"),
        "recall_tpr": mean_metric(rows, "recall_tpr"),
        "fpr": mean_metric(rows, "fpr"),
        "f1_score": mean_metric(rows, "f1_score"),
    }


def load_one(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None
    summary = data.get("summary")
    if not isinstance(summary, dict):
        return None
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        return None

    model = summary.get("model", path.stem)
    return {
        "model": model,
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall_tpr": metrics.get("recall_tpr"),
        "fpr": metrics.get("fpr"),
        "f1_score": metrics.get("f1_score"),
    }


def print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No valid result JSON found.")
        return

    headers = ["Model", "Accuracy", "Precision", "Recall(TPR)", "FPR", "F1-score"]
    avg_row = build_avg_row(rows)

    table_rows = []
    for r in rows + [avg_row]:
        table_rows.append(
            [
                fmt(r["model"]),
                fmt(r["accuracy"]),
                fmt(r["precision"]),
                fmt(r["recall_tpr"]),
                fmt(r["fpr"]),
                fmt(r["f1_score"]),
            ]
        )

    widths = [len(h) for h in headers]
    for row in table_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(sep: str = "-") -> str:
        return "+" + "+".join(sep * (w + 2) for w in widths) + "+"

    print(line("-"))
    print("| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |")
    print(line("="))
    for i, row in enumerate(table_rows):
        print("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |")
        if i == len(table_rows) - 2:
            print(line("-"))
    print(line("-"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Show RQ1 summary metrics table from result JSON files")
    parser.add_argument(
        "--results-dir",
        default="",
        help="Directory containing model result JSON files",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Output JSON file path for RQ1 summary metrics",
    )
    args = parser.parse_args()

    required_args = {
        "--results-dir": args.results_dir,
        "--json-out": args.json_out,
    }
    missing = [name for name, value in required_args.items() if not value]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")

    results_dir = Path(args.results_dir)
    if not results_dir.exists() or not results_dir.is_dir():
        raise ValueError(f"Invalid results dir: {results_dir}")

    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        row = load_one(path)
        if row is not None:
            rows.append(row)

    rows.sort(key=lambda x: str(x["model"]))
    print_table(rows)
    avg_row = build_avg_row(rows)

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "task": "RQ1",
                "source_dir": str(results_dir.resolve()),
                "count_models": len(rows),
                "metrics": rows,
                "avg_metrics": avg_row,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"json_out={out_path}")


if __name__ == "__main__":
    main()
