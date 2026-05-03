#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


POS_LABEL = "Attack"
NEG_LABEL = "Non-Attack"


def normalize_label(value) -> Optional[str]:
    if isinstance(value, bool):
        return POS_LABEL if value else NEG_LABEL
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"attack", "attacks", "true", "1", "yes", "y", "malicious"}:
        return POS_LABEL
    if text in {
        "non-attack",
        "non_attack",
        "non attack",
        "benign",
        "false",
        "0",
        "no",
        "n",
        "fp",
        "normal",
    }:
        return NEG_LABEL
    return None


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    tpr = safe_div(tp, tp + fn)
    fpr = safe_div(fp, fp + tn)
    f1 = safe_div(2 * tp, 2 * tp + fp + fn)
    return {"TPR": tpr, "FPR": fpr, "F1": f1}


def infer_model_prompt_from_filename(path: Path) -> Tuple[str, str]:
    stem = path.stem
    known_tags = ["zero_shot", "one_shot", "few_shot", "user-baseline", "cot"]
    for tag in known_tags:
        suffix = f"_{tag}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)], tag
    if "_" in stem:
        model, prompt = stem.rsplit("_", 1)
        return model, prompt
    return stem, "unknown"


def parse_one_file(path: Path) -> Optional[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data.get("summary") or {}
    model = summary.get("model")
    prompt_tag = summary.get("prompt_tag")
    if not model or not prompt_tag:
        fallback_model, fallback_prompt = infer_model_prompt_from_filename(path)
        model = model or fallback_model
        prompt_tag = prompt_tag or fallback_prompt

    results: List[Dict] = data.get("results") or []
    tp = fp = tn = fn = 0
    valid = 0
    for row in results:
        true_label = normalize_label(row.get("true_label"))
        pred_label = normalize_label(row.get("llm_label"))
        if true_label is None or pred_label is None:
            continue
        valid += 1
        if true_label == POS_LABEL and pred_label == POS_LABEL:
            tp += 1
        elif true_label == NEG_LABEL and pred_label == POS_LABEL:
            fp += 1
        elif true_label == NEG_LABEL and pred_label == NEG_LABEL:
            tn += 1
        elif true_label == POS_LABEL and pred_label == NEG_LABEL:
            fn += 1

    if valid == 0:
        conf = (((summary.get("metrics") or {}).get("confusion_matrix")) or {})
        tp = int(conf.get("TP", 0) or 0)
        fp = int(conf.get("FP", 0) or 0)
        tn = int(conf.get("TN", 0) or 0)
        fn = int(conf.get("FN", 0) or 0)
        valid = tp + fp + tn + fn
        if valid == 0:
            return None

    metric_values = metrics_from_counts(tp, fp, tn, fn)
    return {
        "model": model,
        "prompt_tag": prompt_tag,
        "file": str(path),
        "valid_eval_count": valid,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "TPR": metric_values["TPR"],
        "FPR": metric_values["FPR"],
        "F1": metric_values["F1"],
    }


def format_float(x: float) -> str:
    return f"{x:.6f}"


def print_table(rows: List[Dict]) -> None:
    if not rows:
        print("No valid result files found.")
        return
    print(
        "model\tprompt_tag\tTPR\tFPR\tF1\tvalid_eval_count\tTP\tFP\tTN\tFN"
    )
    for r in rows:
        print(
            f"{r['model']}\t{r['prompt_tag']}\t{format_float(r['TPR'])}\t"
            f"{format_float(r['FPR'])}\t{format_float(r['F1'])}\t{r['valid_eval_count']}\t"
            f"{r['TP']}\t{r['FP']}\t{r['TN']}\t{r['FN']}"
        )


def save_json(rows: List[Dict], path: Path) -> None:
    payload = {
        "metric_description": {
            "TPR": "TP / (TP + FN)",
            "FPR": "FP / (FP + TN)",
            "F1": "2*TP / (2*TP + FP + FN)",
            "positive_class": POS_LABEL,
            "negative_class": NEG_LABEL,
        },
        "rows": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_csv(rows: List[Dict], path: Path) -> None:
    fieldnames = [
        "model",
        "prompt_tag",
        "TPR",
        "FPR",
        "F1",
        "valid_eval_count",
        "TP",
        "FP",
        "TN",
        "FN",
        "file",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            row = dict(r)
            row["TPR"] = format_float(row["TPR"])
            row["FPR"] = format_float(row["FPR"])
            row["F1"] = format_float(row["F1"])
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize TPR/FPR/F1 for each model and prompting strategy in RQ2 Exp2"
    )
    parser.add_argument(
        "--results-dir",
        default="",
        help="Directory containing RQ2 Exp2 prompting result files",
    )
    parser.add_argument(
        "--out-json",
        default="",
        help="Output JSON file path for the metrics summary",
    )
    parser.add_argument(
        "--out-csv",
        default="",
        help="Output CSV file path for the metrics summary",
    )
    args = parser.parse_args()

    required_args = {
        "--results-dir": args.results_dir,
        "--out-json": args.out_json,
        "--out-csv": args.out_csv,
    }
    missing = [name for name, value in required_args.items() if not value]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")

    results_dir = Path(args.results_dir)
    files = sorted(
        p
        for p in results_dir.glob("*.json")
        if not p.name.endswith(".bak") and "_failed_" not in p.name
    )

    rows: List[Dict] = []
    for path in files:
        try:
            parsed = parse_one_file(path)
            if parsed:
                rows.append(parsed)
        except Exception as exc:
            print(f"[WARN] skip {path}: {exc}")

    rows.sort(key=lambda x: (x["model"], x["prompt_tag"]))
    print_table(rows)
    save_json(rows, Path(args.out_json))
    save_csv(rows, Path(args.out_csv))
    print(f"\nSaved JSON: {args.out_json}")
    print(f"Saved CSV : {args.out_csv}")


if __name__ == "__main__":
    main()
