#!/usr/bin/env python3
"""Compute TPR/FPR grids for two sampling models and draw a 2x2 comparison figure.

Layout:
- Top-left:  Model 1 TPR
- Top-right: Model 1 FPR
- Bottom-left:  Model 2 TPR
- Bottom-right: Model 2 FPR
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import colors
import numpy as np

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"


@dataclass
class MetricCell:
    tpr: float
    fpr: float
    tp: int
    fp: int
    tn: int
    fn: int


COLOR_THEME = "green"  # "green" or "red"
PANEL_COLOR_PADDING_RATIO = 0


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"{path} is not a JSON object")
    return obj


def sorted_unique_float(values: list[float]) -> list[float]:
    return sorted(set(float(v) for v in values))


def fmt_float(v: float) -> str:
    return f"{v:g}"


def collect_metrics(result_dir: Path) -> dict[str, dict[tuple[float, float], MetricCell]]:
    out: dict[str, dict[tuple[float, float], MetricCell]] = {}

    for path in sorted(result_dir.glob("*_temp_*_topp_*.json")):
        if path.name.endswith(".bak"):
            continue

        obj = load_json(path)
        summary = obj.get("summary", {})
        if not isinstance(summary, dict):
            continue

        model = summary.get("model")
        temperature = summary.get("temperature")
        top_p = summary.get("top_p")
        metrics = summary.get("metrics", {})
        cm = metrics.get("confusion_matrix", {}) if isinstance(metrics, dict) else {}

        if not isinstance(model, str):
            continue
        if temperature is None or top_p is None:
            continue
        if not isinstance(metrics, dict) or not isinstance(cm, dict):
            continue

        try:
            tpr = float(metrics.get("recall_tpr"))
            fpr = float(metrics.get("fpr"))
            tp = int(cm.get("TP"))
            fp = int(cm.get("FP"))
            tn = int(cm.get("TN"))
            fn = int(cm.get("FN"))
        except Exception:
            continue

        out.setdefault(model, {})[(float(temperature), float(top_p))] = MetricCell(
            tpr=tpr,
            fpr=fpr,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
        )

    return out


def build_matrix(
    model_map: dict[tuple[float, float], MetricCell],
    temperatures: list[float],
    top_ps: list[float],
    metric: str,
) -> tuple[np.ndarray, list[list[str]]]:
    mat = np.full((len(temperatures), len(top_ps)), np.nan, dtype=float)
    annotations = [["-" for _ in top_ps] for _ in temperatures]

    for i, t in enumerate(temperatures):
        for j, p in enumerate(top_ps):
            cell = model_map.get((t, p))
            if cell is None:
                continue

            if metric == "tpr":
                value = cell.tpr
                denom = cell.tp + cell.fn
                numer = cell.tp
            else:
                value = cell.fpr

            mat[i, j] = value
            annotations[i][j] = f"{value:.3f}"

    return mat, annotations


def build_colormap(theme: str) -> colors.LinearSegmentedColormap:
    if theme == "green":
        return colors.LinearSegmentedColormap.from_list(
            "soft_green",
            ["#e5f9e4", "#b9f2b2", "#70e875"],
        )
    if theme == "red":
        return colors.LinearSegmentedColormap.from_list(
            "soft_red",
            ["#ffe7e5", "#ffb6b0", "#ff7878"],
        )
    raise ValueError(f"Unsupported COLOR_THEME: {theme}. Use 'green' or 'red'.")


def panel_color_limits(mat: np.ndarray) -> tuple[float, float]:
    finite_values = mat[np.isfinite(mat)]
    if finite_values.size == 0:
        return 0.0, 1.0

    vmin = float(np.min(finite_values))
    vmax = float(np.max(finite_values))
    if vmin == vmax:
        pad = 0.05
    else:
        pad = (vmax - vmin) * PANEL_COLOR_PADDING_RATIO
    return max(0.0, vmin - pad), min(1.0, vmax + pad)


def draw_metric_table(
    ax: Any,
    mat: np.ndarray,
    annotations: list[list[str]],
    temperatures: list[float],
    top_ps: list[float],
    title: str,
    vmin: float,
    vmax: float,
    higher_is_better: bool,
) -> None:
    rows = len(temperatures)
    cols = len(top_ps)
    cmap = build_colormap(COLOR_THEME)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    header_color = "#c8c8c8"
    cell_gap = 0.08

    ax.set_xlim(-1.25, cols)
    ax.set_ylim(-0.95, rows + 1.35)
    ax.axis("off")

    ax.text(
        cols / 2,
        rows + 1.03,
        "top_p",
        ha="center",
        va="center",
        fontsize=8.8,
        fontweight="bold",
    )
    ax.text(
        -1.15,
        rows / 2,
        "temperature",
        ha="center",
        va="center",
        rotation=90,
        fontsize=8.2,
        fontweight="bold",
    )

    for j, top_p in enumerate(top_ps):
        ax.add_patch(
            Rectangle(
                (j + cell_gap / 2, rows + cell_gap / 2),
                1 - cell_gap,
                0.72,
                facecolor=header_color,
                edgecolor="white",
                linewidth=1.0,
            )
        )
        ax.text(j + 0.5, rows + 0.36, fmt_float(top_p), ha="center", va="center", fontsize=8.0, fontweight="bold")

    for i, temperature in enumerate(temperatures):
        y = rows - 1 - i
        ax.add_patch(
            Rectangle(
                (-0.85 + cell_gap / 2, y + cell_gap / 2),
                0.75,
                1 - cell_gap,
                facecolor=header_color,
                edgecolor="white",
                linewidth=1.0,
            )
        )
        ax.text(-0.47, y + 0.5, fmt_float(temperature), ha="center", va="center", fontsize=8.0, fontweight="bold")

    for i in range(rows):
        for j in range(cols):
            y = rows - 1 - i
            value = mat[i, j]
            if not np.isfinite(value):
                ax.text(j + 0.5, y + 0.5, "-", ha="center", va="center", fontsize=10.0)
                continue
            ax.add_patch(
                Rectangle(
                    (j + cell_gap / 2, y + cell_gap / 2),
                    1 - cell_gap,
                    1 - cell_gap,
                    facecolor=cmap(norm(value) if higher_is_better else 1.0 - norm(value)),
                    edgecolor="white",
                    linewidth=1.0,
                )
            )
            ax.text(j + 0.5, y + 0.5, annotations[i][j], ha="center", va="center", fontsize=9.4, fontweight="bold")

    ax.text(
        cols / 2,
        -0.28,
        title,
        ha="center",
        va="center",
        fontsize=9.6,
        fontweight="bold",
    )


def choose_models(all_models: list[str], model_arg: str) -> list[str]:
    if model_arg.strip():
        models = [x.strip() for x in model_arg.split(",") if x.strip()]
        if len(models) != 2:
            raise ValueError("--models must contain exactly two model names separated by comma")
        missing = [m for m in models if m not in all_models]
        if missing:
            raise ValueError(f"Model(s) not found in results: {missing}. Available: {all_models}")
        return models

    if len(all_models) < 2:
        raise ValueError(f"Need at least 2 models, found: {all_models}")
    if len(all_models) > 2:
        raise ValueError(
            "Found more than 2 models. Please set --models explicitly, for example: "
            f"--models {all_models[0]},{all_models[1]}"
        )
    return all_models


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ2 Exp3 sampling TPR/FPR statistics and plotting")
    parser.add_argument("--result-dir", default="", help="Directory containing sampling experiment result JSON files")
    parser.add_argument("--out-pdf", default="", help="Output PDF file path for the TPR/FPR grid figure")
    parser.add_argument(
        "--models",
        default="",
        help="Exactly two model names separated by comma. If omitted, auto-uses exactly two discovered models.",
    )
    args = parser.parse_args()

    required_args = {
        "--result-dir": args.result_dir,
        "--out-pdf": args.out_pdf,
    }
    missing = [name for name, value in required_args.items() if not value]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")

    result_dir = Path(args.result_dir)
    out_pdf = Path(args.out_pdf)

    data = collect_metrics(result_dir)
    all_models = sorted(data.keys())
    models = choose_models(all_models, args.models)

    temperatures = sorted_unique_float(
        [t for m in models for (t, _p) in data[m].keys()]
    )
    top_ps = sorted_unique_float(
        [p for m in models for (_t, p) in data[m].keys()]
    )

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.8))

    for row, model in enumerate(models):
        tpr_mat, tpr_ann = build_matrix(data[model], temperatures, top_ps, metric="tpr")
        fpr_mat, fpr_ann = build_matrix(data[model], temperatures, top_ps, metric="fpr")
        tpr_vmin, tpr_vmax = panel_color_limits(tpr_mat)
        fpr_vmin, fpr_vmax = panel_color_limits(fpr_mat)

        draw_metric_table(
            axes[row, 0],
            tpr_mat,
            tpr_ann,
            temperatures,
            top_ps,
            title=f"{model} - TPR",
            vmin=tpr_vmin,
            vmax=tpr_vmax,
            higher_is_better=True,
        )
        draw_metric_table(
            axes[row, 1],
            fpr_mat,
            fpr_ann,
            temperatures,
            top_ps,
            title=f"{model} - FPR",
            vmin=fpr_vmin,
            vmax=fpr_vmax,
            higher_is_better=False,
        )

    plt.subplots_adjust(wspace=0, hspace=0)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"models={models}")
    print(f"temperatures={temperatures}")
    print(f"top_ps={top_ps}")
    print(f"figure_file={out_pdf}")


if __name__ == "__main__":
    main()
