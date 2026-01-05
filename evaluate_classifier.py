#!/usr/bin/env python3
"""
Evaluation script for the AI Third Umpire LBW classification system.

This script reads a dataset containing the ground-truth LBW decisions and the
model predictions, computes a complete set of classification metrics, and
generates a confusion-matrix visualization that can be included in technical
reports or research papers.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


POSITIVE_LABEL = "OUT"
NEGATIVE_LABEL = "NOT OUT"


def safe_divide(numerator: float, denominator: float) -> float:
    """Return numerator / denominator, guarding against division-by-zero."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


@dataclass
class ClassificationMetrics:
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1_score: float
    false_positive_rate: float
    false_negative_rate: float

    def as_pretty_dict(self) -> Dict[str, str | float | int]:
        """Return metrics using pretty labels for reporting."""
        return {
            "True Positive (TP)": self.true_positive,
            "True Negative (TN)": self.true_negative,
            "False Positive (FP)": self.false_positive,
            "False Negative (FN)": self.false_negative,
            "Accuracy": round(self.accuracy, 4),
            "Precision": round(self.precision, 4),
            "Recall / Sensitivity": round(self.recall, 4),
            "Specificity": round(self.specificity, 4),
            "F1-Score": round(self.f1_score, 4),
            "False Positive Rate (FPR)": round(self.false_positive_rate, 4),
            "False Negative Rate (FNR)": round(self.false_negative_rate, 4),
        }


def compute_metrics(actual: List[str], predicted: List[str]) -> ClassificationMetrics:
    """Compute confusion-matrix components and derived metrics."""
    labels = [POSITIVE_LABEL, NEGATIVE_LABEL]
    cm = confusion_matrix(actual, predicted, labels=labels)
    tp, fn, fp, tn = (
        int(cm[0, 0]),
        int(cm[0, 1]),
        int(cm[1, 0]),
        int(cm[1, 1]),
    )

    total = tp + tn + fp + fn
    accuracy = safe_divide(tp + tn, total)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    f1_score = safe_divide(2 * precision * recall, precision + recall)
    fpr = safe_divide(fp, fp + tn)
    fnr = safe_divide(fn, fn + tp)

    return ClassificationMetrics(
        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1_score=f1_score,
        false_positive_rate=fpr,
        false_negative_rate=fnr,
    )


def plot_confusion_matrix(metrics: ClassificationMetrics, output_path: Path) -> None:
    """Generate and save a confusion matrix heatmap."""
    cm = np.array(
        [
            [metrics.true_positive, metrics.false_negative],
            [metrics.false_positive, metrics.true_negative],
        ]
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=[POSITIVE_LABEL, NEGATIVE_LABEL],
        yticklabels=[POSITIVE_LABEL, NEGATIVE_LABEL],
        ylabel="Actual",
        xlabel="Predicted",
        title="LBW Classification Confusion Matrix",
    )

    # Annotate each cell with value
    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=12,
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_metrics_json(metrics: ClassificationMetrics, output_path: Path) -> None:
    """Save metrics in JSON format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics.as_pretty_dict(), f, indent=2)


def evaluate(data_file: Path, output_dir: Path) -> ClassificationMetrics:
    """Load dataset, compute metrics, and generate artifacts."""
    df = pd.read_csv(data_file)
    if "actual" not in df.columns or "predicted" not in df.columns:
        raise ValueError("Input CSV must contain 'actual' and 'predicted' columns.")

    actual = df["actual"].str.upper().str.strip().tolist()
    predicted = df["predicted"].str.upper().str.strip().tolist()

    metrics = compute_metrics(actual, predicted)

    # Save visualizations and summaries
    plot_confusion_matrix(metrics, output_dir / "confusion_matrix.png")
    save_metrics_json(metrics, output_dir / "metrics_summary.json")

    # Print summary table
    print("\nLBW Classification Metrics")
    print("=" * 32)
    for key, value in metrics.as_pretty_dict().items():
        print(f"{key:<30}: {value}")
    print(f"\nSaved confusion matrix plot to: {output_dir / 'confusion_matrix.png'}")
    print(f"Saved metrics summary to: {output_dir / 'metrics_summary.json'}\n")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LBW classifier predictions and compute metrics."
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        required=True,
        help="CSV file with columns 'actual' and 'predicted' (values: OUT / NOT OUT)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluation"),
        help="Directory to store plots and metric summaries",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(args.data_file, args.output_dir)


if __name__ == "__main__":
    main()


