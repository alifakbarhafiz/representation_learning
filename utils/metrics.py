"""Metrics and pretty-printing utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Compute accuracy, macro F1, and per-class accuracy.

    Args:
        y_true: Shape (N,)
        y_pred: Shape (N,)
    """

    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))

    classes = np.unique(y_true)
    per_class_acc: dict[int, float] = {}
    for c in classes:
        mask = y_true == c
        per_class_acc[int(c)] = float((y_pred[mask] == y_true[mask]).mean()) if mask.any() else 0.0

    return {"accuracy": acc, "macro_f1": macro_f1, "per_class_acc": per_class_acc}


def print_metrics_table(results_dict: dict[str, dict[str, Any]]) -> None:
    """Pretty print method->metrics comparison table.

    Expects each metrics dict to include:
      - accuracy (float)
      - macro_f1 (float)
      - optional: train_time (float), feature_time (float)
    """

    methods = list(results_dict.keys())
    headers = ["Method", "Test Acc", "Macro F1", "Train Time", "Feature Time"]
    rows: list[list[str]] = []

    for m in methods:
        d = results_dict[m]
        acc = d.get("accuracy", None)
        f1 = d.get("macro_f1", None)
        train_time = d.get("train_time", None)
        feat_time = d.get("feature_time", None)
        rows.append(
            [
                str(m),
                f"{acc * 100:.2f}%" if isinstance(acc, (int, float)) else "-",
                f"{f1:.4f}" if isinstance(f1, (int, float)) else "-",
                f"{train_time:.2f}s" if isinstance(train_time, (int, float)) else "-",
                f"{feat_time:.2f}s" if isinstance(feat_time, (int, float)) else "-",
            ]
        )

    col_widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def fmt_row(cols: list[str]) -> str:
        return "│ " + " │ ".join(c.ljust(col_widths[i]) for i, c in enumerate(cols)) + " │"

    top = "┌" + "┬".join("─" * (w + 2) for w in col_widths) + "┐"
    mid = "├" + "┼".join("─" * (w + 2) for w in col_widths) + "┤"
    bot = "└" + "┴".join("─" * (w + 2) for w in col_widths) + "┘"

    print(top)
    print(fmt_row(headers))
    print(mid)
    for r in rows:
        print(fmt_row(r))
    print(bot)


if __name__ == "__main__":
    y_t = np.array([0, 1, 1, 2, 2, 2])
    y_p = np.array([0, 1, 0, 2, 1, 2])
    print(compute_metrics(y_t, y_p))
    print_metrics_table({"Demo": {"accuracy": 0.5, "macro_f1": 0.4, "train_time": 1.2, "feature_time": 0.3}})

