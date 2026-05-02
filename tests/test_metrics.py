"""Tests for evaluation metrics helpers."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score

from utils.metrics import compute_metrics, print_metrics_table


def test_compute_metrics_perfect() -> None:
    """Perfect predictions yield perfect scores."""
    y_true = np.array([0, 1, 2, 3, 4] * 4, dtype=np.int64)
    y_pred = y_true.copy()
    res = compute_metrics(y_true, y_pred)
    assert res["accuracy"] == pytest.approx(1.0)
    assert res["macro_f1"] == pytest.approx(1.0)


def test_compute_metrics_random() -> None:
    """Random predictions stay within valid metric ranges."""
    rng = np.random.default_rng(0)
    y_true = np.repeat(np.arange(5, dtype=np.int64), repeats=60)
    y_pred = rng.integers(0, 5, size=y_true.shape, dtype=np.int64)
    res = compute_metrics(y_true, y_pred)
    assert 0.0 <= res["accuracy"] <= 1.0
    assert 0.0 <= res["macro_f1"] <= 1.0


def test_compute_metrics_keys() -> None:
    """compute_metrics should emit the documented keys."""
    y_true = np.repeat(np.arange(5, dtype=np.int64), repeats=40)
    y_pred = y_true.copy()
    res = compute_metrics(y_true, y_pred)
    assert set(res.keys()) == {"accuracy", "macro_f1", "per_class_acc"}


def test_per_class_acc_length() -> None:
    """per_class_acc dictionary should enumerate each observed class."""
    y_true = np.repeat(np.arange(5, dtype=np.int64), repeats=25)
    y_pred = y_true.copy()
    res = compute_metrics(y_true, y_pred)
    assert len(res["per_class_acc"]) == len(np.unique(y_true))


def test_per_class_acc_range() -> None:
    """Per-class accuracies must remain proper fractions."""
    rng = np.random.default_rng(1)
    y_true = np.repeat(np.arange(5, dtype=np.int64), repeats=30)
    y_pred = rng.integers(0, 5, size=y_true.shape, dtype=np.int64)
    res = compute_metrics(y_true, y_pred)
    assert all(0.0 <= float(v) <= 1.0 for v in res["per_class_acc"].values())


def test_accuracy_matches_sklearn() -> None:
    """accuracy should mirror sklearn.metrics.accuracy_score."""
    rng = np.random.default_rng(2)
    y_true = np.repeat(np.arange(5, dtype=np.int64), repeats=50)
    y_pred = rng.integers(0, 5, size=y_true.shape, dtype=np.int64)
    res = compute_metrics(y_true, y_pred)
    assert res["accuracy"] == pytest.approx(float(accuracy_score(y_true, y_pred)))


def test_macro_f1_matches_sklearn() -> None:
    """macro-F1 must match sklearn averaging definitions."""
    rng = np.random.default_rng(3)
    y_true = np.repeat(np.arange(5, dtype=np.int64), repeats=50)
    y_pred = rng.integers(0, 5, size=y_true.shape, dtype=np.int64)
    res = compute_metrics(y_true, y_pred)
    assert res["macro_f1"] == pytest.approx(float(f1_score(y_true, y_pred, average="macro")))


def test_print_metrics_table_runs(capsys: pytest.CaptureFixture[str]) -> None:
    """Table printer should emit non-empty console output."""
    print_metrics_table(
        {
            "PCA": {"accuracy": 0.6, "macro_f1": 0.58, "train_time": 1.23, "feature_time": 0.45},
        }
    )
    out = capsys.readouterr().out.strip()
    assert len(out) > 0
