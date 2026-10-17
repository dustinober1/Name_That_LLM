"""Unit tests for competition metrics."""

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score

from name_that_llm.config import TARGET_MODELS
from name_that_llm.metrics import compute_balanced_accuracy, compute_metrics


def test_compute_balanced_accuracy_perfect():
    y_true = np.array(["gpt-5.5", "claude-sonnet-5", "gemini-3.6-flash"])
    y_pred = np.array(["gpt-5.5", "claude-sonnet-5", "gemini-3.6-flash"])
    assert compute_balanced_accuracy(y_true, y_pred) == 1.0


def test_compute_balanced_accuracy_unbalanced():
    # Model A: 10 samples (10 correct) -> recall 1.0
    # Model B: 1 sample (0 correct)     -> recall 0.0
    # Standard accuracy = 10 / 11 = 90.9%
    # Balanced accuracy = (1.0 + 0.0) / 2 = 50.0%
    y_true = np.array(["gpt-5.5"] * 10 + ["claude-sonnet-5"])
    y_pred = np.array(["gpt-5.5"] * 11)
    acc = float(np.mean(y_true == y_pred))
    bal_acc = compute_balanced_accuracy(y_true, y_pred)
    assert acc > 0.9
    assert pytest.approx(bal_acc, 0.001) == 0.5


def test_compute_metrics_dictionary_keys():
    y_true = np.array(["gpt-5.5", "claude-sonnet-5"])
    y_pred = np.array(["gpt-5.5", "claude-sonnet-5"])
    m = compute_metrics(y_true, y_pred)
    assert "balanced_accuracy" in m
    assert "macro_f1" in m
    assert "per_class_recalls" in m
    assert "provider_balanced_accuracy" in m
