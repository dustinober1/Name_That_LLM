"""Evaluation metrics for the 'Name That LLM' competition.

The official Kaggle evaluation metric is Balanced Accuracy:
the macro-average of recall across all individual model classes.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)

from name_that_llm.config import MODEL_TO_PROVIDER, TARGET_MODELS


def compute_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the official competition evaluation metric: Balanced Accuracy."""
    return float(balanced_accuracy_score(y_true, y_pred))


def compute_provider_balanced_accuracy(
    y_true_models: List[str], y_pred_models: List[str]
) -> float:
    """Compute balanced accuracy at the provider / lab level (e.g. OpenAI vs Anthropic)."""
    true_providers = [MODEL_TO_PROVIDER.get(m, "unknown") for m in y_true_models]
    pred_providers = [MODEL_TO_PROVIDER.get(m, "unknown") for m in y_pred_models]
    return float(balanced_accuracy_score(true_providers, pred_providers))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute comprehensive performance metrics.

    Args:
        y_true: Ground truth target labels (strings or ints).
        y_pred: Predicted target labels (strings or ints).
        y_prob: Predicted probability distribution array of shape (N, num_classes).
        class_labels: Names of classes corresponding to probability indices.

    Returns:
        Dictionary of computed metrics including balanced accuracy, macro F1, and per-class recall.
    """
    labels = class_labels or TARGET_MODELS
    bal_acc = compute_balanced_accuracy(y_true, y_pred)
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    metrics: Dict[str, Any] = {
        "balanced_accuracy": bal_acc,
        "standard_accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }

    # If true and pred are model string names, compute provider-level accuracy
    if len(y_true) > 0 and isinstance(y_true[0], str):
        metrics["provider_balanced_accuracy"] = compute_provider_balanced_accuracy(
            list(y_true), list(y_pred)
        )

    # Per-class recall (since balanced accuracy is the unweighted mean of per-class recalls)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class_recalls = {}
    for idx, label in enumerate(labels):
        total = cm[idx].sum()
        correct = cm[idx, idx]
        recall = float(correct / total) if total > 0 else 0.0
        per_class_recalls[label] = {
            "recall": recall,
            "correct": int(correct),
            "total": int(total),
        }
    metrics["per_class_recalls"] = per_class_recalls
    metrics["confusion_matrix"] = cm

    if y_prob is not None:
        try:
            # Top-3 accuracy
            top3_preds = np.argsort(y_prob, axis=1)[:, -3:]
            if isinstance(y_true[0], (int, np.integer)):
                top3_correct = np.any(top3_preds == np.array(y_true)[:, None], axis=1)
            else:
                label_to_idx = {l: i for i, l in enumerate(labels)}
                y_true_indices = np.array([label_to_idx.get(val, -1) for val in y_true])
                top3_correct = np.any(top3_preds == y_true_indices[:, None], axis=1)
            metrics["top_3_accuracy"] = float(np.mean(top3_correct))
        except Exception:
            pass

        try:
            metrics["log_loss"] = float(log_loss(y_true, y_prob, labels=labels))
        except Exception:
            pass

    return metrics


def print_metrics_summary(metrics: Dict[str, Any]) -> None:
    """Print an aesthetic summary of the metrics."""
    print("=" * 65)
    print("  COMPETITION METRIC SUMMARY (Balanced Accuracy)")
    print("=" * 65)
    print(f"  ★ Balanced Accuracy (Official Metric) : {metrics['balanced_accuracy']:.4f}")
    print(f"  - Standard Accuracy                   : {metrics['standard_accuracy']:.4f}")
    print(f"  - Macro F1 Score                      : {metrics['macro_f1']:.4f}")
    if "provider_balanced_accuracy" in metrics:
        print(f"  - Provider-Level Balanced Accuracy    : {metrics['provider_balanced_accuracy']:.4f}")
    if "top_3_accuracy" in metrics:
        print(f"  - Top-3 Accuracy                      : {metrics['top_3_accuracy']:.4f}")
    if "log_loss" in metrics:
        print(f"  - Multi-class Log Loss                : {metrics['log_loss']:.4f}")
    print("-" * 65)
    print("  Per-Class Recall Breakdown (Ranked lowest to highest):")
    recalls = metrics.get("per_class_recalls", {})
    sorted_recalls = sorted(recalls.items(), key=lambda item: item[1]["recall"])
    for name, stats in sorted_recalls:
        bar = "█" * int(stats["recall"] * 20) + "░" * (20 - int(stats["recall"] * 20))
        print(f"    {name:<24} [{bar}] {stats['recall']:.2f} ({stats['correct']}/{stats['total']})")
    print("=" * 65)
