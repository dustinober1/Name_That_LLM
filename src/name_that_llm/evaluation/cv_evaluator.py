"""Cross-validation runner with leak-free StratifiedGroupKFold, OOF tracking, and metric reporting."""

import copy
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from name_that_llm.config import (
    DEFAULT_N_FOLDS,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    TARGET_MODELS,
)
from name_that_llm.data import get_stratified_group_splits
from name_that_llm.metrics import compute_metrics, print_metrics_summary

console = Console()


class CrossValidationRunner:
    """Manages StratifiedGroupKFold training, out-of-fold predictions, and metric evaluation."""

    def __init__(
        self,
        model_factory: Callable[[], Any],
        n_folds: int = DEFAULT_N_FOLDS,
        seed: int = RANDOM_SEED,
        experiment_name: str = "model",
    ) -> None:
        self.model_factory = model_factory
        self.n_folds = n_folds
        self.seed = seed
        self.experiment_name = experiment_name

        self.fitted_models: List[Any] = []
        self.oof_predictions: Optional[np.ndarray] = None
        self.oof_probabilities: Optional[np.ndarray] = None
        self.metrics_summary: Optional[Dict[str, Any]] = None

    def run(
        self,
        df_train: pd.DataFrame,
        save_oof: bool = True,
    ) -> Dict[str, Any]:
        """Execute cross-validation over the training dataset."""
        n_samples = len(df_train)
        num_classes = len(TARGET_MODELS)

        oof_probs = np.zeros((n_samples, num_classes), dtype=np.float32)
        oof_preds = np.empty(n_samples, dtype=object)

        splits = list(
            get_stratified_group_splits(df_train, n_splits=self.n_folds, seed=self.seed)
        )
        fold_scores: List[float] = []

        console.rule(f"[bold cyan]Starting Cross-Validation: {self.experiment_name}")
        console.print(
            f"Folds: {self.n_folds} | Total Samples: {n_samples} | Grouped on: [yellow]prompt_id[/yellow]"
        )

        for fold, (train_idx, val_idx) in enumerate(splits):
            start_time = time.time()
            train_sub = df_train.iloc[train_idx]
            val_sub = df_train.iloc[val_idx]

            model = self.model_factory()
            model.fit(train_sub["text"].values, train_sub["model"].values)
            self.fitted_models.append(model)

            val_preds = model.predict(val_sub["text"].values)
            oof_preds[val_idx] = val_preds

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(val_sub["text"].values)
                # Map probabilities to official TARGET_MODELS ordering
                model_classes = list(model.classes_)
                class_to_col = {c: i for i, c in enumerate(model_classes)}
                for i, target_cls in enumerate(TARGET_MODELS):
                    if target_cls in class_to_col:
                        oof_probs[val_idx, i] = probs[:, class_to_col[target_cls]]

            fold_metrics = compute_metrics(
                y_true=val_sub["model"].values,
                y_pred=val_preds,
            )
            bal_acc = fold_metrics["balanced_accuracy"]
            fold_scores.append(bal_acc)
            elapsed = time.time() - start_time

            console.print(
                f"  Fold [bold green]{fold + 1}/{self.n_folds}[/bold green] - "
                f"Balanced Acc: [bold yellow]{bal_acc:.4f}[/bold yellow] - "
                f"Elapsed: {elapsed:.1f}s"
            )

        self.oof_predictions = oof_preds
        self.oof_probabilities = oof_probs

        # Overall OOF metrics
        overall_metrics = compute_metrics(
            y_true=df_train["model"].values,
            y_pred=oof_preds,
            y_prob=oof_probs if np.any(oof_probs > 0) else None,
            class_labels=TARGET_MODELS,
        )
        overall_metrics["fold_scores"] = fold_scores
        overall_metrics["mean_fold_score"] = float(np.mean(fold_scores))
        overall_metrics["std_fold_score"] = float(np.std(fold_scores))
        self.metrics_summary = overall_metrics

        print_metrics_summary(overall_metrics)

        if save_oof:
            self._save_oof_data(df_train)

        return overall_metrics

    def predict_test(self, df_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Generate ensembled test predictions across all fitted fold models."""
        if not self.fitted_models:
            raise RuntimeError("Models have not been trained yet. Call .run() first.")

        test_texts = df_test["text"].values
        accumulated_probs = np.zeros((len(df_test), len(TARGET_MODELS)), dtype=np.float32)

        for model in self.fitted_models:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(test_texts)
                model_classes = list(model.classes_)
                class_to_col = {c: i for i, c in enumerate(model_classes)}
                for i, target_cls in enumerate(TARGET_MODELS):
                    if target_cls in class_to_col:
                        accumulated_probs[:, i] += probs[:, class_to_col[target_cls]]
            else:
                preds = model.predict(test_texts)
                for row_idx, p in enumerate(preds):
                    if p in TARGET_MODELS:
                        accumulated_probs[row_idx, TARGET_MODELS.index(p)] += 1.0

        avg_probs = accumulated_probs / len(self.fitted_models)
        pred_indices = np.argmax(avg_probs, axis=1)
        final_preds = np.array([TARGET_MODELS[i] for i in pred_indices])

        return final_preds, avg_probs

    def _save_oof_data(self, df_train: pd.DataFrame) -> None:
        """Persist out-of-fold predictions to data/processed for blending."""
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        oof_df = pd.DataFrame(
            {
                "id": df_train["id"],
                "prompt_id": df_train["prompt_id"],
                "true_model": df_train["model"],
                "pred_model": self.oof_predictions,
            }
        )
        if self.oof_probabilities is not None:
            for idx, col in enumerate(TARGET_MODELS):
                oof_df[f"prob_{col}"] = self.oof_probabilities[:, idx]

        out_path = PROCESSED_DATA_DIR / f"oof_{self.experiment_name}.parquet"
        oof_df.to_parquet(out_path, index=False)
        console.print(f"[dim]Saved OOF predictions to {out_path}[/dim]")
