"""Train and evaluate 5-Fold Hierarchical Specialist Classifier (Phase 3)."""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from rich.console import Console

from name_that_llm.config import (
    DEFAULT_N_FOLDS,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    SUBMISSIONS_DIR,
    TARGET_MODELS,
)
from name_that_llm.data import get_stratified_group_splits, load_test_data, load_train_data
from name_that_llm.metrics import compute_metrics, print_metrics_summary
from name_that_llm.models.hierarchical import HierarchicalSpecialistClassifier
from name_that_llm.submission import create_submission

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Train Hierarchical Specialist Classifier.")
    parser.add_argument("--folds", type=int, default=DEFAULT_N_FOLDS, help="Number of CV folds")
    parser.add_argument("--alpha", type=float, default=0.30, help="Specialist blend weight (0 to 1)")
    parser.add_argument("--c", type=float, default=1.5, help="Specialist LogisticRegression regularization C")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    parser.add_argument(
        "--global-oof",
        type=str,
        default="data/processed/oof_tri_blend_v4_tokenizer.parquet",
        help="Path to global multi-model ensemble OOF matrix",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    console.rule("[bold cyan]Phase 3: Hierarchical 2-Stage Classifier Evaluation[/bold cyan]")

    # 1. Load Data and Global Multi-Model OOF
    df_train = load_train_data()
    global_oof_path = Path(args.global_oof)
    if not global_oof_path.exists():
        raise FileNotFoundError(f"Global OOF file not found at: {global_oof_path}")

    df_global = pd.read_parquet(global_oof_path)
    assert (df_train["id"] == df_global["id"]).all(), "Training data and global OOF are not aligned!"

    prob_cols = [f"prob_{m}" for m in TARGET_MODELS]
    p_global = df_global[prob_cols].values
    y_true = df_train["model"].values

    base_metrics = compute_metrics(y_true, df_global["pred_model"].values, y_prob=p_global, class_labels=TARGET_MODELS)
    console.print(f"Base Multi-Model Global Balanced Acc: [bold yellow]{base_metrics['balanced_accuracy']:.4f}[/bold yellow]")

    # 2. 5-Fold Cross Validation
    splits = list(get_stratified_group_splits(df_train, n_splits=args.folds, seed=args.seed))
    oof_refined_probs = np.zeros_like(p_global)

    fitted_specialists = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        console.rule(f"[yellow]Fold {fold_idx + 1}/{args.folds}[/yellow]")
        train_sub = df_train.iloc[train_idx]
        val_sub = df_train.iloc[val_idx]
        val_global_probs = p_global[val_idx]

        model = HierarchicalSpecialistClassifier(
            alpha=args.alpha,
            specialist_c=args.c,
        )
        model.fit(train_sub["text"].tolist(), train_sub["model"].tolist())

        val_refined = model.refine_probabilities(val_sub["text"].tolist(), val_global_probs)
        oof_refined_probs[val_idx] = val_refined

        val_preds = [TARGET_MODELS[i] for i in np.argmax(val_refined, axis=1)]
        fold_metrics = compute_metrics(val_sub["model"].values, val_preds, y_prob=val_refined, class_labels=TARGET_MODELS)
        console.print(
            f"  Fold {fold_idx + 1} Refined Bal Acc: [bold green]{fold_metrics['balanced_accuracy']:.4f}[/bold green]"
        )
        fitted_specialists.append(model)

    # 3. Overall 5-Fold Evaluation
    console.rule("[bold cyan]Global Hierarchical 5-Fold Metrics Summary[/bold cyan]")
    overall_preds = [TARGET_MODELS[i] for i in np.argmax(oof_refined_probs, axis=1)]
    overall_metrics = compute_metrics(
        y_true=y_true,
        y_pred=overall_preds,
        y_prob=oof_refined_probs,
        class_labels=TARGET_MODELS,
    )
    print_metrics_summary(overall_metrics)

    # 4. Save OOF Parquet
    oof_df = df_train[["id", "prompt_id", "model"]].copy()
    oof_df.rename(columns={"model": "true_model"}, inplace=True)
    oof_df["pred_model"] = overall_preds
    for i, col in enumerate(TARGET_MODELS):
        oof_df[f"prob_{col}"] = oof_refined_probs[:, i]

    out_oof_path = PROCESSED_DATA_DIR / "oof_hierarchical_specialist_v5.parquet"
    oof_df.to_parquet(out_oof_path, index=False)
    console.print(f"[bold green]✓ Persisted Hierarchical OOF to: {out_oof_path}[/bold green]")


if __name__ == "__main__":
    main()
