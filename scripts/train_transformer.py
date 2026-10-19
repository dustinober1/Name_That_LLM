"""Fine-tune a Transformer model (DeBERTa-v3, RoBERTa, etc.) with StratifiedGroupKFold."""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from rich.console import Console

from name_that_llm.config import (
    CHECKPOINTS_DIR,
    DEFAULT_N_FOLDS,
    MODEL_TO_IDX,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    SUBMISSIONS_DIR,
    TARGET_MODELS,
)
from name_that_llm.data import get_stratified_group_splits, load_test_data, load_train_data
from name_that_llm.metrics import compute_metrics, print_metrics_summary
from name_that_llm.models.transformer_classifier import TransformerClassifier, get_device
from name_that_llm.submission import create_submission

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Transformer model for Name That LLM.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="microsoft/deberta-v3-small",
        help="HuggingFace model backbone",
    )
    parser.add_argument("--folds", type=int, default=DEFAULT_N_FOLDS, help="Number of folds")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs per fold")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--accum-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate for transformer backbone")
    parser.add_argument("--head-lr", type=float, default=None, help="Learning rate for classification head (defaults to 5x lr)")
    parser.add_argument("--max-length", type=int, default=384, help="Max token length")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    parser.add_argument("--pooling", type=str, default="cls", choices=["cls", "mean"], help="Pooling strategy")
    parser.add_argument("--fold-to-run", type=int, default=None, help="Run only a single fold (0-indexed)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    model_slug = args.model_name.replace("/", "_")
    console.rule(f"[bold cyan]Transformer Fine-Tuning: {args.model_name} on {device}[/bold cyan]")

    df_train = load_train_data()
    splits = list(get_stratified_group_splits(df_train, n_splits=args.folds, seed=args.seed))

    oof_preds = [None] * len(df_train)
    oof_probs = np.zeros((len(df_train), len(TARGET_MODELS)), dtype=np.float32)
    fitted_models = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        if args.fold_to_run is not None and fold_idx != args.fold_to_run:
            continue

        console.rule(f"[yellow]Fold {fold_idx + 1}/{args.folds}[/yellow]")
        train_sub = df_train.iloc[train_idx]
        val_sub = df_train.iloc[val_idx]

        model = TransformerClassifier(
            model_name=args.model_name,
            max_length=args.max_length,
            batch_size=args.batch_size,
            accum_steps=args.accum_steps,
            lr=args.lr,
            head_lr=args.head_lr,
            epochs=args.epochs,
            pooling=args.pooling,
            device=device,
        )

        model.fit(
            train_texts=train_sub["text"].tolist(),
            train_labels=train_sub["target"].tolist(),
            val_texts=val_sub["text"].tolist(),
            val_labels=val_sub["target"].tolist(),
        )

        val_probs = model.predict_proba(val_sub["text"].tolist())
        val_preds = np.argmax(val_probs, axis=-1)

        for i, idx_pos in enumerate(val_idx):
            oof_preds[idx_pos] = TARGET_MODELS[val_preds[i]]
            oof_probs[idx_pos] = val_probs[i]

        fold_metrics = compute_metrics(
            y_true=val_sub["model"].values,
            y_pred=[TARGET_MODELS[p] for p in val_preds],
            y_prob=val_probs,
            class_labels=TARGET_MODELS,
        )
        print_metrics_summary(fold_metrics)

        # Save single fold OOF predictions
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        fold_oof_df = pd.DataFrame(
            {
                "id": val_sub["id"].values,
                "prompt_id": val_sub["prompt_id"].values,
                "true_model": val_sub["model"].values,
                "pred_model": [TARGET_MODELS[p] for p in val_preds],
            }
        )
        for i, col in enumerate(TARGET_MODELS):
            fold_oof_df[f"prob_{col}"] = val_probs[:, i]
        fold_save_path = PROCESSED_DATA_DIR / f"oof_{model_slug}_fold{fold_idx}.parquet"
        fold_oof_df.to_parquet(fold_save_path, index=False)
        console.print(f"[dim]Saved fold {fold_idx + 1} OOF predictions to {fold_save_path}[/dim]")

        # Save model checkpoint
        ckpt_dir = CHECKPOINTS_DIR / f"{model_slug}_fold{fold_idx}"
        model.save(ckpt_dir)
        console.print(f"[dim]Saved model checkpoint to {ckpt_dir}[/dim]")

        fitted_models.append(model)

    if args.fold_to_run is None:
        console.rule("[bold cyan]Overall OOF Transformer Metrics[/bold cyan]")
        overall_metrics = compute_metrics(
            y_true=df_train["model"].values,
            y_pred=oof_preds,
            y_prob=oof_probs,
            class_labels=TARGET_MODELS,
        )
        print_metrics_summary(overall_metrics)

        # Save overall OOF
        oof_df = pd.DataFrame(
            {
                "id": df_train["id"].values,
                "prompt_id": df_train["prompt_id"].values,
                "true_model": df_train["model"].values,
                "pred_model": oof_preds,
            }
        )
        for i, col in enumerate(TARGET_MODELS):
            oof_df[f"prob_{col}"] = oof_probs[:, i]
        overall_save_path = PROCESSED_DATA_DIR / f"oof_{model_slug}.parquet"
        oof_df.to_parquet(overall_save_path, index=False)
        console.print(f"[bold green]Saved overall OOF predictions to {overall_save_path}[/bold green]")

        try:
            df_test = load_test_data()
            console.print("[dim]Generating ensemble test predictions...[/dim]")
            test_probs = sum(m.predict_proba(df_test["text"].tolist()) for m in fitted_models) / len(fitted_models)
            final_preds = [TARGET_MODELS[i] for i in test_probs.argmax(axis=-1)]
            create_submission(
                ids=df_test["id"],
                predictions=final_preds,
                output_path=SUBMISSIONS_DIR / f"submission_{model_slug}.parquet",
            )
        except FileNotFoundError:
            pass



if __name__ == "__main__":
    main()
