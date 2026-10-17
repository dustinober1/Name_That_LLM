"""Fine-tune a Transformer model (DeBERTa-v3, RoBERTa, etc.) with StratifiedGroupKFold."""

import argparse
from pathlib import Path
from rich.console import Console

from name_that_llm.config import (
    CHECKPOINTS_DIR,
    DEFAULT_N_FOLDS,
    MODEL_TO_IDX,
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
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=384, help="Max token length")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    parser.add_argument("--fold-to-run", type=int, default=None, help="Run only a single fold (0-indexed)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    console.rule(f"[bold cyan]Transformer Fine-Tuning: {args.model_name} on {device}[/bold cyan]")

    df_train = load_train_data()
    splits = list(get_stratified_group_splits(df_train, n_splits=args.folds, seed=args.seed))

    oof_preds = [None] * len(df_train)
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
            lr=args.lr,
            epochs=args.epochs,
            device=device,
        )

        model.fit(
            train_texts=train_sub["text"].tolist(),
            train_labels=train_sub["target"].tolist(),
            val_texts=val_sub["text"].tolist(),
            val_labels=val_sub["target"].tolist(),
        )

        val_preds = model.predict(val_sub["text"].tolist())
        for idx_pos, p in zip(val_idx, val_preds):
            oof_preds[idx_pos] = TARGET_MODELS[p]

        fold_metrics = compute_metrics(val_sub["model"].values, [TARGET_MODELS[p] for p in val_preds])
        console.print(f"Fold {fold_idx + 1} Balanced Accuracy: [bold green]{fold_metrics['balanced_accuracy']:.4f}[/bold green]")
        fitted_models.append(model)

    if args.fold_to_run is None:
        console.rule("[bold cyan]Overall OOF Transformer Metrics[/bold cyan]")
        overall_metrics = compute_metrics(df_train["model"].values, oof_preds)
        print_metrics_summary(overall_metrics)

        try:
            df_test = load_test_data()
            console.print("[dim]Generating ensemble test predictions...[/dim]")
            test_probs = sum(m.predict_proba(df_test["text"].tolist()) for m in fitted_models) / len(fitted_models)
            final_preds = [TARGET_MODELS[i] for i in test_probs.argmax(axis=-1)]
            create_submission(
                ids=df_test["id"],
                predictions=final_preds,
                output_path=SUBMISSIONS_DIR / "submission_transformer.parquet",
            )
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
