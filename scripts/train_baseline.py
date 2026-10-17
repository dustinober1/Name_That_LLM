"""Train linear baseline with TF-IDF and stylometry using 5-fold StratifiedGroupKFold."""

import argparse
from pathlib import Path
from rich.console import Console

from name_that_llm.config import DEFAULT_N_FOLDS, RANDOM_SEED, SUBMISSIONS_DIR
from name_that_llm.data import load_test_data, load_train_data
from name_that_llm.evaluation import CrossValidationRunner
from name_that_llm.models.baseline_linear import LinearModelPipeline
from name_that_llm.submission import create_submission

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline TF-IDF + Stylometry linear model.")
    parser.add_argument("--folds", type=int, default=DEFAULT_N_FOLDS, help="Number of CV folds")
    parser.add_argument("--c", type=float, default=2.0, help="Inverse regularization strength")
    parser.add_argument(
        "--classifier",
        choices=["logistic", "ridge", "sgd"],
        default="logistic",
        help="Linear classifier algorithm",
    )
    parser.add_argument(
        "--no-stylometry",
        action="store_true",
        help="Disable hand-crafted stylometric features",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    parser.add_argument(
        "--output-name",
        type=str,
        default="baseline_linear",
        help="Name prefix for submission and OOF files",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    console.rule("[bold cyan]Name That LLM - Baseline Training Pipeline[/bold cyan]")

    # 1. Load Data
    console.print("[dim]Loading training data...[/dim]")
    df_train = load_train_data()
    console.print(f"Loaded [bold]{len(df_train)}[/bold] training samples across {df_train['prompt_id'].nunique()} prompts.")

    # 2. Define Model Factory
    def model_factory():
        return LinearModelPipeline(
            classifier_type=args.classifier,
            c=args.c,
            use_stylometry=not args.no_stylometry,
            random_state=args.seed,
        )

    # 3. Cross-Validation
    runner = CrossValidationRunner(
        model_factory=model_factory,
        n_folds=args.folds,
        seed=args.seed,
        experiment_name=args.output_name,
    )
    metrics = runner.run(df_train)

    # 4. Generate Test Submission
    try:
        df_test = load_test_data()
        console.print(f"[dim]Generating predictions for {len(df_test)} test samples...[/dim]")
        test_preds, test_probs = runner.predict_test(df_test)

        out_path = SUBMISSIONS_DIR / f"submission_{args.output_name}.parquet"
        create_submission(
            ids=df_test["id"],
            predictions=test_preds,
            output_path=out_path,
        )
    except FileNotFoundError:
        console.print("[yellow]Test data not found; skipped test submission generation.[/yellow]")


if __name__ == "__main__":
    main()
