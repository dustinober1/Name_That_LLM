"""Train LightGBM gradient boosted trees with stylometric and SVD features."""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from rich.console import Console

from name_that_llm.config import DEFAULT_N_FOLDS, RANDOM_SEED, SUBMISSIONS_DIR
from name_that_llm.data import load_test_data, load_train_data
from name_that_llm.evaluation import CrossValidationRunner
from name_that_llm.models.gbm_classifier import GBDTClassifierPipeline
from name_that_llm.submission import create_submission

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Train LightGBM with Stylometry, Tokenizer Fingerprinting, and SVD.")
    parser.add_argument("--folds", type=int, default=DEFAULT_N_FOLDS, help="Number of CV folds")
    parser.add_argument("--n-svd", type=int, default=128, help="Number of SVD components")
    parser.add_argument("--n-estimators", type=int, default=200, help="Number of boosting trees")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    parser.add_argument(
        "--use-tokenizers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include tokenizer fingerprinting features (Qwen, GLM, tiktoken)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="lgbm_stylometry_tokenizer",
        help="Experiment name prefix",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    console.rule("[bold cyan]Name That LLM - LightGBM Pipeline[/bold cyan]")

    # 1. Load Data
    df_train = load_train_data()
    console.print(f"Loaded [bold]{len(df_train)}[/bold] training samples.")

    # 2. Define Model Factory
    def model_factory():
        return GBDTClassifierPipeline(
            n_svd_components=args.n_svd,
            use_tokenizer_features=args.use_tokenizers,
            n_estimators=args.n_estimators,
            learning_rate=args.lr,
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

    # 4. Display top feature importances from first fold
    if runner.fitted_models:
        imp_df = runner.fitted_models[0].get_feature_importances()
        console.rule("[bold yellow]Top 15 Most Discriminative Features[/bold yellow]")
        for idx, row in imp_df.head(15).iterrows():
            console.print(f"  {idx + 1:2d}. {row['feature']:<30} (importance: {row['importance']})")

    # 5. Generate Test Submission
    try:
        df_test = load_test_data()
        console.print(f"[dim]Predicting on {len(df_test)} test samples...[/dim]")
        test_preds, _ = runner.predict_test(df_test)

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
