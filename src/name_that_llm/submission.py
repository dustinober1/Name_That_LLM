"""Submission generation and validation for the Name That LLM Kaggle competition."""

from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import pandas as pd
from rich.console import Console

from name_that_llm.config import SUBMISSION_FILE, SUBMISSIONS_DIR, TARGET_MODELS

console = Console()


def validate_submission_dataframe(df: pd.DataFrame, expected_ids: Optional[List[str]] = None) -> None:
    """Rigorous validation of the submission dataframe against competition constraints."""
    # 1. Column check
    required_cols = ["id", "model"]
    if list(df.columns) != required_cols:
        raise ValueError(
            f"Submission columns must be strictly {required_cols}, found {list(df.columns)}"
        )

    # 2. Null check
    if df.isna().any().any():
        null_counts = df.isna().sum().to_dict()
        raise ValueError(f"Submission contains null values: {null_counts}")

    # 3. Unique ID check
    if df["id"].duplicated().any():
        num_dupes = df["id"].duplicated().sum()
        raise ValueError(f"Submission contains {num_dupes} duplicate IDs!")

    # 4. Target classes check
    valid_classes = set(TARGET_MODELS)
    submitted_classes = set(df["model"].unique())
    invalid_classes = submitted_classes - valid_classes
    if invalid_classes:
        raise ValueError(
            f"Submission contains invalid target model names not in official list: {invalid_classes}"
        )

    # 5. Expected ID alignment check
    if expected_ids is not None:
        if len(df) != len(expected_ids):
            raise ValueError(
                f"Submission length ({len(df)}) does not match expected test set length ({len(expected_ids)})"
            )
        if list(df["id"].astype(str)) != [str(i) for i in expected_ids]:
            raise ValueError("Submission 'id' order does not strictly match expected test set IDs.")


def create_submission(
    ids: Union[pd.Series, np.ndarray, List[str]],
    predictions: Union[pd.Series, np.ndarray, List[str]],
    output_path: Optional[Path] = None,
    save_csv_backup: bool = True,
) -> pd.DataFrame:
    """Create, validate, and write the official competition submission.parquet.

    Args:
        ids: Series, array, or list of test passage IDs.
        predictions: Series, array, or list of predicted model names.
        output_path: Path to write submission.parquet. Defaults to `submissions/submission.parquet`.
        save_csv_backup: If True, writes a human-readable submission.csv alongside the parquet.

    Returns:
        The validated submission DataFrame.
    """
    df = pd.DataFrame(
        {
            "id": [str(i) for i in ids],
            "model": [str(p) for p in predictions],
        }
    )

    validate_submission_dataframe(df, expected_ids=[str(i) for i in ids])

    if output_path is None:
        SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
        out_file = SUBMISSIONS_DIR / SUBMISSION_FILE
    else:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

    # Save official parquet format
    df.to_parquet(out_file, index=False)
    console.print(f"[bold green]✓ Successfully generated official submission:[/bold green] {out_file}")

    if save_csv_backup:
        csv_path = out_file.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        console.print(f"[dim]Saved CSV inspection copy: {csv_path}[/dim]")

    # Print class prediction distribution
    console.print("[dim]Prediction distribution across models:[/dim]")
    dist = df["model"].value_counts()
    for model_name, count in dist.items():
        console.print(f"  - {model_name:<24}: {count} ({count / len(df) * 100:.1f}%)")

    return df
