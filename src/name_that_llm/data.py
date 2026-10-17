"""Data loading, validation, and leak-free cross-validation splitters."""

from pathlib import Path
from typing import Generator, List, Optional, Tuple
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from name_that_llm.config import (
    DEFAULT_N_FOLDS,
    KAGGLE_COMPETITIONS_DIR,
    KAGGLE_INPUT_DIR,
    MODEL_TO_IDX,
    MODEL_TO_PROVIDER,
    RAW_DATA_DIR,
    SAMPLE_DATA_DIR,
    TARGET_MODELS,
)


def resolve_data_dir() -> Path:
    """Resolve data directory based on environment (Local workspace vs Kaggle container)."""
    # 1. Local raw data directory
    if RAW_DATA_DIR.exists() and (RAW_DATA_DIR / "train.csv").exists():
        return RAW_DATA_DIR

    # 2. Kaggle environment standard path
    if KAGGLE_INPUT_DIR.exists() and (KAGGLE_INPUT_DIR / "train.csv").exists():
        return KAGGLE_INPUT_DIR

    # 3. Kaggle competitions nested path
    if KAGGLE_COMPETITIONS_DIR.exists() and (KAGGLE_COMPETITIONS_DIR / "train.csv").exists():
        return KAGGLE_COMPETITIONS_DIR

    # 4. Local sample / mock data directory
    if SAMPLE_DATA_DIR.exists() and (SAMPLE_DATA_DIR / "train.csv").exists():
        return SAMPLE_DATA_DIR

    # Default to raw data dir
    return RAW_DATA_DIR


def load_train_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """Load and validate training dataset."""
    path = file_path or (resolve_data_dir() / "train.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"Train data file not found at: {path}. "
            f"Run `python scripts/download_data.py` or `python scripts/generate_sample_data.py` first."
        )

    if str(path).endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    # Validate required columns
    required_cols = {"id", "prompt_id", "text", "model"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"train data missing required columns: {missing}")

    # Clean text
    df["text"] = df["text"].fillna("").astype(str)
    df["prompt_id"] = df["prompt_id"].astype(str)
    df["id"] = df["id"].astype(str)

    # Encode targets
    df["target"] = df["model"].map(MODEL_TO_IDX)
    df["provider"] = df["model"].map(MODEL_TO_PROVIDER)

    # Check for invalid models
    unknown_models = df[df["target"].isna()]["model"].unique()
    if len(unknown_models) > 0:
        raise ValueError(f"Found unknown target models in dataset: {unknown_models}")

    df["target"] = df["target"].astype(int)
    return df


def load_test_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """Load and validate test dataset."""
    path = file_path or (resolve_data_dir() / "test.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"Test data file not found at: {path}. "
            f"Run `python scripts/download_data.py` or `python scripts/generate_sample_data.py` first."
        )

    if str(path).endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    required_cols = {"id", "text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"test data missing required columns: {missing}")

    df["text"] = df["text"].fillna("").astype(str)
    df["id"] = df["id"].astype(str)
    if "prompt_id" in df.columns:
        df["prompt_id"] = df["prompt_id"].astype(str)

    return df


def get_stratified_group_splits(
    df: pd.DataFrame,
    n_splits: int = DEFAULT_N_FOLDS,
    seed: int = 42,
) -> Generator[Tuple[pd.Index, pd.Index], None, None]:
    """Generate train/val indices using StratifiedGroupKFold on `prompt_id`.

    CRITICAL FOR THIS COMPETITION:
    The test dataset uses unseen prompts. Evaluating on random folds will leak
    prompt-specific vocabulary and produce overly optimistic CV scores.
    Grouping by `prompt_id` simulates the real test distribution.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    groups = df["prompt_id"].values
    y = df["target"].values if "target" in df.columns else df["model"].values

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(df, y, groups)):
        # Verify no group leakage
        train_prompts = set(df.iloc[train_idx]["prompt_id"])
        val_prompts = set(df.iloc[val_idx]["prompt_id"])
        leakage = train_prompts.intersection(val_prompts)
        if leakage:
            raise RuntimeError(
                f"Data leakage detected in fold {fold_idx}! Overlapping prompts: {leakage}"
            )
        yield train_idx, val_idx


def add_fold_column(
    df: pd.DataFrame,
    n_splits: int = DEFAULT_N_FOLDS,
    seed: int = 42,
) -> pd.DataFrame:
    """Add a `fold` column (0 to n_splits - 1) to dataframe based on StratifiedGroupKFold."""
    df = df.copy()
    df["fold"] = -1
    for fold, (_, val_idx) in enumerate(get_stratified_group_splits(df, n_splits=n_splits, seed=seed)):
        df.iloc[val_idx, df.columns.get_loc("fold")] = fold
    return df
