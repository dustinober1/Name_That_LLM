"""Tests for cross-validation splitting and leak prevention."""

import pandas as pd
import pytest

from name_that_llm.data import get_stratified_group_splits, load_train_data


def test_no_prompt_leakage_across_folds():
    df_train = load_train_data()
    n_splits = 5

    for fold_idx, (train_idx, val_idx) in enumerate(
        get_stratified_group_splits(df_train, n_splits=n_splits, seed=42)
    ):
        train_prompts = set(df_train.iloc[train_idx]["prompt_id"])
        val_prompts = set(df_train.iloc[val_idx]["prompt_id"])

        # Crucial test: No overlap between training prompt_ids and validation prompt_ids
        overlap = train_prompts.intersection(val_prompts)
        assert len(overlap) == 0, f"Fold {fold_idx} has prompt leakage: {overlap}"
