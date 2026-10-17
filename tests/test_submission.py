"""Tests for competition submission format and validation."""

import numpy as np
import pandas as pd
import pytest

from name_that_llm.config import TARGET_MODELS
from name_that_llm.submission import create_submission, validate_submission_dataframe


def test_valid_submission_dataframe():
    df = pd.DataFrame(
        {
            "id": ["test_001", "test_002"],
            "model": ["gpt-5.5", "claude-sonnet-5"],
        }
    )
    # Should not raise
    validate_submission_dataframe(df, expected_ids=["test_001", "test_002"])


def test_invalid_columns_rejected():
    df = pd.DataFrame(
        {
            "sample_id": ["test_001"],
            "prediction": ["gpt-5.5"],
        }
    )
    with pytest.raises(ValueError, match="columns must be strictly"):
        validate_submission_dataframe(df)


def test_invalid_target_class_rejected():
    df = pd.DataFrame(
        {
            "id": ["test_001"],
            "model": ["non_existent_llm_model"],
        }
    )
    with pytest.raises(ValueError, match="invalid target model names"):
        validate_submission_dataframe(df)


def test_nulls_rejected():
    df = pd.DataFrame(
        {
            "id": ["test_001", "test_002"],
            "model": ["gpt-5.5", None],
        }
    )
    with pytest.raises(ValueError, match="contains null values"):
        validate_submission_dataframe(df)


def test_duplicate_ids_rejected():
    df = pd.DataFrame(
        {
            "id": ["test_001", "test_001"],
            "model": ["gpt-5.5", "gpt-5.5"],
        }
    )
    with pytest.raises(ValueError, match="duplicate IDs"):
        validate_submission_dataframe(df)
