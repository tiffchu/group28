import pandas as pd
import pytest
from src.data_split import split_data
from typing import List


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Fixture to create a simple, reproducible DataFrame for testing."""
    data = {
        "feature_a": range(100),
        "feature_b": [i * 2 for i in range(100)],
        "target": [i % 2 for i in range(100)],
    }
    return pd.DataFrame(data)


def test_split_data_return_type(sample_data):
    """Verifies that the function returns a tuple of two pandas DataFrames."""
    result = split_data(sample_data)
    assert isinstance(result, tuple), "Result should be a tuple."
    assert len(result) == 2, "Tuple should contain exactly two elements."
    assert isinstance(
        result[0], pd.DataFrame
    ), "First element must be a DataFrame (train)."
    assert isinstance(
        result[1], pd.DataFrame
    ), "Second element must be a DataFrame (test)."


def test_split_data_default_size(sample_data):
    """Verifies the default test_size=0.3 results in correct row counts."""
    df_train, df_test = split_data(sample_data, test_size=0.3, random_state=123)
    total_rows = len(sample_data)

    assert len(df_train) == 70, "Train set size is incorrect for default 0.3 split."
    assert len(df_test) == 30, "Test set size is incorrect for default 0.3 split."
    assert len(df_train) + len(df_test) == total_rows, "Total rows must be preserved."


def test_split_data_custom_size(sample_data):
    """Verifies functionality with a non-default test_size."""
    df_train, df_test = split_data(sample_data, test_size=0.25, random_state=42)

    assert len(df_train) == 75, "Train set size is incorrect for 0.25 split."
    assert len(df_test) == 25, "Test set size is incorrect for 0.25 split."


def test_split_data_reproducibility(sample_data):
    """Verifies that the same random_state produces identical splits."""

    train1, test1 = split_data(sample_data, random_state=999)
    train2, test2 = split_data(sample_data, random_state=999)

    pd.testing.assert_frame_equal(train1, train2)
    pd.testing.assert_frame_equal(test1, test2)
