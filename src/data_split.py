# File: src/data_split.py

import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple


def split_data(
    df: pd.DataFrame, test_size: float = 0.3, random_state: int = 123
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame into training and testing sets.

    This function serves as a reusable wrapper around sklearn's
    train_test_split, ensuring consistent parameters are used
    across different scripts or projects.

    Parameters
    ----------
    df : pandas.DataFrame
        The validated and cleaned input DataFrame.
    test_size : float, optional
        The proportion of the data to include in the test split.
        Defaults to 0.3 (30%).
    random_state : int, optional
        Controls the shuffling applied to the data before splitting.
        Defaults to 123 for reproducibility.

    Returns
    -------
    tuple of (pandas.DataFrame, pandas.DataFrame)
        A tuple containing (train_df, test_df).

    Examples
    --------
    >>> data = pd.DataFrame({'a': range(10), 'b': range(10)})
    >>> train, test = split_data(data, test_size=0.2, random_state=42)
    >>> len(test)
    2
    >>> len(train)
    8
    """

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    return train_df, test_df
