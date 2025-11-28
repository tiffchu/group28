import pandas as pd
import pandera.pandas as pa
from pandera import Check
import numpy as np


class IrisPreSplitSchema(pa.DataFrameModel):
    sepal_length: float = pa.Field(ge=0, le=10, nullable=False)
    sepal_width: float = pa.Field(ge=0, le=10, nullable=False)
    petal_length: float = pa.Field(ge=0, le=10, nullable=False)
    petal_width: float = pa.Field(ge=0, le=10, nullable=False)
    species: str = pa.Field(isin=["setosa", "versicolor", "virginica"], nullable=False)

    class Config:
        strict = True  # No extra columns allowed
        coerce = True  # Convert types if possible


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the Iris dataset for use in analysis or modeling.

    Performs schema validation, checks for empty rows, duplicates,
    valid category levels, and basic target distribution sanity.

    Parameters
    ----------
    df : pd.DataFrame
        The Iris dataset to validate. Expected columns:
        ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

    Returns
    -------
    pd.DataFrame
        Cleaned and validated DataFrame. Duplicate rows are dropped.

    Raises
    ------
    ValueError
        If completely empty rows are found.
        If unexpected species categories are present.

    Notes
    -----
    Checks performed:
    1. Schema validation (column names, types, numeric ranges 0-10, strict columns)
    2. No completely empty rows
    3. Duplicate rows are removed
    4. Species column has only allowed levels: 'setosa', 'versicolor', 'virginica'
    5. Target variable distribution sanity (warns if any species < 10% of dataset)
    """

    df = IrisPreSplitSchema.validate(df)
    print("Schema validation passed (columns + types + basic ranges)")

    empty_rows = (df == "").all(axis=1).sum()
    if empty_rows > 0:
        raise ValueError(f"Found {empty_rows} completely empty rows")
    print("No empty rows")

    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print(f"Found {duplicate_count} duplicate rows. Dropping them.")
        df = df.drop_duplicates()
    else:
        print("No duplicate rows")

    species_levels = df["species"].unique()
    expected_levels = ["setosa", "versicolor", "virginica"]
    if not set(species_levels).issubset(set(expected_levels)):
        raise ValueError(f"Unexpected species levels found: {species_levels}")
    print("Species categories OK")

    target_counts = df["species"].value_counts(normalize=True)
    if (target_counts < 0.1).any():
        print(f"Warning: Some species underrepresented:\n{target_counts}")
    else:
        print("Target variable distribution looks reasonable")

    return df
