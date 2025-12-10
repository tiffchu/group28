import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check
import numpy as np
from typing import Tuple


class IrisPreSplitSchema(pa.DataFrameModel):
    sepal_length: float = pa.Field(ge=0, le=10, nullable=False)
    sepal_width: float = pa.Field(ge=0, le=10, nullable=False)
    petal_length: float = pa.Field(ge=0, le=10, nullable=False)
    petal_width: float = pa.Field(ge=0, le=10, nullable=False)
    species: str = pa.Field(isin=["setosa", "versicolor", "virginica"], nullable=False)

    class Config:
        strict = True  # No extra columns allowed
        coerce = True  # Convert types if possible


def check_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Returns True for each value that is NOT an outlier by Z-score."""
    z = (series - series.mean()) / series.std()
    return z.abs() <= threshold


def check_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """Returns True for each value that is NOT an outlier by IQR."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return (series >= q1 - factor * iqr) & (series <= q3 + factor * iqr)


def check_empty_rows(df: pd.DataFrame):
    """Checks for and raises an error if completely empty rows are found."""
    empty_rows = (df == "").all(axis=1).sum()
    if empty_rows > 0:
        raise ValueError(f"Found {empty_rows} completely empty rows")
    print("Check: No completely empty rows found.")


def check_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Checks for duplicates and drops them, returning the cleaned DataFrame."""
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        cleaned_df = df.drop_duplicates()
        print(
            f"Check: {duplicate_count} duplicate rows found and dropped. "
            f"New shape: {cleaned_df.shape}"
        )
        return cleaned_df
    else:
        print("Check: No duplicate rows found.")
        return df


def check_species_levels(df: pd.DataFrame):
    """Checks if the 'species' column contains only expected category levels."""
    species_levels = df["species"].unique()
    expected_levels = ["setosa", "versicolor", "virginica"]
    if not set(species_levels).issubset(set(expected_levels)):
        raise ValueError(f"Unexpected species levels found: {species_levels}")
    print("Check: Species categories OK.")


def check_target_distribution(df: pd.DataFrame, min_proportion: float = 0.1):
    """Checks if any species is underrepresented below a minimum proportion."""
    target_counts = df["species"].value_counts(normalize=True)
    if (target_counts < min_proportion).any():
        print(
            f"Warning: Target distribution issue. Some species under {min_proportion*100:.0f}%:\n"
            f"{target_counts[target_counts < min_proportion]}"
        )
    else:
        print("Check: Target variable distribution looks reasonable.")


def check_outliers(df: pd.DataFrame):
    """Performs Z-score and IQR outlier checks using pandera."""
    float_cols = df.select_dtypes(include="float").columns

    # Only validate numeric columns
    outlier_schema = pa.DataFrameSchema(
        {
            col: Column(
                dtype=float,
                checks=[
                    Check(
                        lambda s: check_zscore(s), error=f"{col} has Z-score outliers"
                    ),
                    Check(lambda s: check_iqr(s), error=f"{col} has IQR outliers"),
                ],
            )
            for col in float_cols
        }
    )

    try:
        # Use only the float columns for validation
        outlier_schema.validate(df[float_cols], lazy=True)
        print("Check: No significant outliers detected!")
    except pa.errors.SchemaErrors as err:
        print(
            "Warning: Outliers detected! Consider using StandardScaler transformation for the affected features."
        )


def check_missing_data(df: pd.DataFrame, threshold: float = 0.05):
    """Checks if missingness in any column exceeds the allowed threshold."""
    missing_schema = pa.DataFrameSchema(
        {
            col: Column(
                dtype=df[col].dtype,
                checks=[
                    Check(
                        lambda s: s.isna().mean() < threshold,
                        error=f"Missingness exceeds allowed threshold of {threshold:.2f}",
                    )
                ],
            )
            for col in df.columns
        }
    )
    try:
        missing_schema.validate(df, lazy=True)
        print("Check: Missingness is within allowed limits.")
    except pa.errors.SchemaErrors as err:
        print(
            "Warning: Missingness exceeds threshold in one or more columns! "
            "Consider using an imputer when training the model."
        )


def check_correlations(df: pd.DataFrame, threshold: float = 0.95):
    """
    Checks for high target-feature and feature-feature correlations.

    This combines two correlation checks into one function for efficiency.
    """
    feature_df = df.drop(columns="species", errors="ignore")
    num_cols = feature_df.select_dtypes(include=np.number).columns

    if feature_df.empty or len(num_cols) < 2:
        print(
            "Warning: Skipping correlation checks due to insufficient numeric features."
        )
        return

    # 1. Target–Feature Correlations
    target_species = df["species"].astype("category").cat.codes
    target_corr = feature_df[num_cols].corrwith(target_species, method="pearson")

    high_target_corr = target_corr[target_corr.abs() > threshold]
    if not high_target_corr.empty:
        print(
            f"Warning: High Target-Feature Correlation (> {threshold:.2f}) found. "
            f"Could indicate data leakage or risk of overfitting:\n{high_target_corr}"
        )
    else:
        print("Check: Target-feature correlations are in an acceptable range.")

    # 2. Feature-Feature Correlations (Multicollinearity)
    corr_matrix = feature_df[num_cols].corr().abs()
    # Filter for upper triangle (k=1) and correlations > threshold
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = upper[upper > threshold].stack()

    if not high_corr_pairs.empty:
        print(
            f"Warning: High Feature-Feature Correlation (> {threshold:.2f}) found. "
            f"Consider removing redundant features to avoid multicollinearity:\n{high_corr_pairs}"
        )
    else:
        print("Check: Feature-feature correlations are in an acceptable range.")


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrates data validation checks.

    1. Schema validation (mandatory first step).
    2. Drops duplicates (mandatory cleaning step).
    3. Runs remaining checks (non-blocking errors issue warnings or raise specific errors).

    Parameters
    ----------
    df : pd.DataFrame
        The Iris dataset to validate.

    Returns
    -------
    pd.DataFrame
        Cleaned and validated DataFrame (duplicates removed).

    Raises
    ------
    ValueError
        If completely empty rows or unexpected species levels are found.
    pandera.errors.SchemaErrors
        If the initial schema validation fails.
    """
    print("--- Starting Data Validation ---\n")

    # 1. Mandatory Schema Validation (Pandera handles column/type checks)
    # The returned df is type-coerced and validated against basic ranges (0-10).
    df = IrisPreSplitSchema.validate(df)
    print("Check: Initial schema validation passed (columns + types + ranges).\n")

    # 2. Checks that modify or block data processing
    check_empty_rows(df.copy())  # Use a copy so original validation is maintained
    df = check_duplicates(df)

    print("\n--- Running Non-Blocking Checks ---\n")

    # 3. Non-blocking checks (issue warnings or print status)
    check_species_levels(df)
    check_target_distribution(df)
    check_outliers(df)
    check_missing_data(df)
    check_correlations(df)

    print("\n--- Data Validation Complete ---")
    return df
