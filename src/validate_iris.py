import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check
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
    6. Check for outlier or anomalous values
    7. Check whether the proportion of missing data is higher than the threshold
    8. Correlation between target and explanatory features to avoid ovefitting or data leakage
    9. Correlation between explanatory features to avoid multicollinearity or redundant feature
    """

    df = IrisPreSplitSchema.validate(df)
    print("\nSchema validation passed (columns + types + basic ranges)\n")

    ##Checking for empty rows
    empty_rows = (df == "").all(axis=1).sum()
    if empty_rows > 0:
        raise ValueError(f"Found {empty_rows} completely empty rows")
    print("No empty rows\n")

    ##Checking for duplicates
    total_rows = df.shape[0]
    total_columns = df.shape[1]
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print(
            f"Validation FAILED: Found {duplicate_count} duplicate rows. They will be dropped. \nThe dataset now has {total_rows - duplicate_count} rows and {total_columns} columns.\n"
        )
        df = df.drop_duplicates()
    else:
        print("No duplicate rows\n")

    # Checking for target Correct category levels
    species_levels = df["species"].unique()
    expected_levels = ["setosa", "versicolor", "virginica"]
    if not set(species_levels).issubset(set(expected_levels)):
        raise ValueError(f"Unexpected species levels found: {species_levels}")
    print("Species categories OK\n")

    # Checking Target/response variable follows expected distribution
    target_counts = df["species"].value_counts(normalize=True)
    if (target_counts < 0.1).any():
        print(f"Validation FAILED: Some species underrepresented:\n{target_counts}")
    else:
        print("Target variable distribution looks reasonable\n")

    # Checking outlier or anomalous values
    # Z-score outlier check (helper function))
    def check_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
        """
        Returns True for each value that is NOT an outlier by Z-score
        """
        z = (series - series.mean()) / series.std()
        return z.abs() <= threshold

    def check_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
        """
        Returns True for each value that is NOT an outlier by IQR
        """
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        return (series >= q1 - factor * iqr) & (series <= q3 + factor * iqr)

    # outlier check
    schema = pa.DataFrameSchema(
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
            for col in df.select_dtypes(include="float").columns
        }
    )

    try:
        validated_df = schema.validate(df, lazy=True)
        print("No outliers detected!\n")
    except pa.errors.SchemaErrors as err:
        print(
            "Validation FAILED: Outliers detected! Might want to consider using StandardScaler transformation.\n"
        )

    ## Checking missingness not beyond expected threshold
    threshold = 0.05
    schema = pa.DataFrameSchema(
        {
            col: Column(
                dtype=df[col].dtype if col != "target" else int,
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
        validated_df = schema.validate(df, lazy=True)
        print("Missingness is within allowed limits.\n")
    except pa.errors.SchemaErrors as err:
        print(
            "Validation FAILED: Missingness exceeds threshold! \n May want to consider using SimpleImputer along with other transformations when training the model.\n"
        )

    # Target–feature correlations
    corrfeat_result = []
    target_species = df["species"].astype("category").cat.codes
    feature_df = df.drop(columns="species")
    correlation_tar = feature_df.corrwith(target_species, method="pearson")
    for feat, corr in correlation_tar.items():
        if abs(corr) > 0.95:
            print(
                f"Warning: Feature {feat} has way to high correlation with the target column (species) and could lead to ovefitting or data leakage: {corr}\n"
            )
            corrfeat_result.append("problem")

    if len(corrfeat_result) == 0:
        print("Target-feature correlations is in an acceptable range\n")

    # feature-feature correlations
    corr_corr_result = []
    num_col = list(feature_df.columns)

    for i in range(len(num_col)):
        for j in range(i + 1, len(num_col)):
            correlation_feat = feature_df[num_col[i]].corr(feature_df[num_col[j]])

            if abs(correlation_feat) > 0.95:
                print(
                    f"Warning: Features '{num_col[i]}' and '{num_col[j]}' are too correlated and could lead to multicollinearity or repeat feature: {correlation_feat}\n"
                )
                corr_corr_result.append("problem")

    if len(corr_corr_result) == 0:
        print("Feature-feature correlations is in an acceptable range\n")

    return validated_df
