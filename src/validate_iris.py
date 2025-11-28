import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check
import numpy as np
import warnings


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

    ##Checking for empty rows
    empty_rows = (df == "").all(axis=1).sum()
    if empty_rows > 0:
        raise ValueError(f"Found {empty_rows} completely empty rows")
    print("No empty rows")
    
    ##Checking for duplicates
    total_rows = df.shape[0]
    total_columns = df.shape[1]
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print(f"Validation FAILED: Found {duplicate_count} duplicate rows. They will be dropped. \nThe dataset now have {total_rows - duplicate_count} rows and {total_columns} columns.")
        df = df.drop_duplicates()
    else:
        print("No duplicate rows")
    
    #Checking for target Correct category levels 
    species_levels = df["species"].unique()
    expected_levels = ["setosa", "versicolor", "virginica"]
    if not set(species_levels).issubset(set(expected_levels)):
        raise ValueError(f"Unexpected species levels found: {species_levels}")
    print("Species categories OK")
    
    #Checking Target/response variable follows expected distribution 
    target_counts = df["species"].value_counts(normalize=True)
    if (target_counts < 0.1).any():
        print(f"Validation FAILED: Some species underrepresented:\n{target_counts}")
    else:
        print("Target variable distribution looks reasonable")

    #Checking outlier or anomalous values
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
        return (series >= q1 - factor*iqr) & (series <= q3 + factor*iqr)

    #outlier check
    schema = pa.DataFrameSchema({
        col: Column(
            dtype=float,
            checks=[
                Check(lambda s: check_zscore(s), error=f"{col} has Z-score outliers"),
                Check(lambda s: check_iqr(s), error=f"{col} has IQR outliers")
            ]
        ) for col in df.select_dtypes(include='float').columns
    })

    try:
        validated_df = schema.validate(df, lazy=True)
        print("No outliers detected!")
    except pa.errors.SchemaErrors as err:
        print("Validation FAILED: Outliers detected! Might want to consider using StandardScaler transformation.")
        

    ## Checking missingness not beyond expected threshold
    threshold = 0.05
    schema = pa.DataFrameSchema({
        col: Column(
            dtype=df[col].dtype if col != "target" else int,
            checks=[Check(lambda s: s.isna().mean() < threshold,
                        error=f"Missingness exceeds allowed threshold of {threshold:.2f}")]
        )
        for col in df.columns
    })
    try:
        validated_df = schema.validate(df, lazy=True)
        print("Missingness is within allowed limits.")
    except pa.errors.SchemaErrors as err:
        print("Validation FAILED: Missingness exceeds threshold! \n May want to consider using SimpleImputer along with other transformations when training the model.")

    return df


