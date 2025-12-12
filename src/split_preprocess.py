import click
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from validate_iris import validate_data
from data_split import split_data


@click.command()
@click.option(
    "--rawdata",
    type=str,
    help="Path to raw data",
    default="./data/raw/iris.csv",
    show_default=True,
)
@click.option(
    "--path",
    type=str,
    help="Path where the processed data is saved",
    default="./data/processed",
    show_default=True,
)
@click.option(
    "--test-size",
    type=float,
    help="Proportion of data to use for test split",
    default=0.3,
    show_default=True,
)
@click.option(
    "--random-state",
    type=int,
    help="Random seed for reproducibility of the split",
    default=123,
    show_default=True,
)
def main(rawdata, path, test_size, random_state):
    """Perform data validation on raw data, clean up the data column names,
    and split the data into train and test data set

    Parameters
    ----------
    rawdata : str or path-like
        Path to the raw CSV file containing the input dataset.
    path : str or path-like
        Directory path where the processed train and test CSV files
        will be saved. The directory is created if it does not exist.
    test_size : float
        Proportion of the dataset to include in the test split. Should be a
        value between 0 and 1.
    random_state : int
        Seed used by the random number generator to ensure reproducible
        train/test splits.

    Returns
    -------
    None
        This function saves the processed train and test datasets to disk
        as ``iris_train.csv`` and ``iris_test.csv`` in the specified path.
        No value is returned.

    Examples
    --------
    >>> main("./data/raw/iris.csv", "./data/processed/", test_size=0.2, random_state=42)
    The train and test data are saved in data/processed/ folder
    """

    df = pd.read_csv(rawdata)
    df = df.loc[:, "sepal_length":]
    validated_data = validate_data(df)

    train_df, test_df = split_data(
        validated_data, test_size=test_size, random_state=random_state
    )

    # output_dir = "./data/processed/"
    os.makedirs(path, exist_ok=True)

    train_df.to_csv(os.path.join(path, "iris_train.csv"), index=False)
    test_df.to_csv(os.path.join(path, "iris_test.csv"), index=False)
    print(f"The train and test data are saved in {path} folder")


if __name__ == "__main__":
    main()
