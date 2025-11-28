import pandas as pd
import os


def load_data(
    url="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
):
    save_path = "./data/iris.csv"

    # Create the data folder if it doesn't already exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Read from URL
    try:
        df = pd.read_csv(url)
    except Exception as e:
        raise ValueError(f"Failed to read {url} as CSV. Original error:\n{e}")

    # Save locally
    df.to_csv(save_path, index=False)

    print(f"File saved to: {save_path}")
    return df
