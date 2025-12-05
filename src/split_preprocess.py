import click
from sklearn.model_selection import train_test_split
import pandas as pd
from validate_iris import validate_data
import os

@click.command()

@click.option('--rawdata', type=str, help="Path to raw data", default = '../data/raw/iris.csv')
def main(rawdata):
    """Perform data validation on raw data, clean up the data column names, and split the data into train and test data set"""
    df = pd.read_csv(rawdata)
    df = df.loc[:, 'sepal_length':]
    data_validation = validate_data(df)

    train_df, test_df = train_test_split(data_validation, test_size=0.3, random_state=123)

    try:
        train_df.to_csv('../data/processed/iris_train.csv')
        test_df.to_csv('../data/processed/iris_test.csv')
    except:
        os.makedirs('../data/processed/')
        train_df.to_csv('../data/processed/iris_train.csv')
        test_df.to_csv('../data/processed/iris_test.csv')
    
if __name__ == '__main__':
    main()
