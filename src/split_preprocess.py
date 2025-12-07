import click
from sklearn.model_selection import train_test_split
import pandas as pd
from validate_iris import validate_data
import os

@click.command()

@click.option('--rawdata', type=str, help="Path to raw data", default ='./data/raw/iris.csv', show_default = True)
def main(rawdata):
    """Perform data validation on raw data, clean up the data column names, and split the data into train and test data set"""

    df = pd.read_csv(rawdata)
    df = df.loc[:, 'sepal_length':]
    validated_data = validate_data(df)

    train_df, test_df = train_test_split(validated_data, test_size=0.3, random_state=123)

    output_dir = "./data/processed/"
    os.makedirs(output_dir, exist_ok=True)

    train_df.to_csv(os.path.join(output_dir, "iris_train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "iris_test.csv"), index=False)
    print(f'The train and test data are saved in {output_dir} folder')
    
if __name__ == '__main__':
    main()
