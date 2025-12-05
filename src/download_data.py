import click
import pandas as pd
import os

@click.command()
@click.option('--url', type=str, help="URL of dataset to be downloaded", default= "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv", show_default=True)

def main(url):
    """Downloads data csv data from the web to a local filepath and extracts it."""

    data = pd.read_csv(url)

    output_path = "../data/raw/iris.csv"

    #ensure path exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        data.to_csv('../data/raw/iris.csv')
        print('The Iris data is saved in ../data/raw directory.')

    except Exception as e:
        os.makedirs('../data/raw/')
        data.to_csv('../data/raw/iris.csv')
        print("Failed to download or save data:", e)
        print('The Iris data is saved in ../data/raw directory.')
        

if __name__ == '__main__':
    main()