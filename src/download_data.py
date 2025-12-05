import click
import pandas as pd
import os

@click.command()
@click.option('--url', type=str, help="URL of dataset to be downloaded")

def main(url):
    """Downloads data csv data from the web to a local filepath and extracts it."""
    data = pd.read_csv(url)
    try:
        data.to_csv('../data/raw/iris.csv')
    except:
        os.makedirs('../data/raw/')
        data.to_csv('../data/raw/iris.csv')

if __name__ == '__main__':
    main()