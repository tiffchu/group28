import click
import pandas as pd
import os

@click.command()
@click.option('--url', type=str, help="URL of dataset to be downloaded", flag_value= 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

def main(url):
    """Downloads data csv data from the web to a local filepath and extracts it."""
    data = pd.read_csv(url)
    try:
        data.to_csv('../data/raw/iris.csv')
    except Exception as e:
        os.makedirs('../data/raw/')
        data.to_csv('../data/raw/iris.csv')
        print("Error:", e)
        

if __name__ == '__main__':
    main()