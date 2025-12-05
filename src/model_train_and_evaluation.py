
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
import pandas as pd
import click
import pickle
import os

@click.command()
@click.option('--training-data', type=str, help="Path to training data", default = "../data/processed/iris_train.csv")
@click.option('--test-data', type=str, help="Path to test data", default = "../data/processed/iris_test.csv")
@click.option('--models-to', type=str, help="Path to directory where the pipeline object will be written to", default = "../results/models")
@click.option('--tables-to', type=str, help="Path to directory where evaluation tables will be written to",default="../results/tables")

def main(training_data, test_data, models_to, tables_to):

    #create folder for models and result tables (like confusion matrix, train and test score tables)
    os.makedirs(models_to, exist_ok=True)
    os.makedirs(tables_to, exist_ok=True)

    train_df = pd.read_csv(training_data)
    test_df = pd.read_csv(test_data)
    
    X_train = train_df.drop('species', axis=1)
    y_train = train_df['species']

    X_test = test_df.drop('species', axis=1)
    y_test = test_df['species']

    scaling = StandardScaler()

if __name__ == '__main__':
    main()

