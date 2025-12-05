
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

    #Decision Tree

    param_grid = {
        "decisiontreeclassifier__max_depth": range(1,20)
        }

    pipe = make_pipeline(scaling, DecisionTreeClassifier())
    ds_random_search = RandomizedSearchCV(
                pipe, param_distributions=param_grid, n_jobs=-1, n_iter=50, cv=5, return_train_score=True, random_state=123
        )

    ds_random_search.fit(X_train, y_train)
    
    ds_result = (pd.DataFrame(ds_random_search.cv_results_)[['mean_fit_time', 'mean_score_time', 'param_decisiontreeclassifier__max_depth', 
                                                            'mean_test_score', 'mean_train_score']].sort_values('mean_test_score', ascending = False).head() )
    
    #save knn cross val result to table
    ds_result.to_csv(os.path.join(tables_to, "knn_results.csv"), index=False)

if __name__ == '__main__':
    main()

