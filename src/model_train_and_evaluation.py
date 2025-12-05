
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

    #K-NN classifer

    param_grid = {
        "n_neighbors": range(1,20),
        }

    param_grid = {
        "kneighborsclassifier__n_neighbors": range(1,20)
        }

    pipe = make_pipeline(scaling, KNeighborsClassifier())
    knn_random_search = RandomizedSearchCV(
                pipe, param_distributions=param_grid, n_jobs=-1, n_iter=50, cv=5, return_train_score=True, random_state=123
        )

    knn_random_search.fit(X_train, y_train)
    
    knn_result = (pd.DataFrame(knn_random_search.cv_results_)[['mean_fit_time', 'mean_score_time', 'param_kneighborsclassifier__n_neighbors', 
                                                            'mean_test_score', 'mean_train_score']].sort_values('mean_test_score', ascending = False).head() )
    
    #save knn cross val result to table
    knn_result.to_csv(os.path.join(tables_to, "knn_results.csv"), index=False)

    decision_tree = ds_random_search.best_estimator_
    knn = knn_random_search.best_estimator_

    #save knn_classifier_model into the results/models file
    with open(os.path.join(models_to, "decision_tree.pickle"), 'wb') as f:
        pickle.dump(knn_random_search, f)

    #save decision_tree model results/models file
    with open(os.path.join(models_to, "knn.pickle"), 'wb') as f:
        pickle.dump(ds_random_search, f)

    #Test on test set for both Classifcation mode

    print("Decision tree model accuracy on test set: ", decision_tree.score(X_test, y_test))

    cm_ds = pd.DataFrame(
        confusion_matrix(y_test, decision_tree.predict(X_test)),
        index=decision_tree.classes_, 
        columns=decision_tree.classes_
    )
    print("Decision tree model confusion matrix predict on test set:")
    print(cm_ds)

    cm_ds.to_csv(os.path.join(tables_to, "confusion_matrix_ds.csv"))

    print('\n')

    print("KNN model accuracy on test set: ", knn.score(X_test, y_test))
    cm_knn = pd.DataFrame(
        confusion_matrix(y_test, knn.predict(X_test)),
        index=knn.classes_, 
        columns=knn.classes_
    )
    print("K-NN confusion matrix predict on test set:")
    print(cm_knn)

    cm_knn.to_csv(os.path.join(tables_to, "confusion_matrix_knn.csv"))

if __name__ == '__main__':
    main()

