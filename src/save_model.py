# src/save_model.py
import pickle
import os

def save_model(model, path):
    """
    This function saves the best model after optimizing its hyperparameters during training using pickle.

    This function serves as a reusable step for saving models so they can be easily 
    loaded and reused in the future. It also helps reduce the amount of repeated code. 
    
    Parameters
    ----------
    model : any
        The optimized trained model. After model.best_estimator_
    path : str
        The file path where the model is saved to.

    Returns
    -------
    None

    Examples
    --------
    >>> model = decision_tree_random_search.best_estimator_
    >>> save_model(model, f"{models_to}/decision_tree.pickle")

    """
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
