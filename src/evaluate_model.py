# src/save_helpers.py
import pickle
import os

def save_model(model, path):
    """
    Docstring for save_model
    
    :param model: Description
    :param path: Description
    
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
