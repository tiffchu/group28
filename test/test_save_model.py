import os
import sys
import pickle
import pytest
from sklearn.tree import DecisionTreeClassifier
from src.save_model import save_model
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

@pytest.fixture
def model():
    return DecisionTreeClassifier()

@pytest.fixture
def path(tmp_path):
    return tmp_path/"test_model.pickle"

def model_exist_in_path(model, path):
    """This test is to make sure the model is save in the path that specified"""

    save_model(model, path)
    assert path.exists()

    




def correct_model():
    """
    Docstring for correct_model
    """

