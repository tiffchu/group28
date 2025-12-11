import os
import sys
import pickle
import pytest
from sklearn.tree import DecisionTreeClassifier
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.save_model import save_model

@pytest.fixture
def model():
    return DecisionTreeClassifier()

@pytest.fixture
def path(tmp_path):
    return tmp_path/"test_model.pickle"

def test_model_exist_in_path(model, path):
    """This test is to make sure the model is save in the path that specified"""

    save_model(model, path)
    assert path.exists()

def test_correct_model_in_path(model, path):
    """This test is to make sure the correct model saved, by unload it and compare if it is the same model"""

    save_model(model, path)
    unload_model = pickle.loads(path.read_bytes())
    assert isinstance(unload_model, DecisionTreeClassifier)