import os
import sys
import pickle
import pytest
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

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

@pytest.fixture
def dummy_data():
    X, y = make_classification(
        n_samples=50,
        n_features=5,
        n_informative=3,
        random_state=123
    )
    return X, y

def test_save_model_content_similar_to_created_model(dummy_data, tmp_path):
    """This test include a dummy data to see if the content (hyperparameter) in the model is exactly the same after we unload it"""

    X, y = dummy_data
    sample_model = DecisionTreeClassifier(max_depth=10, random_state=123)
    sample_model.fit(X, y)

    path = tmp_path/"sample_model.pickle"
    save_model(sample_model, path)
    load_model = pickle.loads(path.read_bytes())

    assert load_model.get_params() == sample_model.get_params()
    assert (load_model.predict(X) == sample_model.predict(X)).all()
    assert load_model.max_depth == 10
    assert load_model.get_depth() == sample_model.get_depth()

#depth of decision tree -> if its none or someother value we input
