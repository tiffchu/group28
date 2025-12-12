import pandas as pd
import altair as alt
import os 
import sys
import pytest
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.eda import (
    bar_plot,
    box_plot,
    pairwise_plot)

@pytest.fixture

def iris_small_df():
    """mock dataset for testing"""
    return pd.DataFrame({
        "sepal_length": [5.1, 6.3, 5.9],
        "sepal_width": [3.5, 2.9, 3.0],
        "petal_length": [1.4, 5.6, 5.1],
        "petal_width": [0.2, 1.8, 1.8],
        "species": ["setosa", "virginica", "versicolor"]
    })


@pytest.fixture

#check plot types and encoding

def test_barplot_type(iris_small_df):
    chart = bar_plot(iris_small_df)
    assert isinstance(chart, alt.Chart)

def test_barplot_species_encoding(iris_small_df):
    chart = bar_plot(iris_small_df)
    encodings = chart.to_dict()["encoding"]
    assert "y" in encodings
    assert encodings["y"]["field"] == "species"
    assert "x" in encodings

def test_create_pairwise_plot_type(iris_small_df):
    chart = pairwise_plot(iris_small_df)
    assert isinstance(chart, alt.Chart)

def test_pairwise_plot_repeat_structure(iris_small_df):
    chart = pairwise_plot(iris_small_df)
    chart_dict = chart.to_dict()

    assert "repeat" in chart_dict
    assert len(chart_dict["repeat"]["row"]) == 4
    assert len(chart_dict["repeat"]["column"]) == 4

#boxplot type 
def test_boxplot_type(iris_small_df):
    chart = box_plot(iris_small_df)
    assert isinstance(chart, alt.VConcatChart) or isinstance(chart, alt.HConcatChart)


def test_boxplot_chart(iris_small_df):
    chart = box_plot(iris_small_df)
    chart_dict = chart.to_dict()

    assert "concat" in chart_dict
    assert len(chart_dict["concat"]) == 4   # 4 features → 4 boxplots

#barplot and boxplot missing columns 

def test_barplot_missing_species_columns():
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})

    with pytest.raises(KeyError):
        bar_plot(df)


def test_boxplots_missing_features():
    df = pd.DataFrame({"species": ["a", "b", "c"]})

    with pytest.raises(KeyError):
        bar_plot(df)

#see which tests pass, type in cli:  pytest -v