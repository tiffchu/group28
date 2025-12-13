import pandas as pd
import altair as alt
import os 
import sys
import pytest
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.make_barplot import make_bar_plot

@pytest.fixture

def sample_data():
    """mock dataset for testing"""
    return pd.DataFrame({
        "lab1": [5.1, 6.3, 5.9],
        "lab2": [3.5, 2.9, 3.0],
        "lab3": [1.4, 5.6, 5.1],
        "lab4": [0.2, 1.8, 1.8],
        "categories": ["a", "b", "c"]
    })

def test_create_bar_plot_type(sample_data):
    '''
    check if bar plot is an altair chart object
    '''
    plot = make_bar_plot(sample_data, 'categories', 'xlabel', 'value', 'ylabel', 'categories')
    assert isinstance(plot, alt.Chart), 'not a chart'

def test_bar_plot_mark(sample_data):
    '''
    check if chart is using bar mark
    '''
    plot = make_bar_plot(sample_data, 'categories', 'xlabel', 'value', 'ylabel', 'categories')
    assert plot.mark == 'bar', 'not the right mark, should be bar'

def test_bar_plot_mark(sample_data):
    '''
    check if chart is using x, y, and color encodings
    '''
    plot = make_bar_plot(sample_data, 'categories', 'xlabel', 'value', 'ylabel', 'categories')
    enc = plot.encoding
    assert enc.x is not None, 'not using the x encoding'
    assert enc.y is not None, 'not using the y encoding'
    assert enc.color is not None, 'not using the color encoding'

