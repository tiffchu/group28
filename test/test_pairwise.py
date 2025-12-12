import pandas as pd
import altair as alt
import os 
import sys
import pytest
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.make_pairwise import make_pairwise_plot

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

def test_create_pairwise_plot_type(sample_data):
    '''
    check if pairwise plot is a chart object
    '''
    plot = make_pairwise_plot(sample_data, 'categories', ['lab1', 'lab2', 'lab3', 'lab4'], ['lab1', 'lab2', 'lab3', 'lab4'])
    assert isinstance(plot, alt.RepeatChart), 'not a chart'

def test_make_pairwise_properties(sample_data):
    '''
    test if pairwise properties are correct
    '''
    plot = make_pairwise_plot(sample_data, 'categories', ['lab1', 'lab2', 'lab3', 'lab4'], ['lab1', 'lab2', 'lab3', 'lab4'])
    chart_dict = plot.to_dict()
    assert chart_dict['spec']['mark']['type'] == 'point', 'This chart should be repeated scatterplots (points)'

    assert 'repeat' in chart_dict, "Repeat (faceting) section missing"
    assert chart_dict['repeat']['row'] == ['lab1', 'lab2', 'lab3', 'lab4'], 'Repeat must be applied to the correct variables'
    assert chart_dict['repeat']['column'] == ['lab1', 'lab2', 'lab3', 'lab4'], 'Repeat must be applied to the correct variables'

    color_encoding = chart_dict['spec']['encoding']['color']
    assert color_encoding['field'] == 'categories', '.encode(color=..) must be applied to categorical feature'


    