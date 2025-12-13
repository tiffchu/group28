import altair as alt
import pandas as pd
import sys, os
import pytest


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.make_boxplot import make_boxplot 

@pytest.fixture

def sample_data() -> pd.DataFrame:
    """Fixture to create a simple, reproducible DataFrame for testing."""
    data = {
        "feature_a": range(100),
        "species": [i * 2 for i in range(100)],
        "target": [i % 2 for i in range(100)],
    }
    return pd.DataFrame(data)


def test_make_boxplot_return_chart(sample_data):
    plot = make_boxplot(sample_data, "feature_a", "Feature A")
    
    assert isinstance(plot, alt.Chart), "Output should be a chart."

def test_make_boxplot_properties(sample_data):
    plot = make_boxplot(sample_data, "feature_a", "Feature A")
    chart_dict = plot.to_dict()

    assert chart_dict["mark"]['type'] == 'boxplot', "This chart should be a boxplot"
    assert chart_dict['encoding']['x']['field'] == 'feature_a', "x-axis should be feature_a"
    assert chart_dict['encoding']['x']['title'] == 'Feature A', "x-axis label should be Feature A"
    
