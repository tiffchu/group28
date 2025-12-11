import altair as alt
import pandas as pd 

def make_boxplot(train_data: pd.DataFrame, xVar, xLab):
    """
    This function creates a boxplot given inputs. 
    
    Parameters
    ----------
    train_data : pandas.DataFrame
        The validated and cleaned input DataFrame.
    xVar : str
        x-axis variable to plot
    xLab : str
        Label for x-axis

    Returns
    --------
    altair boxplot chart

    Examples
    --------
    >>> plot = make_boxplot(train_data, "sepal_length", "Sepal Length")
    >>> plot2 =make_boxplot(train_data, "sepal_width", "Sepal Width")
    """

    chart = alt.Chart(train_data).mark_boxplot().encode(
            x = alt.X(xVar).title(xLab),
            y = alt.Y("species").title("Species"),
            color = alt.Color("species",
            scale=alt.Scale(
                    domain=['setosa', 'versicolor', 'virginica'],
                    range=['blue', 'orange', 'green'])
            ).legend(None))

    return chart
