import pandas as pd
import altair as alt

def make_pairwise_plot(train_data, color, columns):
    pairwise_plot = alt.Chart(train_data).mark_point(opacity=0.3, size=10).encode(
        alt.X(alt.repeat('column'), type='quantitative'),
        alt.Y(alt.repeat('row'), type='quantitative'),
        color = alt.Color(
        color, type="nominal",
        title = 'species',
        scale=alt.Scale(
            domain=['setosa', 'versicolor', 'virginica'],
            range=['blue', 'orange', 'green'])
        )).properties(
                width=150,
                height=150
            ).repeat(
        row=columns,
        column=columns)
    
    return pairwise_plot


