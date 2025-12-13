import pandas as pd
import altair as alt

def make_pairwise_plot(train_data, color, column, row):
    pairwise_plot = alt.Chart(train_data).mark_point(opacity=0.3, size=10).encode(
        alt.X(alt.repeat('row'), type='quantitative'),
        alt.Y(alt.repeat('column'), type='quantitative'),
        color = alt.Color(
        color, type="nominal",
        scale=alt.Scale(
            domain=['setosa', 'versicolor', 'virginica'],
            range=['blue', 'orange', 'green'])
        ).title("Species")
            ).properties(
                width=150,
                height=150
            ).repeat(
                column=column,
                row=row
            )
    return pairwise_plot


