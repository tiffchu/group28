import pandas as pd
import altair as alt

def make_bar_plot(train_data, xVar, xLab, yVar, yLab, color):
    barplot = alt.Chart(train_data).mark_bar().encode(
        x = alt.X(xVar, type='quantitative').title(xLab),
        y = alt.Y(yVar, type='nominal').title(yLab),
        color = alt.Color(
            color, type='nominal',
            scale=alt.Scale(
                domain=['setosa', 'versicolor', 'virginica'],
                range=['blue', 'orange', 'green'])
            ).legend(None)
        ).properties(
        title = alt.TitleParams(
            text = "Count of Iris Species",
            anchor = 'start',
            fontSize = 15))
    return barplot