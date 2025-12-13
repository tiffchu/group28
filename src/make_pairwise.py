import pandas as pd
import altair as alt

# def make_pairwise_plot(train_data, color, columns):
#     pairwise = alt.Chart(train_data).mark_point(opacity=0.3, size=10).encode(
#         x=alt.X(alt.repeat('column'), type='quantitative'),
#         y=alt.Y(alt.repeat('row'), type='quantitative'),
#         color = alt.Color(
#         color, type="nominal",
#         scale=alt.Scale(
#             domain=['setosa', 'versicolor', 'virginica'],
#             range=['blue', 'orange', 'green']),
#             title = 'species'
#         ))
    
#     pairwise_plot= pairwise.repeat(
#         row=columns,
#         column=columns
#         ).properties(
#                 width=150,
#                 height=150
        #)#.properties(title = 'Pairwise scatterplot of species')
    
def make_pairwise_plot(train_data, color, columns):
    pairwise_plot = alt.Chart(train_data).mark_point(opacity=0.3, size=10).encode(
        alt.X(alt.repeat('row')).type('quantitative'),
        alt.Y(alt.repeat('column')).type('quantitative'),
        color = alt.Color(
        color,
        scale=alt.Scale(
            domain=['setosa', 'versicolor', 'virginica'],
            range=['blue', 'orange', 'green'])
        ).title("Species")
            ).properties(
                width=150,
                height=150
            ).repeat(
                column=columns,
                row=columns
            )
    return pairwise_plot




