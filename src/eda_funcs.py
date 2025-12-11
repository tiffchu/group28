import click
import pandas as pd
import altair as alt
import os

def bar_plot(df: pd.DataFrame):
    if "species" not in df.columns:
        raise KeyError("Dataframe must contain 'species' column.")

    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("count()").title("Count"),
            y=alt.Y("species").title("Species"),
            color=alt.Color(
                "species",
                scale=alt.Scale(
                    domain=["setosa", "versicolor", "virginica"],
                    range=["blue", "orange", "green"],
                ),
            ).legend(None),
        )
        .properties(
            title=alt.TitleParams(
                text="Count of Iris Species",
                anchor="start",
                fontSize=15,
            )))


def pairwise_plot(df: pd.DataFrame):
    required_cols = ["species", "sepal_length", "sepal_width", "petal_length", "petal_width"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    return (
        alt.Chart(df)
        .mark_point(opacity=0.3, size=10)
        .encode(
            alt.X(alt.repeat("row")).type("quantitative"),
            alt.Y(alt.repeat("column")).type("quantitative"),
            color=alt.Color(
                "species",
                scale=alt.Scale(
                    domain=["setosa", "versicolor", "virginica"],
                    range=["blue", "orange", "green"],
                ),
            ).title("Species"),
        )
        .properties(width=150, height=150)
        .repeat(
            row=features,
            column=features,
        )
        .properties(
            title=alt.TitleParams(
                text="Relationship Between Iris Features",
                fontSize=18,
            )))
