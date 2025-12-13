import click
import pandas as pd
import altair as alt
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.make_boxplot import make_boxplot
from src.make_barplot import make_bar_plot
from src.make_pairwise import make_pairwise_plot

@click.command()
@click.option('--training-data', type=str, help="Path to processed training data", default = './data/processed/iris_train.csv', show_default=True)
@click.option('--plot-to', type=str, help="Path to directory where the plot will be written to", default = './results/figures/', show_default=True)
def main(training_data, plot_to):
    ''' 
    Generate EDA plots and save them in results/figure folder.
    '''
    train_data = pd.read_csv(training_data)

    bar_plot = make_bar_plot(training_data, 'count()', 'Count', 'species', 'Species', 'species')

    pairwise_plot = make_pairwise_plot(training_data, 'species', ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    
    sepal_length_BP = make_boxplot(train_data, "sepal_length", "Sepal Length")

    sepal_width_BP = make_boxplot(train_data, "sepal_width", "Sepal Width")

    petal_length_BP = make_boxplot(train_data, "petal_length", "Petal Length")
    
    petal_width_BP = make_boxplot(train_data, "petal_width", "Petal Width")
    
    box_plot = (sepal_length_BP & sepal_width_BP & petal_length_BP & petal_width_BP).properties(
        title = alt.TitleParams(
            text = "Iris Feature Distributions by Species",
            fontSize = 14))

    os.makedirs(plot_to, exist_ok = True)
        
    bar_plot.save(os.path.join(plot_to, "iris_species_barplot.png"), scale_factor=2.0)
    box_plot.save(os.path.join(plot_to, "iris_species_boxplot.png"), scale_factor=2.0)
    pairwise_plot.save(os.path.join(plot_to, "iris_species_pairwise.png"), scale_factor=2.0)

    print('The EDA plots are saved in ./results/figures directory.')

if __name__ == '__main__':
    main()
