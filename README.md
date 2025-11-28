# IrisSpeciesPredictor

## Contributors
- Gaurang Ahuja
- Wai Yan Lee
- Uyen Nguyen Nguyen
- Tiffany Chu

## Project Summary
The aim of this project is to look at the Iris dataset, and figure out if certain features are useful for the classification of iris species. We hope to answer "Can we predict the Iris species using petal and sepal measurements?". By leveraging data analysis and machine learning techniques, we seek to accurately predict the species of iris flowers from given measurements.

## Dependencies
Please look at the environment.yml file for a list of dependencies required to run this project. Can also refer to the conda lock files.
Sample command: `conda env create --name <my-env-name> --file <path/to/environment.yml>`

## How to use container image
To reproduce the analysis and its results, you can use the provided Docker container image. Follow the steps below:

1. Clone the Repository and install Docker 
2. Build the Container Image:
- After cloning the repo, to build the image locally, navigate to the project directory and run:
```bash
docker compose run #start the container
```
Access JupyterLab to work with the analysis notebook in the container:

```bash
docker compose up
```

Once the container is running, you can access JupyterLab by navigating to the URL in the terminal (looks like http://127.0.0.1:8888/lab?token= ...) in your web browser. You can then open the notebook analysis file to run the analysis.
```bash
docker compose down #stop the container
```

## Licenses
This project is distributed under the licenses listed in `LICENSE`:
- MIT License