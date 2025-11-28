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

## How to use, update, and work with container image
This project uses a Docker-based computational environment to ensure that all team members have a consistent and reproducible setup.
The environment is defined by: A Dockerfile (condaforge/miniforge3:latest base image), an environment.yml file, and a docker-compose.yml file for startup

Build the Container Image

After cloning the repo, to build the image:

```bash
docker compose up #start the container
```
Access JupyterLab:

Once the container is running, you can access JupyterLab by navigating to http://localhost:8888 or get the URL in the terminal (looks like http://127.0.0. ...) in your web browser. You can then open the notebook analysis file to run the analysis.
```
docker compose down #stop the container
```

## Licenses
This project is distributed under the licenses listed in `LICENSE`:
- MIT License