# IrisSpeciesPredictor

## Contributors
- Gaurang Ahuja
- Wai Yan Lee
- Uyen Nguyen Nguyen
- Tiffany Chu

## Project Summary
The aim of this project is to look at the Iris dataset, and figure out if certain features are useful for the classification of iris species. We hope to answer "Can we predict the Iris species using petal and sepal measurements?". By leveraging data analysis and machine learning techniques, we seek to accurately predict the species of iris flowers from given measurements.

## Dependencies
Please look at the environment.yml file for a list of dependencies required to run this project. You can also refer to the `conda-lock.yml` file.
Sample command: `conda env create --name <my-env-name> --file <path/to/environment.yml>`

## Prerequisites
Before running the project in Docker, ensure you have:
- Docker installed

## Using the Docker Container
You can reproduce the analysis and run the notebook in a pre configured Docker container.

1. Clone the repository:
    - `git clone git@github.com:tiffchu/group28.git`
    - `cd group28`

2. Build the Docker image
    - `docker-compose build`
This creates the `dockerlock` image with all required packages installed.

3. Start the container and JupyterLab
    - `docker-compose up`
This will start the container, activate the `dockerlock` environment and launch JupyterLab inside the container

4. Open JupyterLab in your browser at
    - `http://localhost:8888/lab`

5. Open and run notebooks
    1. In JupyterLab, open the `[text](iris_predictor_report.ipynb)`
    2. Run all cells

6. Stop the container
Return to the terminal where `docker-compose up` is running and press `Ctrl+C`

## Licenses
This project is distributed under the licenses listed in `LICENSE`:
- MIT License