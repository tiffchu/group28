.DEFAULT_GOAL := help
.PHONY: help target cl env build run up stop docker-build-push docker-build-local clean all

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'


target: ## runs the targets: cl, env, build
	make cl
	make env
	make build

cl: ## create conda lock for multiple platforms
	# the linux-aarch64 is used for ARM Macs using linux docker container
	conda-lock lock \
		--file environment.yml \
		-p linux-aarch64 \
		-p linux-64 \
		-p osx-64 \
		-p osx-arm64 \
		-p win-64 

env: ## remove previous and create environment from lock file
	# remove the existing env, and ignore if missing
	conda env remove -n dockerlock || true
	conda-lock install -n dockerlock conda-lock.yml

build: ## build the docker image from the Dockerfile
	docker build -t dockerlock --file Dockerfile .

run: ## alias for the up target
	make up

up: ## stop and start docker-compose services
	# by default stop everything before re-creating
	make stop
	docker-compose up -d

stop: ## stop docker-compose services
	docker-compose stop

# docker multi architecture build rules (from Claude) -----

docker-build-push: ## Build and push multi-arch image to Docker Hub (amd64 + arm64)
	docker buildx build \
		--platform linux/amd64,linux/arm64 \
		--tag tiffchu/group28:latest \
		--tag tiffchu/group28:local-$(shell git rev-parse --short HEAD) \
		--push \
		.
		
docker-build-local: ## Build single-arch image for local testing (current platform only)
	docker build \
		--tag tiffchu/group28:local \
		.

#download and save data
data/raw/iris.csv: src/download_data.py
	python src/download_data.py --url="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv" --path="./data/raw/iris.csv"

#split data
data/processed/iris_test.csv data/processed/iris_train.csv: src/split_preprocess.py data/raw/iris.csv
	python src/split_preprocess.py --rawdata="./data/raw/iris.csv" --path="./data/processed" --test-size=0.3 --random-state=123

#create and save plots
results/figures/iris_species_barplot.png results/figures/iris_species_boxplot.png results/figures/iris_species_pairwise.png: src/eda.py data/processed/iris_train.csv
	python src/eda.py --training-data="./data/processed/iris_train.csv" --plot-to="./results/figures/"

#create and save models
results/models/decision_tree.pickle results/models/knn.pickle results/tables/confusion_matrix_ds.csv results/tables/confusion_matrix_knn.csv results/tables/ds_results.csv results/tables/knn_results.csv: src/model_train_and_evaluation.py data/processed/iris_train.csv data/processed/iris_test.csv
	python src/model_train_and_evaluation.py --training-data ./data/processed/iris_train.csv --test-data ./data/processed/iris_test.csv --models-to ./results/models --tables-to ./results/tables

#render report to html
reports/iris_predictor_report.html: reports/iris_predictor_report.qmd 
	quarto render reports/iris_predictor_report.qmd --to html

all: \
	data/raw/iris.csv \
	data/processed/iris_test.csv \
	data/processed/iris_train.csv \
	results/figures/iris_species_barplot.png \
	results/figures/iris_species_boxplot.png \
	results/figures/iris_species_pairwise.png \
	results/models/decision_tree.pickle \
	results/models/knn.pickle \
	results/tables/confusion_matrix_ds.csv \
	results/tables/confusion_matrix_knn.csv \
	results/tables/ds_results.csv \
	results/tables/knn_results.csv \
	reports/iris_predictor_report.html 

clean: 
	rm -rf data/raw
	rm -rf data/processed
	rm -rf results/figures
	rm -rf results
	rm -rf reports/iris_predictor_report.html