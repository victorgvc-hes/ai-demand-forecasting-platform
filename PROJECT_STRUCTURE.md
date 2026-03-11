# Project Structure

This document explains the folder structure of the AI Demand Forecasting Platform project.

## Root Directory

```text
ai-demand-forecasting-platform/

## Main files in the root:

README.md — project overview, setup, results, and usage

LICENSE — MIT license

requirements.txt — Python dependencies

.gitignore — files and folders excluded from version control

case_study_demand_forecasting.md — one-page project case study

CONTRIBUTING.md — contribution guidelines

PROJECT_STRUCTURE.md — repository structure reference

## Folder Overview
data/

Local data storage only.

data/raw/ — raw Kaggle CSV files

data/processed/ — processed parquet datasets and training-ready data

These folders are not intended to be committed to GitHub.

models/

Local model artifacts.

Example:

lightgbm_model.pkl

This folder stores serialized trained models for inference and API serving.

pipelines/

Contains the main execution scripts of the project.

pipelines/ingestion/

Scripts for reading and validating raw data.

Example:

load_m5.py

pipelines/features/

Scripts for feature engineering.

Example:

build_training_features.py

pipelines/training/

Scripts for baseline modeling and machine learning training.

Examples:

baseline_seasonal_naive.py

train_lightgbm.py

pipelines/inference/

Scripts for batch prediction and reusable inference.

Example:

run_inference.py

pipelines/evaluation/

Scripts for business and model evaluation.

Example:

business_impact.py

pipelines/visualization/

Scripts for generating plots and visual outputs.

Example:

forecast_plot.py

models/

Local model artifacts.

Example:

lightgbm_model.pkl

This folder stores serialized trained models for inference and API serving.

pipelines/

Contains the main execution scripts of the project.

pipelines/ingestion/

Scripts for reading and validating raw data.

Example:

load_m5.py

pipelines/features/

Scripts for feature engineering.

Example:

build_training_features.py

pipelines/training/

Scripts for baseline modeling and machine learning training.

Examples:

baseline_seasonal_naive.py

train_lightgbm.py

pipelines/inference/

Scripts for batch prediction and reusable inference.

Example:

run_inference.py

pipelines/evaluation/

Scripts for business and model evaluation.

Example:

business_impact.py

pipelines/visualization/

Scripts for generating plots and visual outputs.

Example:

forecast_plot.py

reports/

Stores generated outputs, metrics, plots, and markdown summaries.

Examples:

baseline metrics

LightGBM metrics

validation predictions

feature importance

business impact summaries

forecast vs actual plots

