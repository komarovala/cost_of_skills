==============
Skills Cost Calculator
==============

A project for analyzing and evaluating the cost of professional skills based on job vacancy data.

## Project Description

This system is designed to:
- Analyze job vacancies and required skills across various professional domains
- Calculate the "cost" of each skill based on offered salaries
- Predict salary ranges based on skill combinations

## Technologies

- Python 3.x
- PyTorch - for optimization and skill cost calculation
- CatBoost - for building regression models
- Pandas - for data processing
- NumPy - for numerical computations
- Scipy - for optimization
- Optuna - for hyperparameter optimization

## Installation

1. Clone the repository
2. Install the dependencies
3. Run the script

## Project Structure

- `notebooks/`
  - `2024-01-11-test-gradio.ipynb` - Gradio interface for the calculator
  - `2024-01-13-interval-learn.ipynb` - training models with interval prediction
  - `2024-01-13-point-learn.ipynb` - training models with point prediction

## Core Functions

- `get_all_skills()` - extracts unique skills from data
- `get_matrix()` - builds feature matrix for training
- `calc_torch_interval()` - calculates skill costs using PyTorch
- `calc_catboost_interval()` - calculates skill costs using CatBoost
- `calc_torch_point()` - calculates skill costs using PyTorch

## Usage

1. Load vacancy data
2. Run model training
3. Use Gradio interface to calculate costs for skill combinations