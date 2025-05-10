# Customer Review Classification

This repository contains a machine learning project developed as part of an undergraduate seminar course. The main objective of the project is to demonstrate core MLOps practices using a real-world NLP task: classifying customer reviews.

## Project Objective

The goal of this project is to build and deploy a machine learning pipeline for classifying customer reviews as positive or negative. The project also serves as a case study for applying MLOps concepts such as modular ETL and training pipelines, experiment tracking, and model reproducibility.

## Tech Stack

- **Programming Language**: Python  
- **NLP Library**: spaCy  
- **ML Library**: scikit-learn  
- **Data Storage**: PostgreSQL  
- **Experiment Tracking**: MLflow  
- **Orchestration & Pipelines**: Custom ETL and training pipelines with clear modular structure  

## Features

- Modular ETL pipeline to extract and preprocess customer reviews from PostgreSQL
- Text preprocessing with spaCy (tokenization, lemmatization, etc.)
- Training pipeline using scikit-learn (Logistic Regression, Gradient Boosting Classifier and SGD Classifier)
- Model evaluation with metrics like accuracy, f1 score, precision, and recall
- MLflow integration for:
  - Tracking experiments and hyperparameters
  - Comparing different model runs
  - Storing and loading trained models


