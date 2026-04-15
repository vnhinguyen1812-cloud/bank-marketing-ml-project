# Bank Marketing Machine Learning Project

## Project Overview

This project applies machine learning techniques to predict customer responses to a bank marketing campaign. The goal is to help the bank improve customer engagement and optimize marketing strategies.

## Business Problem

Banks need to identify which customers are most likely to accept a marketing offer (open a new account). Predicting this behavior helps:

* Increase conversion rate
* Reduce marketing costs
* Improve customer targeting

## Dataset

* Source: Bank Marketing Dataset (Kaggle-based)
* ~40,000 customers
* 20+ features (demographics, financial status, campaign data)
* Target variable: Customer response (Yes/No)

## Methods Used

* Logistic Regression
* Decision Tree (with pruning)
* Random Forest (ensemble method)
* SMOTE for class imbalance handling

## Model Evaluation

* Accuracy
* ROC-AUC
* Brier Score
* Lift Analysis (business-focused metric)

## Key Insights

* Customer behavior is strongly influenced by past interactions and campaign intensity
* Ensemble models (Random Forest) provide better predictive performance
* Adjusting classification threshold improves business outcomes

## Project Structure

* `data/` → dataset
* `code/` → R scripts
* `report/` → project proposal & final report
* `presentation/` → slides

## How to Run

1. Open the R script in `code/`
2. Load dataset from `data/`
3. Run the script step-by-step

## Author

Nguyễn Hoàng Vân Nhi


