# Customer Churn Prediction

## Overview
This project builds an end-to-end machine learning pipeline to predict customer churn using telecom customer data. The goal is to identify high-risk customers and support data-driven retention strategies.

## Dataset
Telco Customer Churn Dataset with 7,043 records and 21 features including customer demographics, services, billing details, and churn status.

## Tools & Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- VS Code

## Data Preprocessing
- Converted target variable (Churn) from Yes/No to 1/0
- Removed non-informative features
- Encoded categorical variables using one-hot encoding
- Performed train-test split (80/20)

## Models Implemented
- Logistic Regression
- Random Forest Classifier

## Model Evaluation
- Accuracy
- Confusion Matrix
- Classification Report

## Results
- Logistic Regression achieved ~82% accuracy and performed better on high-dimensional data
- Random Forest slightly underperformed due to feature dimensionality

## Business Insights
- Month-to-month contract customers are more likely to churn
- High monthly charges increase churn risk
- Short-tenure customers require early retention efforts

## Conclusion
Logistic Regression proved to be the most effective model for churn prediction in this dataset and can help businesses proactively reduce customer attrition.

