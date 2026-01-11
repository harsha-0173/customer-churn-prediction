# Customer Churn Prediction

## Objective
To analyze customer behavior and predict churn using machine learning techniques, helping businesses take proactive retention measures.

## Dataset
Telco Customer Churn Dataset containing 7,043 customer records and 21 features including demographics, services used, billing details, and churn status.

## Tools & Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- VS Code

## Data Preprocessing
- Converted target variable (Churn) from Yes/No to 1/0
- Removed non-informative features (customerID)
- Encoded categorical variables using one-hot encoding
- Performed train-test split (80/20)

## Models Used
1. Logistic Regression  
2. Random Forest Classifier  

## Model Performance
- Logistic Regression Accuracy: ~82%
- Random Forest Accuracy: ~85% (better performing model)

## Evaluation Metrics
- Accuracy
- Confusion Matrix
- Precision, Recall, F1-score

## Key Insights
- Customers with month-to-month contracts show higher churn
- Higher monthly charges increase churn probability
- Short-tenure customers are at higher risk

## Conclusion
Random Forest outperformed Logistic Regression and can be used to identify high-risk customers, enabling targeted retention strategies.
