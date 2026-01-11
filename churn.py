import pandas as pd

print("Customer Churn Project Started")

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nChurn value counts:")
print(df['Churn'].value_counts())

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

print("\nChurn after conversion:")
print(df['Churn'].value_counts())

df = df.drop('customerID', axis=1)
print("\ncustomerID column removed")

df = pd.get_dummies(df, drop_first=True)

print("\nData after converting text to numbers:")
print(df.shape)

X = df.drop('Churn', axis=1)
y = df['Churn']

print("\nX shape:", X.shape)
print("y shape:", y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("\nModel training completed")

accuracy = model.score(X_test, y_test)
print("\nModel Accuracy:", accuracy)

from sklearn.metrics import confusion_matrix, classification_report

y_pred = model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

print("\nRandom Forest training completed")

rf_accuracy = rf_model.score(X_test, y_test)
print("Random Forest Accuracy:", rf_accuracy)
