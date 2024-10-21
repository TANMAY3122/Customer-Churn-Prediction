# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib

# Load the dataset
data = pd.read_csv('telecom_churn_data.csv')

# Data preprocessing (Assuming no missing values for simplicity)
data.drop(['customerID'], axis=1, inplace=True)

# Encode categorical variables
data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})
data = pd.get_dummies(data, drop_first=True)

# Split data into features and target
X = data.drop('Churn', axis=1)
y = data['Churn'].map({'Yes': 1, 'No': 0})

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression model
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train, y_train)

# Train Decision Tree model
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

# Evaluate both models
y_pred_log = log_model.predict(X_test)
y_pred_tree = tree_model.predict(X_test)

# Metrics calculation
def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

log_metrics = evaluate_model(y_test, y_pred_log)
tree_metrics = evaluate_model(y_test, y_pred_tree)

# Displaying metrics
print("Logistic Regression Metrics:")
print(f"Accuracy: {log_metrics[0]}, Precision: {log_metrics[1]}, Recall: {log_metrics[2]}, F1 Score: {log_metrics[3]}")
print("\nDecision Tree Metrics:")
print(f"Accuracy: {tree_metrics[0]}, Precision: {tree_metrics[1]}, Recall: {tree_metrics[2]}, F1 Score: {tree_metrics[3]}")

# Confusion matrix for both models
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_test, y_pred_log), annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression Confusion Matrix')

plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y_test, y_pred_tree), annot=True, fmt='d', cmap='Blues')
plt.title('Decision Tree Confusion Matrix')

plt.show()

# Model tuning using GridSearchCV (optional)
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search_tree = GridSearchCV(estimator=tree_model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search_tree.fit(X_train, y_train)

# Best parameters
print(f"Best Parameters for Decision Tree: {grid_search_tree.best_params_}")

# Saving the best model
joblib.dump(log_model, 'logistic_regression_model.pkl')
joblib.dump(tree_model, 'decision_tree_model.pkl')
