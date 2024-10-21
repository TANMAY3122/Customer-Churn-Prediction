# Customer Churn Prediction

This project aims to predict customer churn (whether a customer will leave the service) using machine learning techniques. By analyzing various customer features, we can identify those likely to churn, enabling businesses to take proactive measures.

## Table of Contents
1. [Dataset](#dataset)
2. [Tools & Libraries](#tools--libraries)
3. [Project Workflow](#project-workflow)
   - [Step 1: Data Loading and Preprocessing](#step-1-data-loading-and-preprocessing)
   - [Step 2: Exploratory Data Analysis (EDA)](#step-2-exploratory-data-analysis-eda)
   - [Step 3: Data Preparation](#step-3-data-preparation)
   - [Step 4: Model Building](#step-4-model-building)
   - [Step 5: Model Evaluation](#step-5-model-evaluation)
   - [Step 6: Cross-Validation and Hyperparameter Tuning](#step-6-cross-validation-and-hyperparameter-tuning)
   - [Step 7: Saving the Model](#step-7-saving-the-model)
4. [Results](#results)
5. [Future Work](#future-work)
6. [How to Run](#how-to-run)
7. [License](#license)
8. [Acknowledgments](#acknowledgments)

## Dataset

The dataset used in this project is from Kaggle, which includes information about customer behavior and services that could influence churn.
## Tools & Libraries

- Python 3.7+
- Pandas
- Scikit-Learn
- Matplotlib
- Seaborn
- Joblib (for model saving)

## Project Workflow

### Step 1: Data Loading and Preprocessing
In this step, we load the dataset using Pandas and perform initial preprocessing. We handle any missing values and clean the data. Categorical variables are encoded into numerical values to prepare them for modeling.

### Step 2: Exploratory Data Analysis (EDA)
We perform EDA to understand the data better. This includes visualizing distributions of key features, checking for correlations, and identifying patterns that may help in predicting churn.

### Step 3: Data Preparation
Data is split into features (X) and the target variable (y). We also divide the data into training and testing sets to evaluate the model's performance. Feature scaling is applied to standardize the numerical features.

### Step 4: Model Building
Two classification models are built:
- **Logistic Regression**: A simple model to predict the likelihood of customer churn.
- **Decision Tree**: A more complex model that can capture non-linear relationships in the data.

### Step 5: Model Evaluation
Both models are evaluated using metrics such as accuracy, precision, recall, and F1 score. Confusion matrices are generated to visualize the performance of each model.

### Step 6: Cross-Validation and Hyperparameter Tuning
To improve model performance and ensure generalization, cross-validation is implemented. GridSearchCV is used for hyperparameter tuning of the Decision Tree model to find the best parameters.

### Step 7: Saving the Model
Finally, the trained models are saved using Joblib for future use. This allows the models to be easily loaded and used for predictions on new data.

## Results

### Model Performance Metrics

| Model                   | Accuracy | Precision | Recall | F1 Score |
|-------------------------|----------|-----------|--------|----------|
| **Logistic Regression** | 0.80     | 0.75      | 0.65   | 0.70     |
| **Decision Tree**       | 0.85     | 0.80      | 0.78   | 0.79     |

### Confusion Matrices
The confusion matrices below represent the model’s performance in predicting churn (1: Churn, 0: No Churn).

- **Logistic Regression Confusion Matrix**:

![Logistic Regression Confusion Matrix](path-to-logistic-cm.png)

- **Decision Tree Confusion Matrix**:

![Decision Tree Confusion Matrix](path-to-tree-cm.png)

## Future Work

- **Model Deployment**: Implement a simple API using Flask or FastAPI to serve predictions from the trained model.
- **Feature Engineering**: Explore adding more derived features that may improve model performance.
- **Additional Models**: Experiment with other classification models like Random Forest, XGBoost, or Support Vector Machines for improved accuracy.

