
# Paisa Bazaar Credit Score Classification

## Description

This project addresses Paisa Bazaar's critical need to enhance credit risk assessment. It develops a robust machine learning model to accurately predict a customer's credit score category ('Poor', 'Standard', or 'Good') based on comprehensive financial and personal data. The model provides a reliable, interpretable, and actionable tool for informed, data-driven decisions, improving risk management and overall business outcomes.

---

## Table of Contents
- [Project Introduction](#project-introduction)
- [Dataset Description](#dataset-description)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Data Preparation](#data-preparation)
- [Modeling Phase](#modeling-phase)
- [Evaluation Metric](#evaluation-metric)
- [Conclusion](#conclusion)
- [Project Structure](#project-structure)
- [Notable Techniques Used](#notable-techniques-used)
- [Libraries and Technologies](#libraries-and-technologies)
- [Dataset Source](#dataset-source)

## Project Introduction

The primary challenge for Paisa Bazaar is to accurately and efficiently assess the credit risk of its customers to facilitate loan approvals and offer suitable financial products. Inaccurate assessment can lead to significant financial losses from loan defaults or lost business opportunities from incorrectly rejecting creditworthy applicants.

This project aims to solve this problem by building a machine learning classification model to predict a customer's credit score category ('Poor', 'Standard', or 'Good') based on their financial and personal data. The model must be reliable, interpretable, and provide actionable insights to help Paisa Bazaar make informed, data-driven decisions, thereby enhancing their risk management framework and improving overall business outcomes.

## Dataset Description

The project uses a comprehensive dataset (`dataset-2.csv`) containing various customer attributes, including demographic information, income details, loan history, credit card usage, and payment behavior.

Key information about the dataset:
* **Dimensions:** The dataset contains 100,000 rows and 28 columns.
* **Data Types:** Columns include `int64`, `float64`, and `object` (categorical/text) data types. `object` columns require encoding for machine learning.
* **Target Variable:** The `Credit_Score` column is the target variable, an `object` type with categories 'Good', 'Standard', and 'Poor', indicating a classification problem.
* **Missing Values:** Significant missing values are present in columns like `Num_of_delayed_payment`, `Num_Credit_Inquiries`, `Credit_Mix`, and `Amount_invested_monthly`. `Type_of_Loan` has a particularly high number of nulls.
* **Unique Identifier:** `Customer_ID` is a unique identifier and is dropped before modeling.
* **Data Quality:** No duplicate rows were found, but some columns might contain placeholder or garbage values.

| Variable | Description |
| :--- | :--- |
| **Customer_ID** | Unique identifier for each customer. |
| **Month** | Month of the data record. |
| **Age** | Age of the customer in years. |
| **Occupation** | The customer's profession. |
| **Annual_Income** | The total annual income of the customer. |
| **Monthly_Inhand_Salary** | The customer's net monthly salary. |
| **Num_Bank_Accounts** | The number of bank accounts the customer holds. |
| **Num_Credit_Card** | The number of credit cards the customer possesses. |
| **Interest_Rate** | The average interest rate on the customer's credit products. |
| **Num_of_Loan** | The number of loans the customer has. |
| **Type_of_Loan** | The types of loans the customer has taken. (e.g., Personal, Home) |
| **Delay_from_due_date**| The average number of days a payment is delayed past its due date. |
| **Num_of_delayed_payment**| The total number of payments the customer has delayed. |
| **Changed_Credit_Limit** | The percentage change in the customer's credit limit. |
| **Num_Credit_Inquiries** | The number of credit inquiries made by the customer. |
| **Credit_Mix** | The mix of credit products (e.g., Good, Standard, Bad). |
| **Outstanding_Debt** | The total amount of outstanding debt. |
| **Credit_Utilization_Ratio**| The ratio of credit used to the total available credit. |
| **Credit_History_Age**| The age of the customer's credit history. |
| **Payment_of_Min_Amount**| Indicates if the customer pays the minimum amount due. |
| **Total_EMI_per_month** | The total Equated Monthly Installment (EMI) paid by the customer. |
| **Amount_invested_monthly**| The amount the customer invests monthly. |
| **Payment_Behaviour**| Categorization of the customer's payment behavior. |
| **Monthly_Balance**| The average monthly balance in the customer's account. |
| **Credit_Score**| The customer's credit score category (**Target Variable**). |

## Exploratory Data Analysis (EDA)

The project includes an in-depth exploratory data analysis (EDA) with over 15 visualizations to uncover patterns, relationships, and key drivers of credit scores.

Key findings from EDA:
* **Credit Score Distribution:** The dataset is somewhat imbalanced, with 'Standard' being the most frequent credit score category, followed by 'Good' and then 'Poor'.
* **Annual Income Distribution:** The distribution of `Annual_Income` is heavily right-skewed, with most customers having lower incomes and a long tail of very high earners. This suggests income is a key differentiator, and transformation might be needed for modeling.
* **Customer Age Distribution:** The customer base is widely distributed by age, with the highest concentration between 25 and 40 years old.
* **Credit Score vs. Annual Income:** A clear positive relationship exists; median annual income increases with better credit scores, making income a powerful predictor.
* **Credit Score vs. Delay from Due Date:** A strong negative correlation is observed. 'Poor' credit scores are associated with higher and wider distributions of payment delays, while 'Good' scores show delays concentrated near zero. This is a critical indicator for risk models and early warning systems.
* **Credit Mix Distribution by Credit Score:** There's a strong association where a 'Good' credit mix overwhelmingly correlates with 'Good' credit scores, and a 'Bad' credit mix predominantly correlates with 'Poor' credit scores. This feature is a powerful indicator of credit health.

## Data Preparation

The project involved a structured data science workflow, including data exploration, cleaning, wrangling, and feature engineering.

* **Dropped Irrelevant Columns:** `Customer_ID`, `Name`, `SSN`, `Month`, `ID`, `Type_of_Loan`, and `Monthly_Inhand_Salary` were removed due to lack of predictive value, high missingness/messiness, or multicollinearity.
* **Cleaned and Converted Numerical Columns:** Columns like `Age`, `Annual_Income`, `Num_Bank_Accounts`, `Num_of_Loan`, `Num_of_delayed_payment`, `Num_Credit_Inquiries`, `Amount_invested_monthly`, `Monthly_Balance`, `Changed_Credit_Limit`, and `Outstanding_Debt` were converted to numeric types, handling non-numeric characters and coercing errors to `NaN`.
* **Engineered `Credit_History_Age_Months`:** The `Credit_History_Age` column (e.g., "X Years and Y Months") was converted into a numerical feature representing total credit history in months.
* **Standardized Categorical Placeholders:** Inconsistent placeholder values (e.g., '_______', '#F!', '!@9#%8') in `Credit_Mix`, `Payment_Behaviour`, and `Occupation` were replaced with standard `NaN` values. 'NM' in `Payment_of_Min_Amount` was mapped to 'No'.
* **Missing Value Handling:** Missing values will be handled through strategic imputation during the feature engineering stage.
* **Categorical Encoding:** Categorical variables were encoded using One-Hot Encoding.
* **Feature Scaling:** Numerical features were scaled to prepare data for modeling.

## Modeling Phase

The project implements and evaluates three powerful classification algorithms:
* **Logistic Regression** (as a baseline)
* **Random Forest Classifier**
* **XGBoost Classifier**

Models are fine-tuned using `GridSearchCV` for hyperparameter optimization and cross-validation to ensure robustness.

## Evaluation Metric

As a classification problem, model performance is assessed using a suite of metrics with a special focus on their business implications:
* **Accuracy Score**
* **Precision Score**
* **Recall Score**
* **F1-Score**
* **Classification Report**
* **Confusion Matrix**

## Conclusion

The final model provides Paisa Bazaar with a powerful, data-driven tool to automate and improve the accuracy of credit score classification. This can lead to better risk management, reduced default rates, faster loan approvals, and the ability to offer more targeted financial products to customers, ultimately driving business growth and customer satisfaction. The feature importance analysis confirmed that the model's predictions are driven by financially sound and interpretable factors, building trust in its decisions. The saved model is a valuable asset for Paisa Bazaar, enabling automation of loan application processing, improved risk assessment accuracy, personalized financial products, and a consistent basis for credit decisions.

## Project Structure

```

.
├── data/                       \# Dataset CSV files (e.g., dataset-2.csv)
├── notebooks/                  \# Jupyter notebooks for EDA and modeling
│   └── Paisa Bazaar Credit Score Classification.ipynb
├── models/                     \# Saved machine learning models and artifacts
├── src/                        \# Python scripts for preprocessing and modeling
│   ├── data\_cleaning.py
│   ├── feature\_engineering.py
│   ├── modeling.py
│   └── evaluation.py
├── README.md                   \# Project documentation (this file)
├── requirements.txt            \# Dependencies list
└── results/                    \# Model evaluation visualizations and metrics

```

## Notable Techniques Used

* **Data Wrangling & Cleaning** (handling missing values, placeholder values, data type conversion)
* **Feature Engineering** (e.g., converting `Credit_History_Age` to months)
* **One-Hot Encoding** (for categorical variables)
* **Standard Scaling** (for numerical features)
* **Logistic Regression**
* **Random Forest Classifier**
* **XGBoost Classifier**
* **GridSearchCV** (for hyperparameter optimization and cross-validation)
* **Univariate, Bivariate, Multivariate Analysis**
* **Hypothesis Testing** (ANOVA, Chi-Square)
* **Model Persistence** (`joblib`)

## Libraries and Technologies

* **Python**
* **Pandas**, **NumPy** (data manipulation and analysis)
* **Matplotlib**, **Seaborn**, **Plotly** (data visualization)
* **Scikit-learn** (machine learning preprocessing, models, and evaluation)
* **XGBoost** (gradient boosting)
* **SciPy** (statistical testing)
* **Jupyter Notebooks** (interactive development)

## Dataset Source

* The dataset used is `dataset-2.csv`, which was loaded from a Google Drive link within the notebook.
```
