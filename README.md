# Telco Customer Churn Analysis

## Overview

This project performs Exploratory Data Analysis (EDA) on a telecom customer churn dataset to understand the factors influencing customer attrition.

The goal is to identify patterns and relationships between customer attributes and churn behavior before building machine learning models.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Dataset Features

The dataset contains customer information such as:

- Tenure Months
- Monthly Charges
- Total Charges
- Contract Type
- Payment Method
- Internet Service
- Phone Service
- Churn Label

---

## Exploratory Data Analysis

The following analyses were performed:

### 1. Tenure Distribution

Histogram and KDE plot to understand customer tenure patterns.

### 2. Churn vs Tenure

Boxplot comparing tenure distributions for churned and retained customers.

### 3. Churn vs Monthly Charges

Boxplot analyzing the relationship between monthly charges and churn.

### 4. Contract Type vs Churn

Countplot to identify how different contract types affect churn.

---

## Key Insights

- Customers with lower tenure are more likely to churn.
- Customers with higher monthly charges show higher churn rates.
- Long-term contract customers tend to have lower churn rates.
- Tenure is one of the strongest indicators of customer retention.

---

## Project Structure

```text
ChurnPredictionAndCustomerSegmentation/
│
├── data/
├── notebooks/
├── images/
├── README.md
└── requirements.txt
```

---

## Future Work

- Feature Engineering
- Data Preprocessing
- Churn Prediction using Machine Learning
- Customer Segmentation
- Model Evaluation and Deployment
