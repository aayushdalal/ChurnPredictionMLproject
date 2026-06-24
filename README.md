# Telco Customer Churn Prediction & Retention Strategy

## 📌 Executive Summary

This project is an end-to-end Data Science and Machine Learning pipeline designed to predict customer attrition in the telecommunications sector.

Built using over **7,000 customer records**, the project combines data engineering, exploratory analytics, feature preprocessing, and supervised machine learning to identify customers at risk of churning and uncover the key business factors driving customer retention.

The pipeline includes extensive data cleaning, memory optimization, statistical analysis, feature engineering, and predictive modeling using **Random Forest Classifiers**. Through both business intelligence and machine learning approaches, the project aims to support proactive retention strategies and improve customer lifetime value.

### Key Contributions

* Engineered an end-to-end customer analytics pipeline using **Python, Pandas, NumPy, Seaborn, and Scikit-Learn**, processing **7,000+ customer records** to uncover churn patterns and retention drivers.

* Performed rigorous statistical analysis using **normalized cross-tabulations (`pd.crosstab`)**, identifying a critical **42.7% churn rate** among month-to-month contract users.

* Optimized memory utilization through datatype downcasting (**int64 → int8**) and efficient preprocessing transformations, reducing storage overhead while preserving analytical accuracy.

* Implemented robust data cleaning workflows to handle missing values, hidden string anomalies, and inconsistent data types, creating high-quality features for downstream machine learning models.

* Trained and evaluated a **Random Forest Classifier** to predict customer churn, while actively improving **Accuracy, Precision, Recall, and F1-Score** through feature engineering and model optimization.

---

## 🛠️ Technology Stack & Environment

* **Core Language:** Python 3
* **Data Engineering & Analysis:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Random Forest)
* **Data Visualization:** Matplotlib, Seaborn
* **Environment:** Jupyter Notebooks (VS Code)

---

## ⚙️ Data Engineering Highlights

Before training predictive models, extensive preprocessing and optimization techniques were applied to improve data quality and pipeline scalability:

* **Memory Optimization:** Downcasted default 64-bit numerical columns (e.g., `int64` → `int8`) to significantly reduce memory consumption while maintaining analytical fidelity.

* **Hidden Anomaly Handling:** Utilized `pd.to_numeric(errors='coerce')` to detect and convert hidden string anomalies into proper missing values for reliable downstream processing.

* **Data Cleaning Pipeline:** Handled missing values, inconsistent data formats, categorical preprocessing requirements, and feature quality validation to ensure model-ready datasets.

---

## 📊 Key Business Insights (Statistical Findings)

Through visual analytics, probability analysis, and normalized cross-tabulations, several critical churn drivers were identified:

### 1. The Month-to-Month Trap (Contract Type)

* Month-to-month customers exhibit a **42.7% churn probability**.
* Customers on two-year contracts show a dramatically lower churn probability of only **2.8%**.

### 2. Price Sensitivity (Monthly Charges)

* The median monthly bill for churned customers is **$79.65**, compared to **$64.42** for retained customers.
* Premium pricing emerges as a significant contributor to customer attrition.

### 3. The Tech Support Safety Net

* Customers without technical support churn at substantially higher rates than those receiving support services.
* Lack of issue resolution appears to be a major retention risk factor.

### 4. Billing Friction (Payment Methods)

* Manual payment methods (Electronic Checks) demonstrate significantly higher churn rates than automated payment methods.
* Auto-pay systems appear to create stronger customer retention and behavioral lock-in effects.

---

## 🤖 Machine Learning Pipeline

### Data Preparation

* Missing value handling
* Data type correction
* Feature engineering
* Memory optimization
* Categorical feature encoding

### Model Development

* Random Forest Classifier
* Train/Test Split Validation
* Performance Evaluation using:

  * Accuracy
  * Precision
  * Recall
  * F1-Score
  * Confusion Matrix

### Current Focus

* Hyperparameter tuning
* Feature selection
* Improving Recall and F1-Score for churn detection
* Reducing False Negatives to better identify high-risk customers

---

## 📂 Project Structure

```text
ChurnPredictionAndCustomerSegmentation/
│
├── data/
├── notebooks/
│   ├── ChurnEDA.ipynb
│   └── ChurnModel.ipynb
├── README.md
└── requirements.txt
```

---

## 🚀 Future Enhancements

* Hyperparameter optimization using GridSearchCV and RandomizedSearchCV.
* Compare Random Forest with XGBoost, LightGBM, and Logistic Regression.
* Customer segmentation using K-Means clustering.
* Deploy the trained model as an interactive web application.
* Build retention recommendation strategies based on churn risk predictions.
