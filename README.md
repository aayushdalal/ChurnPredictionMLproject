# Telco Customer Churn Prediction & Retention Strategy

## 📌 Executive Summary
This project is an end-to-end Data Science and Machine Learning pipeline designed to predict customer attrition in the telecommunications sector. 

Currently in the **Exploratory Data Analysis (EDA) and Preprocessing phase**, the project focuses on extracting actionable business intelligence from over 7,000 customer records. By statistically validating churn drivers—such as contract lock-in periods, pricing sensitivity, and technical support access—this analysis lays the groundwork for a predictive machine learning model.

---

## 🛠️ Technology Stack & Environment
* **Core Language:** Python 3
* **Data Engineering & Analysis:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Environment:** Jupyter Notebooks (VS Code)

---

## ⚙️ Data Engineering Highlights
Before analyzing the data, rigorous preprocessing and memory optimization techniques were applied to ensure pipeline scalability:
* **Memory Optimization:** Downcasted default 64-bit numerical columns (e.g., `int64` to `int8`) to drastically reduce the dataset's RAM footprint.
* **Hidden Anomaly Handling:** Utilized `pd.to_numeric` with `errors='coerce'` to identify and safely convert hidden string anomalies (blank spaces representing zero-tenure users) into standard `NaN` missing values for proper mathematical imputation.

---

## 📊 Key Business Insights (Statistical Findings)
Through visual and cross-tabulated probability analysis, several critical drivers of churn were mathematically confirmed:

1. **The Month-to-Month Trap (Contract Type):**
   * Month-to-month customers have an alarming **42.7% probability** of churning.
   * Securing a customer on a two-year contract drops their churn probability to a negligible **2.8%**.
2. **Price Sensitivity (Monthly Charges):**
   * The median monthly bill for a churned customer is **$79.65**, compared to just **$64.42** for a retained customer, proving that high premium costs are actively driving users to competitors.
3. **The Tech Support Safety Net:**
   * Customers lacking technical support churn at vastly disproportionate rates compared to those with support, indicating that network frustration without immediate resolution is a primary catalyst for cancellation.
4. **Billing Friction (Payment Methods):**
   * Manual payment methods (Electronic checks) see significantly higher churn than automated methods (Credit Card/Bank Transfer), highlighting the psychological "lock-in" benefit of auto-pay systems.

---

## 📂 Project Structure

```text
ChurnPredictionAndCustomerSegmentation/
│
├── data/                  # Raw and processed datasets (ignored in git)
├── notebooks/             # Jupyter notebooks for EDA and Model Training
│   └── ChurnEDA.ipynb     # Comprehensive Exploratory Data Analysis
├── images/                # Exported visual plots and correlation heatmaps
├── README.md              # Project documentation
└── requirements.txt       # Python environment dependencies

```

---

## 🚀 Next Steps (Future Work)

* **Categorical Encoding:** Translate text features into machine-readable formats using Label Encoding and One-Hot Encoding (`pd.get_dummies`).
* **Machine Learning Implementation:** Train predictive classification models (Logistic Regression, Random Forest) to identify high-risk customers in real-time.
* **Model Evaluation:** Optimize for Precision, Recall, and F1-Score to account for the imbalanced nature of the churn dataset.
* **Customer Segmentation:** Apply unsupervised learning (K-Means Clustering) to group users based on behavior and lifetime value (CLTV).
