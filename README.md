# Telco Customer Churn Prediction & Retention Strategy

> **An end-to-end Machine Learning pipeline** to identify at-risk telecom customers and uncover the data-driven factors that drive customer attrition — built on 7,043 real customer records using Python, Scikit-Learn, and a 5-stage iterative modeling approach.

| Metric           |         Score |
| ---------------- | ------------: |
| Accuracy         |         76.7% |
| Precision        |         54.3% |
| Recall           |         76.6% |
| F1-Score         |         63.7% |
| Cross Validation |        5-Fold |
| Model for Churn Prediciton            | Random Forest |
| Model for Customer Segmentation         | K-means |


---

## 📌 Executive Summary

The final model — a hyperparameter-tuned Random Forest Classifier with cross-validated performance — achieves a **True Mean Recall of 76.6%**, meaning it reliably identifies roughly **3 out of 4** customers at risk of churning before they leave. This allows the business to proactively intervene with targeted retention offers rather than reacting after the loss.

Dataset:
IBM Telco Customer Churn Dataset (7043 customers)

## 🤖 Machine Learning Pipeline

### Model Iteration History

The pipeline followed a deliberate, 5-stage optimization loop:

| Stage | Model | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score (Churn) |
|---|---|---|---|---|---|
| **V1** | Baseline Random Forest | **79%** | 0.66 | 0.51 | 0.58 |
| **V2** | + Class Weight Balancing | 79% | 0.61 | 0.69 | 0.65 |
| **V3** | + Hyperparameter Tuning (`max_depth=10`) | 77% | 0.57 | **0.78** | **0.66** |
| **V4** | + Feature Selection (drop bottom 2) | 77% | ~0.57 | ~0.79 | ~0.66 |
| **V5 (Final)** | 5-Fold Cross-Validation (`n_estimators=500`) | 76.7% | 0.543 | **76.6%** | — |

**Conclusion:** The model successfully generalizes to unseen data. It will reliably identify roughly **76–77% of all customers who are at risk of churning**, giving the business time to proactively intervene and protect recurring revenue.

---
### Key Contributions

- Engineered an end-to-end customer analytics pipeline using **Python, Pandas, NumPy, Seaborn, and Scikit-Learn**, processing **7,043 customer records** across 33 features to uncover churn patterns and retention drivers.
- Performed rigorous statistical analysis using **normalized cross-tabulations (`pd.crosstab`)**, identifying a critical **42.7% churn rate** among month-to-month contract users — vs. just **2.8%** on two-year contracts.
- Detected and resolved **hidden string anomalies** in the `Total Charges` column using `pd.to_numeric(errors='coerce')`, uncovering 11 new-customer records incorrectly stored as blank strings instead of zero.
- Optimized memory utilization through strategic datatype downcasting (**`int64` → `int8`**) while preserving full analytical accuracy.
- Executed a **5-stage model optimization loop** — from a biased 51% Recall baseline to a cross-validated 76.6% True Mean Recall — with full business justification at every step.
- Applied iterative **Feature Importance Analysis** and principled **Dimensionality Reduction**, proving that the two lowest-importance features (`Multiple Lines_No phone service`, `Phone Service_Yes`) were pure noise and safely removable.
- Validated final model stability through **5-Fold Cross-Validation**, eliminating "lucky split" bias and confirming genuine generalization.

---
## 📈 Business Impact Summary

| Churn Driver | Finding | Recommendation |
|---|---|---|
| **Contract Type** | Month-to-month users churn at 42.7% | Incentivize upgrades to annual contracts |
| **Monthly Charges** | Churners pay ~$15/month more (median) | Loyalty discounts for premium tiers |
| **Tech Support** | No-support customers churn at much higher rates | Include basic tech support in base plans |
| **Payment Method** | Electronic check users churn far more | Offer discount for auto-pay enrollment |
| **Fiber Optic** | Fiber customers churn more than DSL users | Investigate service quality in high-churn areas |
| **Tenure** | New customers (low tenure) are highest risk | Launch targeted 0–6 month onboarding program |

---

## ⚙️ Data Engineering & Preprocessing

### Memory Optimization

Default 64-bit integer columns were downcasted to `int8` where the value range allowed, significantly reducing the DataFrame's memory footprint without losing a single data point.

### Hidden Anomaly Detection

The `Total Charges` column was stored as an `object` (string) type despite containing numerical values. This was caused by 11 new-customer records (Tenure = 0) having their total charges recorded as blank spaces (`" "`) in the raw Excel file.

### Feature Engineering & Encoding

After removing metadata and leakage columns (`CustomerID`, `Churn Score`, `CLTV`, `Churn Reason`, `Churn Label`, geographic fields, and `City`), all remaining categorical features were one-hot encoded using `pd.get_dummies(drop_first=True)`. This produced a clean, model-ready dataset of **29 binary and numerical features**.

---

## 📊 Exploratory Data Analysis: Key Findings

All insights below were validated with both visual plots and statistical confirmation.

### 1. The Month-to-Month Trap (Contract Type)
<img width="704" height="470" alt="image" src="https://github.com/user-attachments/assets/e6705d6d-598e-4568-b54e-e2935afde198" />


The normalized cross-tabulation (`pd.crosstab(normalize='index')`) reveals a stark risk gradient across contract tiers:

| Contract Type | Churn Probability |
|---|---|
| Month-to-Month | **42.7%** |
| One Year | ~11.3% |
| Two Year | **2.8%** |

**Recommendation:** Incentivize month-to-month users to migrate to annual plans (e.g., first month free). Converting even a fraction of these customers yields a dramatic churn reduction.

### 2. Price Sensitivity (Monthly Charges)

A boxplot of `Monthly Charges` segmented by `Churn Label` revealed a clear price gap, confirmed by quantile statistics:
<img width="695" height="470" alt="image" src="https://github.com/user-attachments/assets/2dbcfeee-9999-43a0-9b8e-23858ba771f3" />


| Customer Group | Q1 | Median | Q3 |
|---|---|---|---|
| **Churned (Yes)** | $56.15 | **$79.65** | $94.20 |
| **Retained (No)** | $25.10 | **$64.42** | $88.40 |

A ~$15 median gap exists between churners and loyalists. Premium-tier customers are leaving — likely to cheaper competitors.

### 3. The Tech Support Safety Net
<img width="704" height="470" alt="image" src="https://github.com/user-attachments/assets/95361bb2-ea3c-4f22-9fb8-41abb32a49bc" />

Customers without Technical Support churn at dramatically higher rates. Every unresolved service issue is a potential cancellation. Tech support should be treated as a churn prevention tool, not an upsell.

### 4. Electronic Check Friction (Payment Method)
<img width="708" height="470" alt="image" src="https://github.com/user-attachments/assets/439c8044-ee19-4344-9a9a-a53ef958f191" />

Customers paying via Electronic Check show disproportionately high churn. The reason is behavioral: manual payments force customers to re-evaluate their bill every month, triggering cancellation thoughts. Auto-pay creates psychological lock-in.

**Recommendation:** Offer a small discount (e.g., Rs 50/month) to customers who switch to automatic payment methods.

### 5. The Fiber Optic Problem (Internet Service)
<img width="704" height="470" alt="image" src="https://github.com/user-attachments/assets/436fab08-2afe-4f78-9069-6d335c3fc121" />

Fiber Optic customers churn significantly more than DSL users — despite paying more. This indicates a service quality issue: high-paying customers have elevated expectations, and unresolved outages or throttling will cause them to switch immediately.

### 6. Correlation Heatmap (Numerical Features)
<img width="871" height="528" alt="image" src="https://github.com/user-attachments/assets/85583def-0aa7-414f-a357-3abf5a87d13e" />

| Feature Pair | Correlation | Interpretation |
|---|---|---|
| Tenure vs. Churn Value | **-0.35** | Longer-tenured customers are less likely to churn |
| Monthly Charges vs. Churn Value | **+0.19** | Higher bills correlate with higher churn risk |


---

## 🛠️ Technology Stack

| Category | Libraries |
|---|---|
| **Language** | Python 3 |
| **Data Engineering** | Pandas, NumPy |
| **Machine Learning** | Scikit-Learn (RandomForestClassifier, cross_val_score) |
| **Visualization** | Matplotlib, Seaborn |
| **Environment** | Jupyter Notebooks (VS Code) |

---

## 📂 Project Structure

```text
ChurnPredictionAndCustomerSegmentation/
│
├── data/
│   ├── Telco_customer_churn.xlsx       ← Raw source data
│   └── processed_churn_data.csv        ← Cleaned & encoded data (output of EDA)
│
├── notebooks/
│   ├── ChurnEDA.ipynb                  ← Data cleaning, EDA, preprocessing, encoding
│   └── MLimplementation.ipynb         ← Model training, optimization, cross-validation
│
├── README.md
└── requirements.txt
```

---

## 🚀 Future Enhancements

- [ ] **Advanced Hyperparameter Search:** Implement `GridSearchCV` and `RandomizedSearchCV` with `scoring='recall'` for a more exhaustive search space.
- [ ] **Model Comparison:** Benchmark Random Forest against XGBoost, LightGBM, and Logistic Regression.
- [ ] **SMOTE Oversampling:** Test Synthetic Minority Over-sampling Technique as an alternative to `class_weight='balanced'` for handling class imbalance.
- [ ] **Customer Segmentation:** Apply K-Means clustering to segment customers into risk tiers beyond binary churn/no-churn.
- [ ] **ROC-AUC & F₂-Score:** Add AUC curve analysis and the F₂-Score (which weights Recall twice as heavily as Precision) for more nuanced threshold selection.
- [ ] **Web Deployment:** Deploy the trained model as an interactive web application for real-time churn scoring.
- [ ] **Retention Strategy Engine:** Build rule-based recommendation logic on top of predictions (e.g., "Flag this customer for a contract upgrade offer").

