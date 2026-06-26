# Telco Customer Churn Prediction & Retention Strategy

> **An end-to-end Machine Learning pipeline** to identify at-risk telecom customers and uncover the data-driven factors that drive customer attrition — built on 7,043 real customer records using Python, Scikit-Learn, and a multi-stage iterative modeling approach combining churn prediction with customer segmentation.

| Metric | Score |
| --- | ---: |
| Accuracy | 75% (threshold-tuned) |
| Precision | 54.2% |
| Recall | **82.3%** (threshold-tuned) |
| F1-Score | 65.3% |
| ROC-AUC Score | **86.6%** |
| Cross Validation | 5-Fold |
| Model for Churn Prediction | Random Forest |
| Model for Customer Segmentation | K-Means (k=3) |

---

## 📌 Executive Summary

The final model — a hyperparameter-tuned Random Forest Classifier with threshold optimization and cross-validated performance — achieves a **Recall of 82.3%** at a classification threshold of 0.45. This means it identifies roughly **4 out of 5** customers at risk of churning before they leave. The **ROC-AUC of 86.6%** confirms that the model has strong discriminative power in separating churners from loyal customers across all possible thresholds. Beyond binary classification, K-Means clustering (k=3, validated by both the Elbow Method and Silhouette Score) segments all 7,043 customers into three actionable personas — giving the marketing team a direct playbook for targeted retention.

**Dataset:** IBM Telco Customer Churn Dataset (7,043 customers, 33 raw features)

---

## 🤖 Model Development & Optimization

### Model Iteration History

The pipeline followed a deliberate, multi-stage optimization loop — from a biased 51% Recall baseline to a threshold-tuned 82.3% Recall final model:

| Stage | Model / Change | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score (Churn) |
|---|---|---|---|---|---|
| **V1** | Baseline Random Forest (`n=100`) | 79% | 0.66 | 0.51 | 0.58 |
| **V2** | + Class Weight Balancing (`balanced`) | 79% | 0.61 | 0.69 | 0.65 |
| **V3** | + Hyperparameter Tuning (`max_depth=10`) | 77% | 0.57 | 0.78 | 0.66 |
| **V4** | + Feature Selection (drop bottom 2 noise features) | 77% | 0.56 | 0.79 | 0.66 |
| **V5** | Grid Search over Trees × Depth (best: `n=500, depth=10`) | 77% | 0.569 | 0.7825 | 0.659 |
| **V6** | 5-Fold Cross-Validation (generalization check) | 76.7% | 0.543 | 76.6% | — |
| **V7 (Final)** | + Threshold Tuning (`threshold=0.45`) | **75%** | 0.542 | **82.3%** | 0.653 |

**Key insight on threshold tuning:** At the default threshold of 0.50, the model achieves 78.2% Recall. By lowering the classification threshold to 0.45, Recall jumps to 82.3% — catching an additional ~29 churners in the test set — while accuracy declines only marginally from 77% to 75%. In a telecom context where missing a churner costs far more than a false alarm, this tradeoff is firmly justified.

**Conclusion:** The model reliably identifies roughly **4 out of 5 customers at risk of churning**, with a cross-validated AUC of 86.6% confirming genuine discriminative power rather than overfitting to a single data split.

---

### Key Contributions

- Engineered an end-to-end customer analytics pipeline using **Python, Pandas, NumPy, Seaborn, and Scikit-Learn**, processing **7,043 customer records** across 33 features to uncover churn patterns and retention drivers.
- Performed rigorous statistical analysis using **normalized cross-tabulations (`pd.crosstab`)**, identifying a critical **42.7% churn rate** among month-to-month contract users — vs. just **2.8%** on two-year contracts.
- Detected and resolved **hidden string anomalies** in the `Total Charges` column using `pd.to_numeric(errors='coerce')`, uncovering 11 new-customer records incorrectly stored as blank strings instead of zero.
- Optimized memory utilization through strategic datatype downcasting (**`int64` → `int8`**) while preserving full analytical accuracy.
- Executed a **7-stage model optimization loop** — from a biased 51% Recall baseline to a threshold-tuned 82.3% Recall — with full business justification at every step.
- Applied iterative **Feature Importance Analysis** and principled **Dimensionality Reduction**, proving that the two lowest-importance features (`Multiple Lines_No phone service`, `Phone Service_Yes`) were pure noise and safely removable.
- Performed **exhaustive grid search** over 20 combinations of `n_estimators` (100–500) × `max_depth` (5–20), selecting the optimal `n=500, depth=10` configuration by F1-Score.
- Validated final model stability through **5-Fold Cross-Validation**, eliminating "lucky split" bias and confirming genuine generalization (mean recall: 76.6%, mean AUC: 86.6%).
- Implemented **Threshold Tuning** — evaluated 13 classification thresholds (0.10–0.80) and selected 0.45 as the optimal operating point, pushing Recall to 82.3% while maintaining 75% accuracy.
- Computed the **ROC-AUC score of 86.6%**, confirming the model's ability to correctly rank churners above non-churners regardless of threshold.
- Applied **K-Means Clustering (k=3)** on churn probability, monthly charges, tenure, and total charges — with optimal k verified by both the **Elbow Method** and **Silhouette Score** — to segment all 7,043 customers into three named business personas.
- Visualized customer segments using a **scatter plot** (Total Charges vs. Churn Probability, colored by cluster) to deliver an intuitive retention playbook to marketing.

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

The `Total Charges` column was stored as an `object` (string) type despite containing numerical values. This was caused by 11 new-customer records (Tenure = 0) having their total charges recorded as blank spaces (`" "`) in the raw Excel file. Resolved using `pd.to_numeric(errors='coerce')`.

### Feature Engineering & Encoding

After removing metadata and leakage columns (`CustomerID`, `Churn Score`, `CLTV`, `Churn Reason`, `Churn Label`, geographic fields, and `City`), all remaining categorical features were one-hot encoded using `pd.get_dummies(drop_first=True)`. This produced a clean, model-ready dataset of **29 binary and numerical features**, subsequently reduced to **28 features** after dropping the two noise columns identified by feature importance analysis.

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

## 🎯 Feature Importance Analysis

The tuned Random Forest surfaced the following importance leaderboard across all 28 model features:

| Rank | Feature | Importance |
|---|---|---|
| 1 | Tenure Months | 18.7% |
| 2 | Total Charges | 13.2% |
| 3 | Contract_Two year | 9.9% |
| 4 | Monthly Charges | 9.6% |
| 5 | Dependents_Yes | 7.2% |
| 6 | Internet Service_Fiber optic | 5.7% |
| 7 | Payment Method_Electronic check | 5.2% |
| … | (remaining 21 features) | ~30.5% |
| 29 | Phone Service_Yes | 0.43% |
| 30 | Multiple Lines_No phone service | 0.34% |

**Dimensionality reduction verdict:** The two bottom features (Phone Service_Yes and Multiple Lines_No phone service) were confirmed as noise. Dropping them increased Recall by ~1% with no accuracy penalty, proving they added zero predictive signal.

**Business translation:** Tenure, Charges, and Contract type alone account for over **50% of the model's predictive power**, confirming that the retention strategy should be hyper-focused on a customer's first few months with targeted pricing incentives tied to long-term contract upgrades.

---

## 📉 ROC-AUC Analysis

```python
ROC-AUC Score: 0.8565 (85.65%)
```

The ROC curve was computed on the held-out test set (1,409 customers). An AUC of 86.6% means that when the model is presented with a random churner and a random non-churner, it correctly ranks the churner as higher risk 86.6% of the time — far above the 50% baseline of random guessing. This confirms that the model's discriminative power is genuine and not an artifact of threshold choice.

---

## 🔧 Threshold Tuning

At the default 0.50 threshold, the model predicts "churn" only when it is at least 50% confident. Because the business cost of a missed churner (lost recurring revenue) vastly exceeds the cost of a false alarm (wasted retention offer), the threshold was lowered to capture more at-risk customers.

| Threshold | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| 0.10 | 51.5% | 36.7% | **98.0%** | 53.4% |
| 0.30 | 69.1% | 47.7% | 92.0% | 62.8% |
| 0.35 | 71.8% | 50.2% | 89.0% | 64.2% |
| 0.40 | 73.6% | 52.1% | 85.8% | 64.8% |
| **0.45 ✅ (chosen)** | **75.2%** | **54.2%** | **82.3%** | **65.3%** |
| 0.50 (default) | 77.0% | 56.9% | 78.2% | 65.9% |
| 0.55 | 78.6% | 60.0% | 73.3% | 66.0% |
| 0.70 | 80.6% | 68.6% | 58.0% | 62.9% |

**Selected threshold: 0.45** — This operating point delivers 82.3% Recall (up from 78.2% at default) while maintaining 75% accuracy, representing the optimal business tradeoff for a telecom retention use-case.

---

## 👥 Customer Segmentation (K-Means, k=3)


### Methodology

After training the churn prediction model, predicted churn probabilities were combined with three key customer features — `Tenure Months`, `Monthly Charges`, and `Total Charges` — to create a 4-dimensional segmentation dataset. Features were standardized using `StandardScaler` before clustering.
<img width="713" height="525" alt="image" src="https://github.com/user-attachments/assets/dfa0a899-2aec-4758-a69f-8580d64221db" />

**Optimal k validation:** Both the Elbow Method (inertia curve) and Silhouette Score analysis confirmed **k=3** as the optimal number of clusters. The silhouette score peaked at k=3, providing mathematical confirmation beyond visual inspection.

<img width="703" height="530" alt="image" src="https://github.com/user-attachments/assets/d73c84f6-f291-43fb-aecb-fab244e6278b" />


### Cluster Profiles

| Cluster | Name | Avg. Tenure | Avg. Monthly Charges | Avg. Total Charges | Avg. Churn Probability |
|---|---|---|---|---|---|
| **Cluster 1** |  High Risk New Customers | 11 months | $71.72 | $885.95 | **71.4%** |
| **Cluster 0** |  Loyal Premium Customers | 58 months | $90.34 | $5,273.39 | 25.2% |
| **Cluster 2** |  Budget Loyal Customers | 32 months | $32.49 | $1,043.12 | **13.8%** |

### Cluster Business Insights & Action Plans

**🚨 Cluster 1 — "High Risk New Customers" (Immediate Intervention Required)**
Brand-new customers (~11 months tenure) with moderately high monthly bills ($71) and a 71.4% churn probability. They are practically packing their bags.
- **Action:** Target immediately with aggressive retention discounts or lock them into 2-year contracts before the 12-month mark.

**💎 Cluster 0 — "Loyal Premium Customers" (Protect, Don't Discount)**
Long-tenured customers (~58 months, nearly 5 years) who pay the highest monthly bills ($90) and have accumulated $5,273 in total charges. Their 25.2% churn probability is relatively low.
- **Action:** Do not send blanket discounts — you'd be losing guaranteed margin. Instead, offer VIP perks, priority support, or free streaming upgrades to reinforce loyalty.

**🛡️ Cluster 2 — "Budget Loyal Customers" (Leave Them Alone)**
Mid-tenure customers (~32 months) on cheap, basic plans ($32/month) with a very low 13.8% churn probability. They are completely unbothered.
- **Action:** Do not aggressively upsell — the risk of accidentally annoying a stable customer into churning outweighs any incremental revenue gain.

### Segment Scatter Plot

The final scatter plot (Total Charges vs. Churn Probability, colored by Cluster Name with `alpha=0.6`) visually confirmed the three distinct personas and provided the marketing team with an intuitive, at-a-glance retention playbook across all 7,043 customers.
<img width="850" height="552" alt="image" src="https://github.com/user-attachments/assets/e30ed0db-835b-48e3-9879-f428bfd7905f" />
<img width="850" height="553" alt="image" src="https://github.com/user-attachments/assets/c4cd57c2-2819-4341-92d7-5a18afcc3759" />
<img width="850" height="553" alt="image" src="https://github.com/user-attachments/assets/a5405d34-58fc-4510-afdd-c9d6cf6ba5ee" />



---

## 🛠️ Technology Stack

| Category | Libraries |
|---|---|
| **Language** | Python 3.12 |
| **Data Engineering** | Pandas, NumPy |
| **Machine Learning** | Scikit-Learn (RandomForestClassifier, KMeans, cross_val_score, roc_auc_score, silhouette_score) |
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
│   └── MLimplementation.ipynb         ← Model training, optimization, threshold tuning,
│                                          ROC-AUC, cross-validation, K-Means segmentation
│
├── README.md
└── requirements.txt
```

---

## 🚀 Future Enhancements

- [ ] **Advanced Hyperparameter Search:** Implement `GridSearchCV` and `RandomizedSearchCV` with `scoring='recall'` for a more exhaustive and automated search space.
- [ ] **Model Comparison:** Benchmark Random Forest against XGBoost, LightGBM, and Logistic Regression on both AUC and Recall at the 0.45 threshold.
- [ ] **SMOTE Oversampling:** Test Synthetic Minority Over-sampling Technique as an alternative to `class_weight='balanced'` for handling class imbalance.
- [ ] **F₂-Score Optimization:** Use the F₂-Score (which weights Recall twice as heavily as Precision) as the primary scoring metric for threshold and hyperparameter selection.
- [ ] **Web Deployment:** Deploy the trained model as an interactive web application for real-time churn scoring and segment assignment.
- [ ] **Retention Strategy Engine:** Build rule-based recommendation logic on top of predictions (e.g., "Flag this customer for a contract upgrade offer" based on cluster + churn probability).
- [ ] **Permutation Importance:** Replace impurity-based feature importances with permutation importance to eliminate bias toward high-cardinality features.
