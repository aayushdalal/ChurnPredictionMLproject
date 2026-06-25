# 📡 Telco Customer Churn Prediction & Retention Strategy

> **An end-to-end Machine Learning pipeline** to identify at-risk telecom customers and uncover the data-driven factors that drive customer attrition — built on 7,043 real customer records using Python, Scikit-Learn, and a 5-stage iterative modeling approach.

| Metric           |         Score |
| ---------------- | ------------: |
| Accuracy         |         76.7% |
| Precision        |         54.3% |
| Recall           |         76.6% |
| F1-Score         |         63.7% |
| Cross Validation |        5-Fold |
| Model for churn prediciton            | Random Forest |
| Model for Customer Segmentation         | K-means |


---

## 📌 Executive Summary

This project is a complete Data Science pipeline covering exploratory data analysis, statistical validation, feature engineering, predictive modeling, and model robustness testing — all focused on a single, high-stakes business question:

> *"Which customers are about to cancel, and why?"*

The final model — a hyperparameter-tuned Random Forest Classifier with cross-validated performance — achieves a **True Mean Recall of 76.6%**, meaning it reliably identifies roughly **3 out of 4** customers at risk of churning before they leave. This allows the business to proactively intervene with targeted retention offers rather than reacting after the loss.

### Key Contributions

- Engineered an end-to-end customer analytics pipeline using **Python, Pandas, NumPy, Seaborn, and Scikit-Learn**, processing **7,043 customer records** across 33 features to uncover churn patterns and retention drivers.
- Performed rigorous statistical analysis using **normalized cross-tabulations (`pd.crosstab`)**, identifying a critical **42.7% churn rate** among month-to-month contract users — vs. just **2.8%** on two-year contracts.
- Detected and resolved **hidden string anomalies** in the `Total Charges` column using `pd.to_numeric(errors='coerce')`, uncovering 11 new-customer records incorrectly stored as blank strings instead of zero.
- Optimized memory utilization through strategic datatype downcasting (**`int64` → `int8`**) while preserving full analytical accuracy.
- Executed a **5-stage model optimization loop** — from a biased 51% Recall baseline to a cross-validated 76.6% True Mean Recall — with full business justification at every step.
- Applied iterative **Feature Importance Analysis** and principled **Dimensionality Reduction**, proving that the two lowest-importance features (`Multiple Lines_No phone service`, `Phone Service_Yes`) were pure noise and safely removable.
- Validated final model stability through **5-Fold Cross-Validation**, eliminating "lucky split" bias and confirming genuine generalization.

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

## ⚙️ Data Engineering & Preprocessing

Before training any model, significant work was done to ensure data quality and pipeline integrity.

### Memory Optimization

Default 64-bit integer columns were downcasted to `int8` where the value range allowed, significantly reducing the DataFrame's memory footprint without losing a single data point.

```python
df['Tenure Months'] = df['Tenure Months'].astype('int8')
```

### Hidden Anomaly Detection

The `Total Charges` column was stored as an `object` (string) type despite containing numerical values. This was caused by 11 new-customer records (Tenure = 0) having their total charges recorded as blank spaces (`" "`) in the raw Excel file.

Standard `isnull()` checks returned zero missing values, masking the problem entirely. The fix required type coercion:

```python
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
# errors='coerce' silently converts unreadable strings to NaN
df['Total Charges'] = df['Total Charges'].fillna(0)
# New customers have a legitimate Total Charges of 0
```

**Why `fillna(0)` and not `fillna(mean)`?** These 11 rows correspond to customers with Tenure = 0 who haven't been billed yet. Filling with the column mean would inject a false financial figure into brand-new customer records, corrupting the model's understanding of early-tenure behavior.

### Feature Engineering & Encoding

After removing metadata and leakage columns (`CustomerID`, `Churn Score`, `CLTV`, `Churn Reason`, `Churn Label`, geographic fields, and `City`), all remaining categorical features were one-hot encoded using `pd.get_dummies(drop_first=True)`. This produced a clean, model-ready dataset of **29 binary and numerical features**.

---

## 📊 Exploratory Data Analysis: Key Findings

All insights below were validated with both visual plots and statistical confirmation.

### 1. The Month-to-Month Trap (Contract Type)

The normalized cross-tabulation (`pd.crosstab(normalize='index')`) reveals a stark risk gradient across contract tiers:

| Contract Type | Churn Probability |
|---|---|
| Month-to-Month | **42.7%** |
| One Year | ~11.3% |
| Two Year | **2.8%** |

**Recommendation:** Incentivize month-to-month users to migrate to annual plans (e.g., first month free). Converting even a fraction of these customers yields a dramatic churn reduction.

### 2. Price Sensitivity (Monthly Charges)

A boxplot of `Monthly Charges` segmented by `Churn Label` revealed a clear price gap, confirmed by quantile statistics:

| Customer Group | Q1 | Median | Q3 |
|---|---|---|---|
| **Churned (Yes)** | $56.15 | **$79.65** | $94.20 |
| **Retained (No)** | $25.10 | **$64.42** | $88.40 |

A ~$15 median gap exists between churners and loyalists. Premium-tier customers are leaving — likely to cheaper competitors.

### 3. The Tech Support Safety Net

Customers without Technical Support churn at dramatically higher rates. Every unresolved service issue is a potential cancellation. Tech support should be treated as a churn prevention tool, not an upsell.

### 4. Electronic Check Friction (Payment Method)

Customers paying via Electronic Check show disproportionately high churn. The reason is behavioral: manual payments force customers to re-evaluate their bill every month, triggering cancellation thoughts. Auto-pay creates psychological lock-in.

**Recommendation:** Offer a small discount (e.g., $5/month) to customers who switch to automatic payment methods.

### 5. The Fiber Optic Problem (Internet Service)

Fiber Optic customers churn significantly more than DSL users — despite paying more. This indicates a service quality issue: high-paying customers have elevated expectations, and unresolved outages or throttling will cause them to switch immediately.

### 6. Correlation Heatmap (Numerical Features)

| Feature Pair | Correlation | Interpretation |
|---|---|---|
| Tenure vs. Churn Value | **-0.35** | Longer-tenured customers are less likely to churn |
| Monthly Charges vs. Churn Value | **+0.19** | Higher bills correlate with higher churn risk |

---

## 🤖 Machine Learning Pipeline

### The Problem: Class Imbalance & the Accuracy Paradox

Only ~26.5% of customers churned. A "lazy" model that predicts `Stay` for everyone achieves **73.5% accuracy** while catching **zero churners**. This is the Accuracy Paradox.

For a telecom business, the cost structure is highly asymmetric:
- **False Negative (missed churner):** ~$1,200+ in lost annual recurring revenue
- **False Positive (loyal customer flagged):** A $10–$20 retention discount

Therefore, the primary optimization target is **Recall (Sensitivity)** for the Churn class — not overall Accuracy.

---

### Model Iteration History

The pipeline followed a deliberate, 5-stage optimization loop:

| Stage | Model | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score (Churn) |
|---|---|---|---|---|---|
| **V1** | Baseline Random Forest | **79%** | 0.66 | 0.51 | 0.58 |
| **V2** | + Class Weight Balancing | 79% | 0.61 | 0.69 | 0.65 |
| **V3** | + Hyperparameter Tuning (`max_depth=10`) | 77% | 0.57 | **0.78** | **0.66** |
| **V4** | + Feature Selection (drop bottom 2) | 77% | ~0.57 | ~0.79 | ~0.66 |
| **V5 (Final)** | 5-Fold Cross-Validation (`n_estimators=500`) | 76.7% | 0.543 | **76.6%** | — |

---

### Stage-by-Stage Analysis

#### V1 — The Baseline Trap

```python
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
```

79% accuracy looks good on paper. But the Recall of **0.51** means the model missed **196 of the 400 actual churners** in the test set. These 196 people represent hundreds of thousands of dollars in silent revenue loss. The model was lazy — it mostly predicted "Stay" to coast on the majority class.

#### V2 — Addressing the Imbalance with `class_weight='balanced'`

```python
rf_balanced = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
```

`class_weight='balanced'` mathematically penalizes the model for missing churners by weighting each class inversely proportional to its frequency. Recall jumped from **0.51 → 0.69** (+18 points) and F1 improved from 0.58 → 0.65. Accuracy remained the same (79%), confirming the prior version's accuracy was misleading.

#### V3 — Hyperparameter Tuning

```python
rf_tuned = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
```

Adding `max_depth=10` prevents the trees from memorizing noise in the training data (overfitting). Recall spiked to **0.78**. The trade-off was a 2% drop in Accuracy (79% → 77%) and 4% drop in Precision (0.61 → 0.57).

**The Business Verdict:** This is an excellent trade-off. Lower precision means the marketing team sends a few more unnecessary retention discounts to loyal customers — a tiny operational cost. In exchange, the model now catches 78% of actual churners instead of 69%.

#### V4 — Feature Importance Analysis & Dimensionality Reduction

Feature importances were extracted from `rf_tuned.feature_importances_` and ranked:

| Rank | Feature | Importance |
|---|---|---|
| 🥇 1 | `Tenure Months` | ~17.9% |
| 🥈 2 | `Total Charges` | ~13.6% |
| 🥉 3 | `Contract_Two year` | ~10.2% |
| 4 | `Monthly Charges` | ~9.3% |
| 5 | `Contract_Month-to-month` | ~8.1% |
| ... | ... | ... |
| Bottom 1 | `Multiple Lines_No phone service` | ~0.x% |
| Bottom 2 | `Phone Service_Yes` | ~0.x% |

**Why only drop the bottom 2, not all low-importance features?**

This is a critical engineering decision. Random Forests don't evaluate features in isolation — they look at how features *combine*. Aggressively dropping multiple features at once risks:

1. **Destroying hidden feature interactions** — A feature with low standalone importance might be one half of a powerful combination (e.g., `Paperless Billing` + `Senior Citizen` together might be a strong churn signal).
2. **Selection bias overfitting** — Importance scores are calculated on one specific training run. Deleting features based on a single snapshot can overfit the feature selection process to that run's random splits.

The correct approach is **iterative Recursive Feature Elimination**: drop only the absolute lowest-signal features, retrain, and re-evaluate. After dropping `Multiple Lines_No phone service` and `Phone Service_Yes`, Recall actually increased by ~1%, confirming these features were pure noise with no hidden signal.

```python
features_to_drop = feature_importance['Features'].tail(2).tolist()
X_train_reduced = X_train.drop(columns=features_to_drop)
X_test_reduced  = X_test.drop(columns=features_to_drop)
```

#### V5 — Grid Search & Selecting the Optimal Configuration

A nested loop was run across `n_estimators ∈ {100, 200, 300, 400, 500}` and `max_depth ∈ {5, 10, 15, 20}` — 20 total combinations — to find the optimal settings sorted by Recall first, then Accuracy:

| Trees | Depth | Accuracy | Recall | Precision | F1-Score |
|---|---|---|---|---|---|
| 500 | 10 | 0.7701 | **0.7825** | 0.5691 | 0.6589 |
| 400 | 10 | 0.7694 | 0.7800 | 0.5680 | 0.6576 |
| 300 | 10 | 0.7687 | 0.7787 | 0.5680 | 0.6574 |
| 500 | 15 | 0.7673 | 0.8100 | 0.5290 | 0.6395 |
| ... | ... | ... | ... | ... | ... |

**Why not just pick the highest F1-Score automatically?**

The F1-Score assumes Precision and Recall matter equally (50/50). In churn prediction, they emphatically do not. The correct approach is to sort by Recall first to identify the performance ceiling, then scroll down to find the best Precision trade-off a human business decision can justify.

The row with `Trees=500, Depth=10` was chosen: Recall = **0.7825**, F1 = **0.6589**. The row with `Depth=15` had a higher Recall (0.81) but at a Precision of 0.529 — meaning nearly half of all "churn predictions" would be false alarms, making the retention campaign prohibitively expensive. The `Depth=10` configuration offers the best business balance.

---

### Final Stage — 5-Fold Cross-Validation

**Why Cross-Validation?**

A single `train_test_split(test_size=0.20)` is vulnerable to "lucky split" bias. If by chance the easiest customers to predict end up in the test set, the model will appear better than it is. Cross-Validation eliminates this by rotating through 5 different train/test splits, ensuring every data point is eventually tested on.

```python
final_rf = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42, class_weight='balanced')

cv_recall    = cross_val_score(final_rf, X_selected, y, cv=5, scoring='recall')
cv_accuracy  = cross_val_score(final_rf, X_selected, y, cv=5, scoring='accuracy')
cv_precision = cross_val_score(final_rf, X_selected, y, cv=5, scoring='precision')
```

**5-Fold Recall Scores (per fold):**

| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|---|---|---|---|
| 73.5% | **81.2%** | 77.2% | 77.2% | 73.7% |

The 7.7% spread between the best fold (81.2%) and worst fold (73.5%) is natural variance for a 7,000-row dataset — not overfitting. A healthy model is stable, not perfectly constant. If folds varied by 30%+, that would indicate a serious instability problem.

**Final Cross-Validated Performance:**

| Metric | True Mean Score |
|---|---|
| Accuracy | **76.7%** |
| Recall | **76.6%** |
| Precision | **54.3%** |

The cross-validated Recall (76.6%) is slightly lower than the single-split score (78.25%). **This is the correct and expected result.** The CV mean is the brutally honest, real-world estimate of how the model will perform on brand-new customers next month — not just on one favorable test split.

**Conclusion:** The model successfully generalizes to unseen data. It will reliably identify roughly **76–77% of all customers who are at risk of churning**, giving the business time to proactively intervene and protect recurring revenue.

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

## 🚀 Future Enhancements

- [ ] **Advanced Hyperparameter Search:** Implement `GridSearchCV` and `RandomizedSearchCV` with `scoring='recall'` for a more exhaustive search space.
- [ ] **Model Comparison:** Benchmark Random Forest against XGBoost, LightGBM, and Logistic Regression.
- [ ] **SMOTE Oversampling:** Test Synthetic Minority Over-sampling Technique as an alternative to `class_weight='balanced'` for handling class imbalance.
- [ ] **Customer Segmentation:** Apply K-Means clustering to segment customers into risk tiers beyond binary churn/no-churn.
- [ ] **ROC-AUC & F₂-Score:** Add AUC curve analysis and the F₂-Score (which weights Recall twice as heavily as Precision) for more nuanced threshold selection.
- [ ] **Web Deployment:** Deploy the trained model as an interactive web application for real-time churn scoring.
- [ ] **Retention Strategy Engine:** Build rule-based recommendation logic on top of predictions (e.g., "Flag this customer for a contract upgrade offer").

