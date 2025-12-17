# Day 1 --- Applied AI / ML Engineering

## Supervised Learning (Theory + Hands-on)

------------------------------------------------------------------------

## 1. ML Engineer Mindset

> Machine Learning is not about models.\
> It is about converting raw data into reliable decisions.

**Key ideas** - Models are tools - Data, features, and decisions matter
more than algorithms - A good ML engineer reasons before coding

------------------------------------------------------------------------

## 2. End-to-End ML Pipeline

    Problem Framing
    → Data Collection
    → EDA
    → Data Cleaning
    → Feature Engineering
    → Model Selection
    → Training
    → Evaluation
    → Threshold Tuning
    → Deployment
    → Monitoring

Most real-world failures happen before modeling.

------------------------------------------------------------------------

## 3. Problem Framing (Titanic)

-   Task: Binary classification
-   Target: Survived
-   Risk focus: False Negatives are worse (missing survivors)
-   Metric priority: Recall

------------------------------------------------------------------------

## 4. Baseline Principle

> If a baseline model performs badly,\
> the problem is usually data or features, not the model.

Baseline used: - Logistic Regression

Why: - Stable - Interpretable - Exposes data issues

------------------------------------------------------------------------

## 5. Models Used (Conceptual)

### Logistic Regression

-   Type: Linear classifier
-   Strengths: Fast, interpretable, strong baseline
-   Limitation: Linear decision boundary

### Random Forest

-   Type: Non-linear ensemble
-   Strengths: Handles interactions well, strong for tabular data
-   Limitation: Less interpretable

Rule:

    If Random Forest > Logistic Regression
    → Data has non-linear patterns

------------------------------------------------------------------------

## 6. Evaluation Metrics

### Confusion Matrix

    [[TN FP]
     [FN TP]]

### Accuracy

-   Overall correctness
-   Misleading for imbalanced data

### Precision

-   Trust in positive predictions
-   Important when FP is costly

### Recall

-   Ability to catch positives
-   Important when FN is costly

### F1-score

-   Balance between precision and recall

For Titanic:

    Recall > Accuracy

------------------------------------------------------------------------

## 7. Data Loading & Inspection

``` python
import pandas as pd

df = pd.read_csv("train.csv")
df.info()
df.describe()
df.isnull().sum()
df["Survived"].value_counts()
```

------------------------------------------------------------------------

## 8. Feature Engineering (Domain Driven)

``` python
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
df["IsChild"] = (df["Age"] <= 12).astype(int)
```

------------------------------------------------------------------------

## 9. Feature--Target Split

``` python
X = df.drop("Survived", axis=1)
y = df["Survived"]

numeric_cols = [
    "Pclass", "Age", "SibSp", "Parch", "Fare",
    "FamilySize", "IsAlone", "IsChild"
]

cat_cols = ["Sex", "Embarked"]
```

------------------------------------------------------------------------

## 10. Train--Test Split

``` python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)
```

------------------------------------------------------------------------

## 11. Preprocessing Pipeline

``` python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)
```

------------------------------------------------------------------------

## 12. Baseline Model --- Logistic Regression

``` python
from sklearn.linear_model import LogisticRegression

log_clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

log_clf.fit(X_train, y_train)
```

### Evaluation

``` python
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

y_pred = log_clf.predict(X_test)

accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred)
recall_score(y_test, y_pred)
f1_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)
```

------------------------------------------------------------------------

## 13. Random Forest Model

``` python
from sklearn.ensemble import RandomForestClassifier

rf_clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

rf_clf.fit(X_train, y_train)
```

------------------------------------------------------------------------

## 14. Threshold Tuning

``` python
y_proba = rf_clf.predict_proba(X_test)[:, 1]

threshold = 0.43
y_pred_thresh = (y_proba >= threshold).astype(int)

accuracy_score(y_test, y_pred_thresh)
precision_score(y_test, y_pred_thresh)
recall_score(y_test, y_pred_thresh)
f1_score(y_test, y_pred_thresh)
confusion_matrix(y_test, y_pred_thresh)
```

------------------------------------------------------------------------

## 15. Feature Engineering + Random Forest

``` python
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)

confusion_matrix(y_test, y_pred_rf)
```

------------------------------------------------------------------------

## 16. Decision Framework

    Baseline
    → Metrics
    → Error type (FP vs FN)
    → Action:
       - Threshold tuning
       - Feature engineering
       - Model upgrade

------------------------------------------------------------------------

## 17. Key Learnings (Day 1)

-   Baselines diagnose data health
-   Accuracy alone is dangerous
-   Recall matters when FN is costly
-   Threshold tuning can outperform model changes
-   Feature engineering works best with non-linear models
-   Pipeline order is critical

------------------------------------------------------------------------

## 18. Interview-Ready Summary

I started with a Logistic Regression baseline to validate data quality,
then used Random Forest to capture non-linear patterns. I improved
recall using threshold tuning and domain-driven feature engineering
instead of blindly increasing model complexity.
