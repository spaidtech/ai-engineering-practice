# Day 1 -- Applied AI / ML Engineering (Supervised Learning + Hands-on)

This document contains **both concepts and executable code** used on
Day 1. Dataset: Titanic (Kaggle-style tabular dataset)

------------------------------------------------------------------------

## 1. ML Engineer Mindset

Machine Learning is not about models.\
It is about converting raw data into reliable decisions.

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

------------------------------------------------------------------------

## 3. Data Loading & Inspection

``` python
import pandas as pd

df = pd.read_csv("train.csv")
df.info()
df.describe()
df.isnull().sum()
df["Survived"].value_counts()
```

------------------------------------------------------------------------

## 4. Feature Engineering (Domain Driven)

``` python
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
df["IsChild"] = (df["Age"] <= 12).astype(int)
```

------------------------------------------------------------------------

## 5. Feature--Target Split

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

## 6. Train--Test Split

``` python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)
```

------------------------------------------------------------------------

## 7. Preprocessing Pipeline

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

## 8. Baseline Model -- Logistic Regression

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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

y_pred = log_clf.predict(X_test)

accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred)
recall_score(y_test, y_pred)
f1_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)
```

------------------------------------------------------------------------

## 9. Random Forest Model

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

## 10. Threshold Tuning

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

## 11. Feature Engineering + Random Forest

``` python
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)

confusion_matrix(y_test, y_pred_rf)
```

------------------------------------------------------------------------

## 12. Key Learnings

-   Baseline models diagnose data quality
-   Accuracy alone is misleading
-   Recall is critical when FN is costly
-   Threshold tuning can outperform model changes
-   Feature engineering helps non-linear models more than linear ones
-   Pipeline order is critical

------------------------------------------------------------------------

## 13. Interview-Ready Summary

I started with a Logistic Regression baseline to validate data quality,
then used Random Forest to capture non-linear patterns. I improved
recall using threshold tuning and domain-driven feature engineering
instead of blindly increasing model complexity.
