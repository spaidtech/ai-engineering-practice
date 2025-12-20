# 📘 MODEL SELECTION, VALIDATION & BIAS–VARIANCE (MICRO NOTES)

This document is **lock-in knowledge** for Applied ML Engineers.  
If you understand and remember this — you are safe in **projects + interviews**.

---

## 🧠 PART A — MODEL SELECTION

### 🔹 What is Model Selection?
Choosing **which model family** to use for a problem.

You are **NOT**:
- tuning hyperparameters
- checking final accuracy

You are deciding:   
> **“Which type of model can solve this problem?”**

---

### 🔹 Why Model Selection is Needed
Different models behave differently:
- Simple vs complex  
- Linear vs non-linear  
- Low vs high variance  

❌ Wrong model → wasted time + poor results

---

### 🔹 Naive but Correct Rule
Start simple  
↓  
Increase complexity **only if needed**

---

### 🔹 Practical Model Order (MEMORIZE)

| Problem Type | Start With | Upgrade If Needed |
|--------------|-----------|-------------------|
| Regression | Linear Regression | Tree → RF → Boosting |
| Classification | Logistic Regression | Tree → RF → Boosting |
| Non-linear data | Decision Tree | Random Forest |
| Very large data | Linear / Tree | Neural Network |

---

### 🔹 What to Check During Selection?
✔ **Training performance only**

| Training Error | Meaning |
|---------------|--------|
| High | Model too simple |
| Reasonable | Good candidate |
| Very low | Might overfit (check later) |

---

### 🔹 Model Selection Summary
- Selection decides **WHAT model**
- Not **how good** yet

---

## 🧪 PART B — MODEL VALIDATION

### 🔹 What is Validation?
Testing model performance on **unseen data**

📌 Training performance lies  
📌 Validation performance tells truth

---

### 🔹 Why Validation is Needed
Models can:
- memorize training data  
- fail on new data  

Validation detects this failure.

---

### 🔹 Correct Data Usage (VERY IMPORTANT)

| Split | Purpose |
|-----|--------|
| Train | Learn |
| Validation | Check |
| Test | Final exam (ONCE) |

❌ Never tune using test data

---

### 🔹 Simplest Validation — Hold-out
Split:
- Train: 70%
- Validation: 15%
- Test: 15%

| Train | Val | Meaning |
|------|-----|--------|
| High | Low | Overfitting |
| Low | Low | Underfitting |
| High | High | Good model |

---

### 🔹 Better Validation — Cross-Validation
- Split data into K folds  
- Train K times  
- Average validation score  

✔ Stable  
✔ Reliable  
✔ Industry standard

---

### 🔹 What Validation Tells You

| Observation | Conclusion |
|-----------|-----------|
| Train ≫ Val | Overfitting |
| Train ≈ Val (low) | Model too simple |
| Train ≈ Val (high) | Best model |

---

### 🔹 Validation Summary
Validation decides:
> **“Can I trust this model?”**

---

## 🧩 BIG PICTURE (ONE LINE)
**Select → Validate → Tune → Test → Deploy**

---

## 🧠 ONE-PAGE MEMORY RULE
- Selection → Which model?  
- Validation → Does it generalize?  
- Test → Final proof only  

---

## 🔁 CROSS-VALIDATION (CODE + MEANING)

### 1️⃣ Basic K-Fold (Most Common)

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

scores = cross_val_score(
    model, X, y,
    cv=5,
    scoring="accuracy"
)

print(scores.mean())
```

🧠 Meaning:
- 5 splits  
- 5 trainings  
- Final score = average  

---

### 2️⃣ Regression CV

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    LinearRegression(),
    X, y,
    cv=5,
    scoring="neg_root_mean_squared_error"
)

rmse = -scores.mean()
```

📌 sklearn returns **negative RMSE**

---

### 3️⃣ Manual KFold
Use when:
- reproducibility matters  
- shuffle control needed  

```python
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
```

---

### 4️⃣ Imbalanced Data — StratifiedKFold
📌 Keeps class ratio same in each fold

```python
from sklearn.model_selection import StratifiedKFold
```

---

### 5️⃣ Time-Series CV
📌 Future data never predicts past

```python
from sklearn.model_selection import TimeSeriesSplit
```

---

### 6️⃣ CV Selection Table (MEMORIZE)

| Situation | Use |
|----------|------|
| Normal data | KFold |
| Imbalanced | StratifiedKFold |
| Time-series | TimeSeriesSplit |
| Grouped users | GroupKFold |

---

### 🔒 Sacred Rule
- CV → selection & tuning  
- Test set → final evaluation only  

---

# ⚙️ HYPERPARAMETER TUNING

### 🔹 What Are Hyperparameters?
Settings chosen before training that control:
- model complexity  
- learning behavior  

Examples:
- tree depth  
- learning rate  
- regularization strength  
- k in KNN  

---

### 🔹 Why Tuning Exists
Same model + same data → different performance depending on hyperparameters

Goal:
- reduce bias  
- reduce variance  
- improve generalization  

📌 Tuning does NOT fix bad data

---

### 🔹 Bias–Variance Connection

| Observation | Diagnosis | Action |
|-----------|---------|--------|
| High train & val error | High Bias | Increase complexity |
| Low train, high val | High Variance | Increase regularization |
| Train ≈ Val & low | Good model | Stop |

---

### 🔹 Important Hyperparameters (Only What Matters)

#### Decision Tree
- max_depth  
- min_samples_leaf  

Overfit → ↓ depth, ↑ leaf  
Underfit → ↑ depth

#### Random Forest
- n_estimators  
- max_depth  
- min_samples_leaf  

More trees → ↓ variance

#### Gradient Boosting
- learning_rate  
- n_estimators  
- max_depth  

📌 Low LR + more trees = best

#### Linear / Logistic
- alpha / lambda / C  

↑ regularization → ↓ variance, ↑ bias

#### KNN
- small k → overfit  
- large k → underfit  

---

### 🔹 Industry Workflow
1. Train baseline  
2. Compare train vs val  
3. Diagnose bias/variance  
4. Tune 1–2 params  
5. Re-evaluate  
6. Stop when improvement plateaus  

---

### 🔹 Search Methods

| Method | Notes |
|-------|------|
| Manual | Best for learning |
| Grid Search | Inefficient |
| Random Search | Default choice |
| Bayesian / AutoML | Use later |

📌 Random > Grid in practice

---

### 🔹 Common Mistakes
❌ Tuning before preprocessing  
❌ Tuning too many parameters  
❌ Using test set  
❌ Blind AutoML  

---

### 🔹 Interview One-Liners
- Hyperparameter tuning balances bias–variance using validation data  
- I tune after diagnosing train vs validation error  
- Random search is more efficient than grid search  

---

# ⚖️ BIAS–VARIANCE TRADE-OFF

### 🔹 What Bias Means
- Model too simple  
- Wrong assumptions  
- Underfitting  

---

### 🔹 What Variance Means
- Model too complex  
- Learns noise  
- Overfitting  

---

### 🔹 Metric Diagnosis (MOST IMPORTANT)

| Train | Test | Diagnosis |
|------|------|-----------|
| High | High | Bias |
| Low | High | Variance |
| Low | Low | Good |

---

### 🔹 Learning Curve Interpretation

#### High Bias
- train & val high  
- curves close  
- more data ❌  
- better model ✅  

#### High Variance
- train low, val high  
- big gap  
- more data ✅  
- regularization ✅  

---

### 🔹 Fixing Strategy
**Fix Bias**
- add features  
- increase model complexity  
- reduce regularization  

**Fix Variance**
- simplify model  
- add regularization  
- collect more data  

---

### 🔹 Golden Engineer Workflow
Train  
→ Compare train vs test  
→ Diagnose bias/variance  
→ Plot learning curve  
→ Fix  
→ Re-evaluate  

---

## 🔑 FINAL ONE-LINE MODEL
Bias = model too simple  
Variance = model too sensitive  
**Goal = lowest test error, not lowest train error**
