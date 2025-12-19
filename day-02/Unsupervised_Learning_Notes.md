# 📘 UNSUPERVISED LEARNING — COMPLETE NOTES (Theory + Code + Industry)

## ⭐ Table of Contents
1️⃣ **Core Idea**  
2️⃣ **Industry Use Cases**  
3️⃣ **Supervised vs Unsupervised**  
4️⃣ **Evaluation Philosophy**  
5️⃣ **Preprocessing Theory + Pipeline**  
6️⃣ **Feature Engineering**  
7️⃣ **PCA**  
8️⃣ **Visualization**  
9️⃣ **Algorithm Selection**  
🔟 **K-Means**  
1️⃣1️⃣ **DBSCAN**  
1️⃣2️⃣ **DBSCAN Fail Cases**  
1️⃣3️⃣ **Metrics**  
1️⃣4️⃣ **Real Business Usage**  
1️⃣5️⃣ **Demographics Rule**  
1️⃣6️⃣ **Deployment**  
🔒 **Final Lock**  

---

## 1️⃣ What is Unsupervised Learning

Unsupervised learning discovers structure and patterns in data **without labels**.

- No target variable  
- No correct answer  
- No loss correction  

🎯 Goal → **Discover useful structure, not truth**

---

## 2️⃣ Industry Applications

- Customer segmentation  
- Fraud detection  
- User grouping  
- Recommendation systems  
- Feature understanding  
- Preprocessing for supervised learning  

---

## 3️⃣ Supervised vs Unsupervised

| Aspect | Supervised | Unsupervised |
|--------|----------|-------------|
| Labels | Yes | No |
| Objective | Accuracy | Useful structure |
| Evaluation | Metrics | Visualization |
| Learning | Loss | Geometry |
| Output | Prediction | Segments |

📌 Supervised = correctness  
📌 Unsupervised = usefulness  

---

## 4️⃣ Evaluation Philosophy

Metrics lie in unsupervised learning.

✔ Visualization  
✔ Business reasoning  
✔ Metrics last  

📌 If visualization doesn’t make sense → Reject the model  

---

# UNSUPERVISED PREPROCESSING — COMPLETE THEORY

Preprocessing decides clustering quality.

---

## Why It Matters

Most algorithms are distance-based. Distance only works if:

- Scale is correct  
- Noise handled  
- Features comparable  

📌 No label feedback → mistakes stay forever  

---

## Pipeline

```
Raw Data
↓
Drop IDs
↓
Numeric Only
↓
Handle Missing
↓
Fix Skew
↓
Handle Outliers
↓
Scale (MANDATORY)
↓
PCA (optional)
```

---

## Steps

### Drop IDs
```
df = df.drop(columns=["CustomerID"], errors="ignore")
```

### Numeric First
```
df = df.select_dtypes(include="number")
```

### Missing Handling
```
df = df.fillna(df.median())
```

### Fix Skew
```
import numpy as np
df = np.log1p(df)
```

### Outliers Carefully
```
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5*IQR)) | (df > (Q3 + 1.5*IQR))).any(axis=1)]
```

### Scaling (MANDATORY)
```
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
```

---

## PCA

```
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
```

📌 PCA reveals structure. Doesn’t create it.

---

## 🔥 Why Scaling is Done *Before* PCA (VERY IMPORTANT)

PCA works by analyzing **variance** in features.

- Features with larger numeric scale automatically have higher variance  
- PCA assumes all features are equally important  
- If you do NOT scale → PCA becomes biased toward high‑magnitude features

Example:
- income → 0 – 1,000,000  
- age → 18 – 70  

Without scaling:
👉 PCA thinks income is “more important” only because numbers are bigger  
👉 Result → Wrong principal components, wrong structure, wrong clusters

### ✔ Therefore:
- Always scale before PCA  
- StandardScaler is preferred

```
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
```

📌 Rule To Remember:
> PCA ≈ variance analysis  
> Variance depends on scale  
> Therefore → Scale → THEN PCA  

---

## Feature Engineering

Features = Model.

```
df["spend_to_income"] = df["spending"] / (df["income"] + 1)
```

---

## Visualization

```
plt.scatter(X_pca[:,0], X_pca[:,1])
```

Look for:

- Shape  
- Density  
- Noise  

---

## Algorithm Selection

### K-Means (distance)

Use when:
- Blob shaped clusters
- Similar density

### DBSCAN (density)

Use when:
- Irregular shapes
- Noise important
- Unknown K

📌 KMeans = Distance  
📌 DBSCAN = Density  

---

## K-Means

```
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
labels_km = kmeans.fit_predict(X_pca)
```

---

## DBSCAN

```
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.6, min_samples=5)
labels_db = dbscan.fit_predict(X_pca)
```

---

## Metrics (Support Only)

```
from sklearn.metrics import silhouette_score
silhouette_score(X_pca, labels_km)
```

---

## After Clustering (REAL VALUE)

```
df["cluster"] = labels_km
df.groupby("cluster").mean()
```

📌 Output = Business segments

---

## Deployment Pattern

- Train offline  
- Save scaler + PCA + model  
- Periodically tag users  

---

# FINAL LOCK

Unsupervised ≠ Accuracy  
Visualization > Metrics  
PCA helps reasoning  
KMeans = distance  
DBSCAN = density  
Real output = Segments  

---

### COMPLETE ✔
