# 🛍️ Mall Customer Segmentation using K-Means Clustering

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Clustering-orange?style=flat-square&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=flat-square&logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-informational?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

A complete unsupervised machine learning project applying **K-Means Clustering** to segment mall customers into distinct behavioral groups based on annual income and spending patterns — enabling data-driven marketing strategies.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Key Results](#-key-results)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Business Use Cases](#-business-use-cases)
- [Author](#-author)

---

## 📖 Overview

Customer segmentation is a critical pillar of retail marketing. This project leverages the **K-Means algorithm** — an unsupervised clustering technique — to automatically discover natural groupings within mall customer data. The workflow covers feature selection, data normalization, optimal cluster selection via the Elbow Method, clustering, and Silhouette Score evaluation.

**Silhouette Score achieved: `0.5547`** — indicating well-separated, meaningful clusters.

---

## 📊 Dataset

**Mall Customer Segmentation Dataset** — 200 customer records with purchasing behavior attributes.

| Feature | Type | Description |
|---|---|---|
| `CustomerID` | Integer | Unique customer identifier |
| `Gender` | Categorical | Male / Female |
| `Age` | Integer | Age of the customer |
| `Annual Income (k$)` | Float | Annual income in thousands of USD |
| `Spending Score (1-100)` | Integer | Mall-assigned score based on spending behavior |

- **Records:** 200
- **Features used for clustering:** `Annual Income (k$)`, `Spending Score (1-100)`
- **Source:** [Kaggle — Mall Customers Dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

---

## 🔬 Project Workflow

### Step 1 — Load the Dataset
Data is loaded from Google Drive using `pandas.read_csv()` and explored for structure and quality.

### Step 2 — Feature Selection
Two purchasing-behavior features are selected for clustering:
```python
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
```

### Step 3 — Data Normalization
Since K-Means is distance-based, features are standardized (mean = 0, variance = 1) using `StandardScaler` to prevent bias from differing scales.

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Step 4 — Elbow Method (Optimal K)
WCSS (Within-Cluster Sum of Squares) is plotted for K = 1 to 10. The "elbow" in the curve at **K = 5** identifies the optimal number of clusters.

```python
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
```

### Step 5 — Apply K-Means Clustering
K-Means is fitted with K = 5 and cluster labels are assigned to each customer.

```python
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
```

### Step 6 — Cluster Visualization
Scatter plot of all 5 clusters on the Annual Income vs Spending Score space with color-coded segments.

### Step 7 — Silhouette Score Evaluation
```python
score = silhouette_score(X_scaled, clusters)
# Output: 0.5547
```
A score of **0.55** confirms the clusters are reasonably compact and well-separated with minimal overlap.

---

## 📈 Key Results

Five distinct customer segments were identified:

| Cluster | Segment Label | Annual Income | Spending Score | Insight |
|---|---|---|---|---|
| 0 | 💰 High Earners, High Spenders | High | High | Prime targets — VIP loyalty programs |
| 1 | 🎯 High Earners, Low Spenders | High | Low | Untapped potential — needs engagement |
| 2 | 🛒 Average All-Rounders | Medium | Medium | Largest group — standard promotions |
| 3 | 💸 Low Earners, High Spenders | Low | High | Budget-conscious yet impulsive buyers |
| 4 | 🧩 Low Earners, Low Spenders | Low | Low | Minimal engagement — cost-efficient targeting |

**Silhouette Score: `0.5547`** — clusters are well-defined and meaningful.

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting Elbow Curve and Cluster Visualization |
| `scikit-learn` | `KMeans`, `StandardScaler`, `silhouette_score` |

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib scikit-learn
```

### Clone the Repository
```bash
git clone https://github.com/Taha-Saleem-43/Mall-Customer-Segmentation-using-K-Means-Clustering.git
cd Mall-Customer-Segmentation-using-K-Means-Clustering
```

### Run
Open the notebook PDF or recreate in Jupyter:
```bash
jupyter notebook
```

---

## 📁 Project Structure

```
Mall-Customer-Segmentation-using-K-Means-Clustering/
│
├── assignment-no-04-task-01.pdf    # Full notebook export with code, plots & analysis
└── README.md                       # Project documentation
```

---

## 💼 Business Use Cases

- **Targeted Marketing** — Design campaigns tailored to each segment's income-spending profile
- **Product Placement** — Position products near high-spending customer zones
- **Loyalty Programs** — Reward VIP customers (Cluster 0) with exclusive offers
- **Customer Retention** — Re-engage high-income low-spenders (Cluster 1) through personalized outreach

---

## 👤 Author

**Taha Saleem**
GitHub: [@Taha-Saleem-43](https://github.com/Taha-Saleem-43)

---

> ⭐ If you found this project helpful, please consider giving it a star!
