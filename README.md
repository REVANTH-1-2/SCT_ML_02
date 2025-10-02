# SCT_ML_02
# 🛍️ Mall Customer Segmentation using K-Means Clustering

This project applies **K-Means Clustering** to segment customers based on their **Annual Income** and **Spending Score** from the `Mall_Customers.csv` dataset. The goal is to identify customer groups for better business targeting.

## 📊 Project Overview

- **Algorithm Used**: K-Means Clustering
- **Dataset**: `Mall_Customers.csv`
- **Selected Features**:
  - Annual Income (k$)
  - Spending Score (1-100)

## 🔧 Tools & Libraries

- Python
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## 📈 Evaluation Metrics

- **Inertia (SSE)** – Measures the compactness of clusters
- **Silhouette Score** – Measures cluster separation quality
- **Davies-Bouldin Index** – Lower score indicates better clustering

## 📂 Output

- `mall_customers_clustered.csv` – Original dataset with an additional `Cluster` column for cluster labels

## 📉 Visualization

- A 2D scatter plot shows clusters of customers with different colors
- Cluster centroids are marked with black "X" symbols

## ✅ How it Works

1. Load and scale the selected features.
2. Apply K-Means with `k=5`.
3. Predict clusters and add them to the dataset.
4. Visualize the results and print evaluation metrics.

