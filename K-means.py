import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
df = pd.read_csv('Mall_Customers.csv')
features = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
k = 5
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(features_scaled)
centroids = kmeans.cluster_centers_
df.to_csv('mall_customers_clustered.csv', index=False)
inertia = kmeans.inertia_
silhouette_avg = silhouette_score(features_scaled, df['Cluster'])
db_index = davies_bouldin_score(features_scaled, df['Cluster'])
print(f"Inertia (SSE): {inertia:.2f}")
print(f"Silhouette Score: {silhouette_avg:.2f}")
print(f"Davies-Bouldin Index: {db_index:.2f}")
print("\nCluster-wise Income & Spending:")
print(df.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean())
plt.figure(figsize=(8, 6))
plt.scatter(features_scaled[:, 0], features_scaled[:, 1], c=df['Cluster'], cmap='Set1', edgecolors='k')
plt.scatter(centroids[:, 0], centroids[:, 1], s=250, c='black', marker='X', label='Centroids')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.title('K-Means Clustering: Mall Customers')
plt.legend()
plt.show()