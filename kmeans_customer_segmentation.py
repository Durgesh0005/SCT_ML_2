import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('customers.csv')

# Features: Income and Spending Score
X = df[['AnnualIncome', 'SpendingScore']]

# Standardize features (important for KMeans)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method to find optimal K
inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal K')
plt.grid(True)
plt.show()

# Apply KMeans with chosen K
k_optimal = 4
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Print clustered data
print(df[['CustomerID', 'AnnualIncome', 'SpendingScore', 'Cluster']])

# Visualize the clusters
plt.figure(figsize=(8, 6))
for i in range(k_optimal):
    cluster = df[df['Cluster'] == i]
    plt.scatter(cluster['AnnualIncome'], cluster['SpendingScore'], label=f'Cluster {i}')
    
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Segments (K-Means Clustering)')
plt.legend()
plt.grid(True)
plt.show()
