# Practical 5: DBSCAN - Clustering of Students based on Study Patterns
# ---------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# 1️⃣ Create Dataset
data = {
    'Student': ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12','S13'],
    'Hours_Studied': [2,3,4,5,6,7,8,9,10,11,4,6,8],
    'Attendance': [60,65,70,75,80,85,90,95,98,99,85,60,70],
    'Sleep_Hours': [6,7,6,7,7,8,7,8,7,8,6,7,8]
}

df = pd.DataFrame(data)
print("Original Dataset:\n", df, "\n")

# 2️⃣ Select features for clustering
X = df[['Hours_Studied', 'Attendance', 'Sleep_Hours']]

# 3️⃣ Feature Scaling (very important for DBSCAN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4️⃣ Apply DBSCAN algorithm
dbscan = DBSCAN(eps=1.2, min_samples=2)   # try adjusting eps between 1.0–1.5 if clusters seem off
df['Cluster'] = dbscan.fit_predict(X_scaled)

# 5️⃣ Display Clusters
print("Cluster Assignments:\n", df[['Student', 'Hours_Studied', 'Attendance', 'Sleep_Hours', 'Cluster']], "\n")

# 6️⃣ Visualization
plt.figure(figsize=(7,5))
plt.scatter(df['Hours_Studied'], df['Attendance'], c=df['Cluster'], cmap='viridis', s=100)
plt.xlabel('Hours Studied')
plt.ylabel('Attendance')
plt.title('DBSCAN Clustering of Students')
plt.colorbar(label='Cluster Label')
plt.show()

# 7️⃣ Cluster Analysis
n_clusters = len(set(df['Cluster'])) - (1 if -1 in df['Cluster'].values else 0)
n_noise = list(df['Cluster']).count(-1)

print(f"Number of Clusters: {n_clusters}")
print(f"Number of Noise Points: {n_noise}\n")

print("""
Observations:
1. DBSCAN successfully grouped students based on similar study behavior.
2. Each color in the scatter plot represents a cluster of students with similar patterns.
3. Points marked as '-1' are outliers — students whose habits differ from all clusters.
4. eps (Epsilon) defines the neighborhood radius for clustering.
5. min_samples defines the minimum number of points required to form a dense region.
6. DBSCAN is powerful for detecting non-linear clusters without knowing cluster count.
""")
