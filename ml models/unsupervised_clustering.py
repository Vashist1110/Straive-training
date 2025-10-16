from sklearn.cluster import KMeans
import pandas as pd

# Sample customer data (Age, Annual Income in $1000s)
data = {
    "Age": [25, 34, 28, 52, 46, 56, 23, 40, 60, 48],
    "Income": [40, 60, 50, 80, 70, 90, 30, 65, 100, 85]
}
df = pd.DataFrame(data)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=42)
df["Cluster"] = kmeans.fit_predict(df[["Age", "Income"]])

print(df)