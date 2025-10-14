"""
KMeans clustering for pseudo-label generation.
"""
import pandas as pd
from sklearn.cluster import KMeans

def cluster(data_path, output_path, n_clusters=2):
    df = pd.read_csv(data_path)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(df)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    cluster("../data/transactions_preprocessed.csv", "../data/transactions_clustered.csv")
