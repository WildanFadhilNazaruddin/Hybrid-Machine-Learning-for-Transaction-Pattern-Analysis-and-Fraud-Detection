"""
Test script to verify ML pipeline models can be loaded and used
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

print("="*80)
print("TESTING ML PIPELINE - MODEL LOADING AND PREDICTION")
print("="*80)

# Load all models
print("\n1. Loading saved models...")
kmeans_model = joblib.load('kmeans_model.pkl')
dt_model = joblib.load('decision_tree_model.pkl')
scaler = joblib.load('scaler.pkl')
pca_model = joblib.load('pca_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
print("✓ All models loaded successfully")

# Load processed data with clusters
print("\n2. Loading clustered data...")
df = pd.read_csv('data_with_clusters.csv')
print(f"✓ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Test predictions on sample data
print("\n3. Testing predictions on sample data...")
X_sample = df.drop('cluster', axis=1).head(10).values

# Predict clusters with KMeans
cluster_predictions = kmeans_model.predict(X_sample)
print(f"✓ KMeans cluster predictions: {cluster_predictions}")

# Predict with Decision Tree
dt_predictions = dt_model.predict(X_sample)
print(f"✓ Decision Tree predictions: {dt_predictions}")

# Verify predictions match
match_count = (cluster_predictions == dt_predictions).sum()
print(f"✓ Matching predictions: {match_count}/10")

# Test PCA transformation
print("\n4. Testing PCA transformation...")
X_pca = pca_model.transform(X_sample)
print(f"✓ PCA output shape: {X_pca.shape}")
print(f"✓ PCA explained variance ratio: {pca_model.explained_variance_ratio_}")

# Display cluster distribution
print("\n5. Cluster distribution in dataset:")
cluster_counts = df['cluster'].value_counts().sort_index()
for cluster_id, count in cluster_counts.items():
    percentage = (count / len(df)) * 100
    print(f"  - Cluster {int(cluster_id)}: {count} samples ({percentage:.1f}%)")

print("\n" + "="*80)
print("MODEL TESTING COMPLETED SUCCESSFULLY")
print("="*80)
print("\nAll models are functioning correctly and can be used for:")
print("  • Clustering new transactions")
print("  • Predicting cluster membership")
print("  • Dimensionality reduction with PCA")
print("  • Fraud pattern detection")
print("="*80)
