"""
Example usage of the trained ML pipeline for fraud detection
"""
import pandas as pd
import numpy as np
import joblib

# Load trained models
print("Loading trained models...")
kmeans_model = joblib.load('kmeans_model.pkl')
dt_model = joblib.load('decision_tree_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

print("✓ Models loaded successfully\n")

# Create a new transaction for prediction
print("="*80)
print("FRAUD DETECTION EXAMPLE")
print("="*80)

# Example transaction data (already preprocessed)
new_transactions = pd.DataFrame({
    'amount': [150.50, 2500.00, 45.20],
    'time_of_day': [14, 3, 10],
    'day_of_week': [2, 5, 1],
    'merchant_category': [2, 1, 3],  # encoded
    'transaction_type': [1, 0, 2],  # encoded
    'location_distance': [50.5, 850.0, 25.3],
    'previous_transactions_24h': [3, 15, 2],
    'account_age_days': [1200, 45, 2000],
    'balance': [5000.0, 8000.0, 3500.0],
    'amount_category': [1, 0, 1]  # encoded
})

print("\nNew Transactions:")
print(new_transactions)

# Scale the features
print("\nScaling features...")
X_new = scaler.transform(new_transactions)

# Predict clusters (pattern identification)
print("\nIdentifying transaction patterns...")
clusters = kmeans_model.predict(X_new)
print(f"Cluster assignments: {clusters}")

# Predict with Decision Tree (classification)
print("\nClassifying transactions...")
predictions = dt_model.predict(X_new)
print(f"Pattern predictions: {predictions}")

# Analyze results
print("\n" + "="*80)
print("ANALYSIS RESULTS")
print("="*80)

for idx in range(len(new_transactions)):
    print(f"\nTransaction {idx + 1}:")
    print(f"  Amount: ${new_transactions.iloc[idx]['amount']:.2f}")
    print(f"  Time: {int(new_transactions.iloc[idx]['time_of_day'])}:00")
    print(f"  Location distance: {new_transactions.iloc[idx]['location_distance']:.1f} km")
    print(f"  Previous 24h txns: {int(new_transactions.iloc[idx]['previous_transactions_24h'])}")
    print(f"  Cluster: {clusters[idx]}")
    print(f"  Predicted Pattern: {predictions[idx]}")
    
    # Risk assessment based on cluster and amount
    if clusters[idx] == 1 and new_transactions.iloc[idx]['amount'] > 500:
        risk = "HIGH RISK"
    elif new_transactions.iloc[idx]['amount'] > 1000:
        risk = "MEDIUM RISK"
    else:
        risk = "LOW RISK"
    print(f"  Risk Level: {risk}")

print("\n" + "="*80)
print("Fraud detection analysis completed!")
print("="*80)
