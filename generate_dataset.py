"""
Generate synthetic transaction dataset for fraud detection
"""
import numpy as np
import pandas as pd

np.random.seed(42)

# Generate 1000 transaction records
n_samples = 1000

# Generate features
data = {
    'transaction_id': [f'TXN{str(i).zfill(6)}' for i in range(1, n_samples + 1)],
    'amount': np.random.exponential(scale=100, size=n_samples),
    'time_of_day': np.random.randint(0, 24, size=n_samples),
    'day_of_week': np.random.randint(0, 7, size=n_samples),
    'merchant_category': np.random.choice(['retail', 'restaurant', 'online', 'gas_station', 'grocery'], size=n_samples),
    'transaction_type': np.random.choice(['credit', 'debit', 'cash'], size=n_samples),
    'location_distance': np.random.gamma(2, 50, size=n_samples),
    'previous_transactions_24h': np.random.poisson(3, size=n_samples),
    'account_age_days': np.random.randint(30, 3650, size=n_samples),
    'balance': np.random.normal(5000, 2000, size=n_samples)
}

# Add some outliers for fraud detection
fraud_indices = np.random.choice(n_samples, size=50, replace=False)
for idx in fraud_indices:
    data['amount'][idx] *= np.random.uniform(5, 15)
    data['location_distance'][idx] *= np.random.uniform(3, 8)
    data['previous_transactions_24h'][idx] = np.random.randint(10, 30)

# Add some null values
null_indices = np.random.choice(n_samples, size=20, replace=False)
for idx in null_indices[:10]:
    data['balance'][idx] = np.nan
for idx in null_indices[10:]:
    data['location_distance'][idx] = np.nan

# Add some duplicates
duplicate_indices = np.random.choice(n_samples, size=5, replace=False)
df = pd.DataFrame(data)

# Create duplicates
duplicate_rows = df.iloc[duplicate_indices].copy()
df = pd.concat([df, duplicate_rows], ignore_index=True)

# Save dataset
df.to_csv('transaction_data.csv', index=False)
print(f"Dataset generated: {len(df)} records")
print(f"Columns: {list(df.columns)}")
print(f"Null values: {df.isnull().sum().sum()}")
print(f"Duplicates: {df.duplicated().sum()}")
