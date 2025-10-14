"""
Preprocessing script for transaction data.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess(data_path, output_path):
    df = pd.read_csv(data_path)
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    preprocess("../data/transactions.csv", "../data/transactions_preprocessed.csv")
