"""
Exploratory Data Analysis (EDA) script for transaction data.
"""
import pandas as pd
import matplotlib.pyplot as plt

def run_eda(data_path):
    df = pd.read_csv(data_path)
    print(df.info())
    print(df.describe())
    print(df.head())
    df.hist(figsize=(12,8))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_eda("../data/transactions.csv")
