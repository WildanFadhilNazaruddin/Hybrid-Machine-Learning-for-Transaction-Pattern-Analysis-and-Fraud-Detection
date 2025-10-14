"""
Decision Tree classification using cluster labels.
"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def classify(data_path):
    df = pd.read_csv(data_path)
    X = df.drop('cluster', axis=1)
    y = df['cluster']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    classify("../data/transactions_clustered.csv")
