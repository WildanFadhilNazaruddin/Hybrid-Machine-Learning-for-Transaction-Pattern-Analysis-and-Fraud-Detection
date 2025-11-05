"""
Hybrid ML Pipeline for Transaction Pattern Analysis and Fraud Detection
Combines unsupervised clustering (KMeans) with supervised classification (DecisionTree)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, silhouette_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Create output directories
import os
os.makedirs('models', exist_ok=True)
os.makedirs('output', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)


def generate_sample_dataset(n_samples=1000):
    """Generate sample transaction dataset for demonstration"""
    print("=" * 80)
    print("GENERATING SAMPLE TRANSACTION DATASET")
    print("=" * 80)
    
    data = {
        'transaction_id': range(1, n_samples + 1),
        'amount': np.random.exponential(scale=100, size=n_samples),
        'transaction_count': np.random.poisson(lam=5, size=n_samples),
        'account_age_days': np.random.randint(1, 3650, size=n_samples),
        'transaction_hour': np.random.randint(0, 24, size=n_samples),
        'merchant_category': np.random.choice(['retail', 'online', 'restaurant', 'gas_station'], size=n_samples),
        'device_type': np.random.choice(['mobile', 'desktop', 'tablet'], size=n_samples),
        'location_match': np.random.choice([0, 1], size=n_samples, p=[0.1, 0.9]),
    }
    
    # Add some correlation and patterns
    data['avg_transaction_amount'] = data['amount'] / (data['transaction_count'] + 1) + np.random.normal(0, 10, n_samples)
    data['risk_score'] = (data['amount'] / 100) * (1 - data['location_match']) + np.random.normal(0, 0.5, n_samples)
    
    df = pd.DataFrame(data)
    df.to_csv('data/transactions.csv', index=False)
    print(f"Sample dataset generated with {n_samples} transactions")
    print(f"Saved to: data/transactions.csv\n")
    return df


def perform_eda(df):
    """Perform Exploratory Data Analysis"""
    print("=" * 80)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 80)
    
    # Display first few rows
    print("\n1. HEAD - First 5 rows:")
    print(df.head())
    
    # Display dataset information
    print("\n2. INFO - Dataset information:")
    print(df.info())
    
    # Display statistical description
    print("\n3. DESCRIBE - Statistical summary:")
    print(df.describe())
    
    # Select only numeric columns for correlation
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Correlation matrix
    print("\n4. CORRELATION - Correlation matrix:")
    correlation_matrix = df[numeric_columns].corr()
    print(correlation_matrix)
    
    # Visualize correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Matrix of Numeric Features')
    plt.tight_layout()
    plt.savefig('visualizations/correlation_matrix.png', dpi=300)
    print("Correlation heatmap saved to: visualizations/correlation_matrix.png")
    plt.close()
    
    # Histograms
    print("\n5. HISTOGRAMS - Distribution of numeric features:")
    df[numeric_columns].hist(figsize=(15, 10), bins=30, edgecolor='black')
    plt.suptitle('Distribution of Numeric Features')
    plt.tight_layout()
    plt.savefig('visualizations/histograms.png', dpi=300)
    print("Histograms saved to: visualizations/histograms.png")
    plt.close()
    
    print("\nEDA completed!\n")
    return df


def preprocess_data(df):
    """Preprocess the dataset"""
    print("=" * 80)
    print("DATA PREPROCESSING")
    print("=" * 80)
    
    # 1. Handle null values
    print("\n1. Handling null values:")
    null_counts = df.isnull().sum()
    print(f"Null values per column:\n{null_counts}")
    if null_counts.sum() > 0:
        df = df.fillna(df.median(numeric_only=True))
        print("Null values filled with median for numeric columns")
    else:
        print("No null values found")
    
    # 2. Handle duplicates
    print("\n2. Handling duplicates:")
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows found: {duplicates}")
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"Removed {duplicates} duplicate rows")
    else:
        print("No duplicates found")
    
    # 3. Drop ID column
    print("\n3. Dropping ID column:")
    if 'transaction_id' in df.columns:
        df_processed = df.drop('transaction_id', axis=1)
        print("Dropped 'transaction_id' column")
    else:
        df_processed = df.copy()
        print("No ID column found")
    
    # 4. Encoding categorical variables
    print("\n4. Encoding categorical variables:")
    categorical_columns = df_processed.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}
    
    for col in categorical_columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
        print(f"Encoded '{col}' with {len(le.classes_)} classes")
    
    # Save label encoders
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    print("Label encoders saved to: models/label_encoders.pkl")
    
    # 5. Handle outliers using IQR method
    print("\n5. Handling outliers (IQR method):")
    numeric_columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    outliers_removed = 0
    
    for col in numeric_columns:
        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        before = len(df_processed)
        df_processed = df_processed[(df_processed[col] >= lower_bound) & (df_processed[col] <= upper_bound)]
        removed = before - len(df_processed)
        outliers_removed += removed
        
        if removed > 0:
            print(f"  {col}: removed {removed} outliers")
    
    print(f"Total outliers removed: {outliers_removed}")
    print(f"Remaining samples: {len(df_processed)}")
    
    # 6. Binning (discretization) - Example with amount
    print("\n6. Binning features:")
    if 'amount' in df_processed.columns:
        df_processed['amount_bin'] = pd.cut(df_processed['amount'], 
                                            bins=5, 
                                            labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        df_processed['amount_bin'] = LabelEncoder().fit_transform(df_processed['amount_bin'])
        print("Created 'amount_bin' feature with 5 bins")
    
    # 7. Scaling
    print("\n7. Scaling features:")
    scaler = StandardScaler()
    feature_names = df_processed.columns.tolist()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_processed),
        columns=feature_names
    )
    
    # Save scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    print("StandardScaler fitted and saved to: models/scaler.pkl")
    print(f"Scaled {len(feature_names)} features")
    
    print("\nPreprocessing completed!\n")
    return df_scaled, df_processed, scaler


def perform_clustering(df_scaled, n_clusters_range=(2, 11)):
    """Perform KMeans clustering with Elbow method and Silhouette analysis"""
    print("=" * 80)
    print("K-MEANS CLUSTERING")
    print("=" * 80)
    
    # Elbow Method
    print("\n1. Elbow Method - Finding optimal number of clusters:")
    inertias = []
    silhouette_scores = []
    K_range = range(n_clusters_range[0], n_clusters_range[1])
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(df_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(df_scaled, kmeans.labels_))
        print(f"  k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette Score={silhouette_scores[-1]:.4f}")
    
    # Plot Elbow curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(K_range, inertias, 'bo-')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')
    ax1.grid(True)
    
    # Plot Silhouette scores
    ax2.plot(K_range, silhouette_scores, 'ro-')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('visualizations/elbow_silhouette.png', dpi=300)
    print("\nElbow and Silhouette plots saved to: visualizations/elbow_silhouette.png")
    plt.close()
    
    # Select optimal k (using silhouette score)
    optimal_k = K_range[np.argmax(silhouette_scores)]
    print(f"\n2. Optimal number of clusters: {optimal_k} (based on Silhouette score)")
    
    # Fit final KMeans model
    print("\n3. Fitting final KMeans model:")
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans_final.fit_predict(df_scaled)
    
    final_silhouette = silhouette_score(df_scaled, cluster_labels)
    print(f"Final model - Clusters: {optimal_k}, Silhouette Score: {final_silhouette:.4f}")
    
    # Save KMeans model
    joblib.dump(kmeans_final, 'models/kmeans_model.pkl')
    print("KMeans model saved to: models/kmeans_model.pkl")
    
    print("\nClustering completed!\n")
    return kmeans_final, cluster_labels


def perform_pca_visualization(df_scaled, cluster_labels):
    """Perform PCA for dimensionality reduction and visualization"""
    print("=" * 80)
    print("PCA - DIMENSIONALITY REDUCTION")
    print("=" * 80)
    
    # Fit PCA
    print("\n1. Fitting PCA with 2 components for visualization:")
    pca = PCA(n_components=2, random_state=42)
    principal_components = pca.fit_transform(df_scaled)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.4f}")
    
    # Create PCA DataFrame
    pca_df = pd.DataFrame(
        data=principal_components,
        columns=['PC1', 'PC2']
    )
    pca_df['Cluster'] = cluster_labels
    
    # Visualize clusters in PCA space
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], 
                         c=pca_df['Cluster'], 
                         cmap='viridis', 
                         alpha=0.6, 
                         edgecolors='black', 
                         linewidth=0.5)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('KMeans Clusters Visualization using PCA')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/pca_clusters.png', dpi=300)
    print("PCA cluster visualization saved to: visualizations/pca_clusters.png")
    plt.close()
    
    # Save PCA model
    joblib.dump(pca, 'models/pca_model.pkl')
    print("PCA model saved to: models/pca_model.pkl")
    
    print("\nPCA completed!\n")
    return pca, pca_df


def interpret_clusters(df_original, cluster_labels, scaler):
    """Interpret cluster characteristics by inverse transforming and analyzing"""
    print("=" * 80)
    print("CLUSTER INTERPRETATION")
    print("=" * 80)
    
    # Inverse transform the scaled data
    print("\n1. Inverse transforming scaled data:")
    df_unscaled = pd.DataFrame(
        scaler.inverse_transform(df_original),
        columns=df_original.columns
    )
    df_unscaled['Cluster'] = cluster_labels
    
    # Analyze each cluster
    print("\n2. Cluster statistics:")
    n_clusters = len(np.unique(cluster_labels))
    
    for i in range(n_clusters):
        cluster_data = df_unscaled[df_unscaled['Cluster'] == i]
        print(f"\n--- Cluster {i} ---")
        print(f"Size: {len(cluster_data)} samples ({len(cluster_data)/len(df_unscaled)*100:.1f}%)")
        print("Mean values:")
        print(cluster_data.drop('Cluster', axis=1).mean())
    
    # Export clustered data
    print("\n3. Exporting clustered data:")
    df_unscaled.to_csv('output/clustered_data.csv', index=False)
    print("Clustered data saved to: output/clustered_data.csv")
    
    print("\nCluster interpretation completed!\n")
    return df_unscaled


def train_classifier(df_with_clusters):
    """Train DecisionTree classifier using cluster labels as target"""
    print("=" * 80)
    print("DECISION TREE CLASSIFICATION")
    print("=" * 80)
    
    # Prepare features and target
    print("\n1. Preparing data for classification:")
    X = df_with_clusters.drop('Cluster', axis=1)
    y = df_with_clusters['Cluster']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Split data
    print("\n2. Splitting data (80% train, 20% test):")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train Decision Tree
    print("\n3. Training Decision Tree classifier:")
    dt_classifier = DecisionTreeClassifier(
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    dt_classifier.fit(X_train, y_train)
    print("Decision Tree trained successfully")
    
    # Make predictions
    print("\n4. Making predictions:")
    y_pred = dt_classifier.predict(X_test)
    
    # Evaluate model
    print("\n5. MODEL EVALUATION:")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Feature importance
    print("\n6. Feature importance:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': dt_classifier.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance)
    
    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'][:10], feature_importance['importance'][:10])
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('visualizations/feature_importance.png', dpi=300)
    print("\nFeature importance plot saved to: visualizations/feature_importance.png")
    plt.close()
    
    # Save classifier
    print("\n7. Saving classifier:")
    joblib.dump(dt_classifier, 'models/decision_tree_model.pkl')
    print("Decision Tree model saved to: models/decision_tree_model.pkl")
    
    # Save evaluation metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    joblib.dump(metrics, 'models/evaluation_metrics.pkl')
    print("Evaluation metrics saved to: models/evaluation_metrics.pkl")
    
    print("\nClassification completed!\n")
    return dt_classifier, metrics


def main():
    """Main pipeline execution"""
    print("\n" + "=" * 80)
    print("HYBRID ML PIPELINE FOR TRANSACTION PATTERN ANALYSIS")
    print("=" * 80)
    print("\nPipeline stages:")
    print("1. Data Generation")
    print("2. Exploratory Data Analysis (EDA)")
    print("3. Data Preprocessing")
    print("4. KMeans Clustering")
    print("5. PCA Visualization")
    print("6. Cluster Interpretation")
    print("7. Decision Tree Classification")
    print("=" * 80 + "\n")
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Stage 1: Generate or load dataset
    df = generate_sample_dataset(n_samples=1000)
    
    # Stage 2: EDA
    df = perform_eda(df)
    
    # Stage 3: Preprocessing
    df_scaled, df_processed, scaler = preprocess_data(df)
    
    # Stage 4: Clustering
    kmeans_model, cluster_labels = perform_clustering(df_scaled)
    
    # Stage 5: PCA Visualization
    pca_model, pca_df = perform_pca_visualization(df_scaled, cluster_labels)
    
    # Stage 6: Cluster Interpretation
    df_with_clusters = interpret_clusters(df_scaled, cluster_labels, scaler)
    
    # Stage 7: Classification
    dt_model, metrics = train_classifier(df_with_clusters)
    
    # Final summary
    print("=" * 80)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  Data:")
    print("    - data/transactions.csv")
    print("    - output/clustered_data.csv")
    print("  Models:")
    print("    - models/label_encoders.pkl")
    print("    - models/scaler.pkl")
    print("    - models/kmeans_model.pkl")
    print("    - models/pca_model.pkl")
    print("    - models/decision_tree_model.pkl")
    print("    - models/evaluation_metrics.pkl")
    print("  Visualizations:")
    print("    - visualizations/correlation_matrix.png")
    print("    - visualizations/histograms.png")
    print("    - visualizations/elbow_silhouette.png")
    print("    - visualizations/pca_clusters.png")
    print("    - visualizations/feature_importance.png")
    print("\nFinal Model Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
