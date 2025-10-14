"""
Hybrid Machine Learning Pipeline for Transaction Pattern Analysis and Fraud Detection

This pipeline implements:
1. EDA (Exploratory Data Analysis)
2. Data Preprocessing
3. KMeans Clustering with optimization
4. Decision Tree Classification
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import silhouette_score
import joblib
import warnings
warnings.filterwarnings('ignore')


class TransactionMLPipeline:
    """Complete ML Pipeline for Transaction Analysis"""
    
    def __init__(self, data_path='transaction_data.csv'):
        """Initialize pipeline with data path"""
        self.data_path = data_path
        self.df = None
        self.df_processed = None
        self.scaler = None
        self.label_encoders = {}
        self.kmeans_model = None
        self.pca_model = None
        self.dt_model = None
        self.cluster_labels = None
        
    def load_data(self):
        """Load transaction data"""
        print("="*80)
        print("LOADING DATA")
        print("="*80)
        self.df = pd.read_csv(self.data_path)
        print(f"Data loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns\n")
        return self
    
    def exploratory_data_analysis(self):
        """Perform EDA: head, info, describe, correlation, histograms"""
        print("="*80)
        print("EXPLORATORY DATA ANALYSIS (EDA)")
        print("="*80)
        
        # 1. Head
        print("\n1. First 5 rows:")
        print(self.df.head())
        
        # 2. Info
        print("\n2. Dataset Info:")
        print(self.df.info())
        
        # 3. Describe
        print("\n3. Statistical Summary:")
        print(self.df.describe())
        
        # 4. Correlation matrix for numerical columns
        print("\n4. Correlation Matrix:")
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.df[numerical_cols].corr()
        print(corr_matrix)
        
        # Save correlation heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('eda_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("Correlation heatmap saved: eda_correlation_heatmap.png")
        plt.close()
        
        # 5. Histograms
        print("\n5. Generating histograms for numerical features...")
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, col in enumerate(numerical_cols[:9]):
            axes[idx].hist(self.df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'Distribution of {col}')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('eda_histograms.png', dpi=300, bbox_inches='tight')
        print("Histograms saved: eda_histograms.png\n")
        plt.close()
        
        return self
    
    def preprocess_data(self):
        """
        Preprocessing steps:
        1. Handle null values
        2. Remove duplicates
        3. Drop ID columns
        4. Handle outliers (IQR method)
        5. Binning
        6. Encoding categorical variables
        7. Scaling numerical features
        """
        print("="*80)
        print("DATA PREPROCESSING")
        print("="*80)
        
        self.df_processed = self.df.copy()
        
        # 1. Handle null values
        print("\n1. Handling null values...")
        print(f"Null values before: {self.df_processed.isnull().sum().sum()}")
        # Fill numerical nulls with median
        numerical_cols = self.df_processed.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if self.df_processed[col].isnull().sum() > 0:
                median_val = self.df_processed[col].median()
                self.df_processed[col].fillna(median_val, inplace=True)
                print(f"  - Filled {col} nulls with median: {median_val:.2f}")
        print(f"Null values after: {self.df_processed.isnull().sum().sum()}")
        
        # 2. Remove duplicates
        print("\n2. Removing duplicates...")
        print(f"Duplicates before: {self.df_processed.duplicated().sum()}")
        self.df_processed.drop_duplicates(inplace=True)
        self.df_processed.reset_index(drop=True, inplace=True)
        print(f"Duplicates after: {self.df_processed.duplicated().sum()}")
        print(f"Rows remaining: {len(self.df_processed)}")
        
        # 3. Drop ID columns
        print("\n3. Dropping ID columns...")
        id_cols = [col for col in self.df_processed.columns if 'id' in col.lower()]
        if id_cols:
            print(f"Dropping columns: {id_cols}")
            self.df_processed.drop(columns=id_cols, inplace=True)
        else:
            print("No ID columns found")
        
        # 4. Handle outliers using IQR method
        print("\n4. Handling outliers (IQR method)...")
        numerical_cols = self.df_processed.select_dtypes(include=[np.number]).columns
        outlier_count = 0
        
        for col in numerical_cols:
            Q1 = self.df_processed[col].quantile(0.25)
            Q3 = self.df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((self.df_processed[col] < lower_bound) | 
                       (self.df_processed[col] > upper_bound)).sum()
            
            if outliers > 0:
                # Cap outliers instead of removing
                self.df_processed[col] = self.df_processed[col].clip(lower_bound, upper_bound)
                outlier_count += outliers
                print(f"  - {col}: {outliers} outliers capped")
        
        print(f"Total outliers handled: {outlier_count}")
        
        # 5. Binning for amount
        print("\n5. Creating binned features...")
        if 'amount' in self.df_processed.columns:
            self.df_processed['amount_category'] = pd.cut(
                self.df_processed['amount'], 
                bins=[0, 50, 150, 500, float('inf')],
                labels=['low', 'medium', 'high', 'very_high']
            )
            print("  - Created 'amount_category' with bins: [0-50, 50-150, 150-500, 500+]")
        
        # 6. Encoding categorical variables
        print("\n6. Encoding categorical variables...")
        categorical_cols = self.df_processed.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            self.df_processed[col] = le.fit_transform(self.df_processed[col].astype(str))
            self.label_encoders[col] = le
            print(f"  - Encoded {col}: {len(le.classes_)} unique values")
        
        # 7. Scaling numerical features
        print("\n7. Scaling numerical features...")
        numerical_cols = self.df_processed.select_dtypes(include=[np.number]).columns
        self.scaler = StandardScaler()
        self.df_processed[numerical_cols] = self.scaler.fit_transform(self.df_processed[numerical_cols])
        print(f"  - Scaled {len(numerical_cols)} numerical features using StandardScaler")
        
        print(f"\nPreprocessed data shape: {self.df_processed.shape}")
        return self
    
    def kmeans_clustering(self, max_clusters=10):
        """
        Perform KMeans clustering with:
        - Elbow method
        - Silhouette score
        - PCA visualization
        - Model saving
        """
        print("="*80)
        print("KMEANS CLUSTERING")
        print("="*80)
        
        X = self.df_processed.values
        
        # Elbow Method
        print("\n1. Elbow Method Analysis...")
        inertias = []
        silhouette_scores = []
        K_range = range(2, max_clusters + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, kmeans.labels_))
            print(f"  - K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.3f}")
        
        # Plot Elbow curve
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(K_range, inertias, 'bo-')
        ax1.set_xlabel('Number of Clusters (K)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method')
        ax1.grid(True)
        
        # Plot Silhouette scores
        ax2.plot(K_range, silhouette_scores, 'go-')
        ax2.set_xlabel('Number of Clusters (K)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('clustering_elbow_silhouette.png', dpi=300, bbox_inches='tight')
        print("\nElbow and Silhouette plots saved: clustering_elbow_silhouette.png")
        plt.close()
        
        # Select optimal K based on silhouette score
        optimal_k = K_range[np.argmax(silhouette_scores)]
        print(f"\n2. Optimal number of clusters: {optimal_k}")
        
        # Fit final KMeans model
        print("\n3. Fitting final KMeans model...")
        self.kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        self.cluster_labels = self.kmeans_model.fit_predict(X)
        
        final_silhouette = silhouette_score(X, self.cluster_labels)
        print(f"  - Final Silhouette Score: {final_silhouette:.3f}")
        print(f"  - Cluster sizes: {np.bincount(self.cluster_labels)}")
        
        # PCA for visualization
        print("\n4. PCA visualization...")
        self.pca_model = PCA(n_components=2, random_state=42)
        X_pca = self.pca_model.fit_transform(X)
        
        explained_variance = self.pca_model.explained_variance_ratio_
        print(f"  - PCA explained variance: {explained_variance[0]:.3f}, {explained_variance[1]:.3f}")
        print(f"  - Total explained variance: {sum(explained_variance):.3f}")
        
        # Plot PCA clusters
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                            c=self.cluster_labels, cmap='viridis', 
                            alpha=0.6, edgecolors='black')
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')
        plt.title('KMeans Clustering - PCA Visualization')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('clustering_pca_visualization.png', dpi=300, bbox_inches='tight')
        print("PCA visualization saved: clustering_pca_visualization.png")
        plt.close()
        
        # Save models
        print("\n5. Saving models...")
        joblib.dump(self.kmeans_model, 'kmeans_model.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')
        joblib.dump(self.pca_model, 'pca_model.pkl')
        joblib.dump(self.label_encoders, 'label_encoders.pkl')
        print("Models saved: kmeans_model.pkl, scaler.pkl, pca_model.pkl, label_encoders.pkl")
        
        return self
    
    def interpret_clusters(self):
        """
        Interpret clusters and export data with cluster labels
        Includes inverse transformation to original scale
        """
        print("="*80)
        print("CLUSTER INTERPRETATION")
        print("="*80)
        
        # Add cluster labels to processed data
        df_with_clusters = self.df_processed.copy()
        df_with_clusters['cluster'] = self.cluster_labels
        
        # Inverse transform to original scale for interpretation
        print("\n1. Inverse transforming to original scale...")
        numerical_cols = self.df_processed.select_dtypes(include=[np.number]).columns
        df_inverse = self.df_processed.copy()
        df_inverse[numerical_cols] = self.scaler.inverse_transform(self.df_processed[numerical_cols])
        df_inverse['cluster'] = self.cluster_labels
        
        # Cluster statistics
        print("\n2. Cluster Statistics (Original Scale):")
        print("-" * 80)
        for cluster_id in sorted(df_inverse['cluster'].unique()):
            cluster_data = df_inverse[df_inverse['cluster'] == cluster_id]
            print(f"\nCluster {cluster_id} (n={len(cluster_data)}):")
            print(cluster_data.describe().round(2))
        
        # Export data
        print("\n3. Exporting clustered data...")
        df_inverse.to_csv('data_with_clusters.csv', index=False)
        print("Clustered data saved: data_with_clusters.csv")
        
        # Export cluster summary
        cluster_summary = df_inverse.groupby('cluster').agg(['mean', 'std', 'min', 'max'])
        cluster_summary.to_csv('cluster_summary.csv')
        print("Cluster summary saved: cluster_summary.csv")
        
        return self
    
    def decision_tree_classification(self, test_size=0.2):
        """
        Train Decision Tree classifier using cluster labels as target
        Includes train_test_split and full evaluation
        """
        print("="*80)
        print("DECISION TREE CLASSIFICATION")
        print("="*80)
        
        # Prepare data
        X = self.df_processed.values
        y = self.cluster_labels
        
        # Train-test split
        print(f"\n1. Splitting data (test_size={test_size})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        print(f"  - Training set: {X_train.shape[0]} samples")
        print(f"  - Test set: {X_test.shape[0]} samples")
        
        # Train Decision Tree
        print("\n2. Training Decision Tree Classifier...")
        self.dt_model = DecisionTreeClassifier(
            random_state=42,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5
        )
        self.dt_model.fit(X_train, y_train)
        print("Model trained successfully")
        
        # Make predictions
        print("\n3. Making predictions...")
        y_train_pred = self.dt_model.predict(X_train)
        y_test_pred = self.dt_model.predict(X_test)
        
        # Evaluation metrics
        print("\n4. Model Evaluation:")
        print("-" * 80)
        print("\nTraining Set Performance:")
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred, average='weighted', zero_division=0)
        train_recall = recall_score(y_train, y_train_pred, average='weighted', zero_division=0)
        train_f1 = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
        
        print(f"  - Accuracy:  {train_accuracy:.4f}")
        print(f"  - Precision: {train_precision:.4f}")
        print(f"  - Recall:    {train_recall:.4f}")
        print(f"  - F1-Score:  {train_f1:.4f}")
        
        print("\nTest Set Performance:")
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
        
        print(f"  - Accuracy:  {test_accuracy:.4f}")
        print(f"  - Precision: {test_precision:.4f}")
        print(f"  - Recall:    {test_recall:.4f}")
        print(f"  - F1-Score:  {test_f1:.4f}")
        
        # Feature importance
        print("\n5. Top 10 Feature Importances:")
        feature_importance = pd.DataFrame({
            'feature': self.df_processed.columns,
            'importance': self.dt_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(feature_importance.head(10).to_string(index=False))
        
        # Plot feature importance
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Feature Importances - Decision Tree')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('classification_feature_importance.png', dpi=300, bbox_inches='tight')
        print("\nFeature importance plot saved: classification_feature_importance.png")
        plt.close()
        
        # Save model
        print("\n6. Saving Decision Tree model...")
        joblib.dump(self.dt_model, 'decision_tree_model.pkl')
        print("Model saved: decision_tree_model.pkl")
        
        # Save evaluation metrics
        metrics = {
            'train_accuracy': train_accuracy,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_f1': train_f1,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1
        }
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv('classification_metrics.csv', index=False)
        print("Metrics saved: classification_metrics.csv")
        
        return self
    
    def run_complete_pipeline(self):
        """Execute the complete ML pipeline"""
        print("\n" + "="*80)
        print("HYBRID ML PIPELINE FOR TRANSACTION PATTERN ANALYSIS")
        print("="*80 + "\n")
        
        self.load_data()
        self.exploratory_data_analysis()
        self.preprocess_data()
        self.kmeans_clustering()
        self.interpret_clusters()
        self.decision_tree_classification()
        
        print("\n" + "="*80)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nGenerated Files:")
        print("  - eda_correlation_heatmap.png")
        print("  - eda_histograms.png")
        print("  - clustering_elbow_silhouette.png")
        print("  - clustering_pca_visualization.png")
        print("  - classification_feature_importance.png")
        print("  - kmeans_model.pkl")
        print("  - decision_tree_model.pkl")
        print("  - scaler.pkl")
        print("  - pca_model.pkl")
        print("  - label_encoders.pkl")
        print("  - data_with_clusters.csv")
        print("  - cluster_summary.csv")
        print("  - classification_metrics.csv")
        print("="*80 + "\n")


if __name__ == "__main__":
    # Create and run pipeline
    pipeline = TransactionMLPipeline('transaction_data.csv')
    pipeline.run_complete_pipeline()
