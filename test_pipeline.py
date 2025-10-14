"""
Test script to validate the ML pipeline and model loading
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

def test_files_exist():
    """Test that all required files are generated"""
    print("Testing file generation...")
    
    required_files = [
        'data/transactions.csv',
        'output/clustered_data.csv',
        'models/label_encoders.pkl',
        'models/scaler.pkl',
        'models/kmeans_model.pkl',
        'models/pca_model.pkl',
        'models/decision_tree_model.pkl',
        'models/evaluation_metrics.pkl',
        'visualizations/correlation_matrix.png',
        'visualizations/histograms.png',
        'visualizations/elbow_silhouette.png',
        'visualizations/pca_clusters.png',
        'visualizations/feature_importance.png',
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✓ All required files exist")
        return True


def test_model_loading():
    """Test that models can be loaded correctly"""
    print("\nTesting model loading...")
    
    try:
        # Load models
        scaler = joblib.load('models/scaler.pkl')
        kmeans = joblib.load('models/kmeans_model.pkl')
        pca = joblib.load('models/pca_model.pkl')
        dt_classifier = joblib.load('models/decision_tree_model.pkl')
        label_encoders = joblib.load('models/label_encoders.pkl')
        metrics = joblib.load('models/evaluation_metrics.pkl')
        
        # Validate model types
        assert isinstance(scaler, StandardScaler), "Scaler is not StandardScaler"
        assert isinstance(kmeans, KMeans), "KMeans model is not KMeans"
        assert isinstance(pca, PCA), "PCA model is not PCA"
        assert isinstance(dt_classifier, DecisionTreeClassifier), "Classifier is not DecisionTree"
        assert isinstance(label_encoders, dict), "Label encoders is not a dictionary"
        assert isinstance(metrics, dict), "Metrics is not a dictionary"
        
        print("✓ All models loaded successfully")
        print(f"  - Scaler features: {scaler.n_features_in_}")
        print(f"  - KMeans clusters: {kmeans.n_clusters}")
        print(f"  - PCA components: {pca.n_components}")
        print(f"  - Decision Tree depth: {dt_classifier.get_depth()}")
        print(f"  - Label encoders: {list(label_encoders.keys())}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False


def test_data_loading():
    """Test that data files can be loaded correctly"""
    print("\nTesting data loading...")
    
    try:
        # Load original data
        df_original = pd.read_csv('data/transactions.csv')
        print(f"✓ Original data loaded: {df_original.shape}")
        
        # Load clustered data
        df_clustered = pd.read_csv('output/clustered_data.csv')
        print(f"✓ Clustered data loaded: {df_clustered.shape}")
        
        # Validate clustered data has cluster column
        assert 'Cluster' in df_clustered.columns, "Cluster column not found"
        print(f"  - Number of clusters: {df_clustered['Cluster'].nunique()}")
        print(f"  - Cluster distribution:\n{df_clustered['Cluster'].value_counts().sort_index()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False


def test_prediction():
    """Test making predictions with loaded models"""
    print("\nTesting prediction pipeline...")
    
    try:
        # Load models
        scaler = joblib.load('models/scaler.pkl')
        kmeans = joblib.load('models/kmeans_model.pkl')
        dt_classifier = joblib.load('models/decision_tree_model.pkl')
        
        # Create sample data
        sample_data = pd.DataFrame({
            'amount': [100.0],
            'transaction_count': [5],
            'account_age_days': [1000],
            'transaction_hour': [14],
            'merchant_category': [1],
            'device_type': [0],
            'location_match': [1],
            'avg_transaction_amount': [20.0],
            'risk_score': [0.5],
            'amount_bin': [2]
        })
        
        # Scale data
        sample_scaled = scaler.transform(sample_data)
        
        # Predict cluster
        cluster = kmeans.predict(sample_scaled)
        print(f"✓ Cluster prediction: {cluster[0]}")
        
        # Predict with classifier
        prediction = dt_classifier.predict(sample_scaled)
        print(f"✓ Classifier prediction: {prediction[0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error making predictions: {e}")
        return False


def test_metrics():
    """Test that metrics are reasonable"""
    print("\nTesting model metrics...")
    
    try:
        metrics = joblib.load('models/evaluation_metrics.pkl')
        
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        
        # Check if metrics are within reasonable range
        assert 0 <= metrics['accuracy'] <= 1, "Accuracy out of range"
        assert 0 <= metrics['precision'] <= 1, "Precision out of range"
        assert 0 <= metrics['recall'] <= 1, "Recall out of range"
        assert 0 <= metrics['f1_score'] <= 1, "F1-score out of range"
        
        print("✓ All metrics are within valid range")
        return True
        
    except Exception as e:
        print(f"❌ Error validating metrics: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 80)
    print("TESTING ML PIPELINE")
    print("=" * 80)
    
    tests = [
        test_files_exist,
        test_data_loading,
        test_model_loading,
        test_prediction,
        test_metrics
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    print("\n" + "=" * 80)
    print(f"TEST RESULTS: {sum(results)}/{len(results)} PASSED")
    print("=" * 80)
    
    if all(results):
        print("\n✓ All tests passed successfully!")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    exit(main())
