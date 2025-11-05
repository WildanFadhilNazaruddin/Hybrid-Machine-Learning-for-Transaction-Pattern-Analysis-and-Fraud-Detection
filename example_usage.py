"""
Example usage of the trained models for inference
"""

import joblib
import pandas as pd
import numpy as np

def load_models():
    """Load all trained models"""
    print("Loading models...")
    models = {
        'scaler': joblib.load('models/scaler.pkl'),
        'kmeans': joblib.load('models/kmeans_model.pkl'),
        'pca': joblib.load('models/pca_model.pkl'),
        'classifier': joblib.load('models/decision_tree_model.pkl'),
        'label_encoders': joblib.load('models/label_encoders.pkl'),
        'metrics': joblib.load('models/evaluation_metrics.pkl')
    }
    print("✓ Models loaded successfully\n")
    return models


def predict_transaction(models, transaction_data):
    """
    Predict cluster and classification for a new transaction
    
    Parameters:
    -----------
    models : dict
        Dictionary containing all loaded models
    transaction_data : dict
        Transaction features (categorical features should be strings or already encoded as integers)
    
    Returns:
    --------
    dict : Prediction results
    """
    # Create DataFrame from input
    df = pd.DataFrame([transaction_data])
    
    # Encode categorical features if they are strings
    for col, encoder in models['label_encoders'].items():
        if col in df.columns and len(df) > 0:
            # Only transform if the column contains string values
            if df[col].dtype == 'object' or (not pd.isna(df[col].iloc[0]) and isinstance(df[col].iloc[0], str)):
                df[col] = encoder.transform(df[col])
            # If already numeric, assume it's already encoded
    
    # Scale features
    df_scaled = models['scaler'].transform(df)
    
    # Predict cluster
    cluster = models['kmeans'].predict(df_scaled)[0]
    
    # Predict with classifier
    classification = models['classifier'].predict(df_scaled)[0]
    
    # Get PCA coordinates for visualization
    pca_coords = models['pca'].transform(df_scaled)[0]
    
    return {
        'cluster': int(cluster),
        'classification': int(classification),
        'pca_pc1': float(pca_coords[0]),
        'pca_pc2': float(pca_coords[1])
    }


def main():
    """Example usage"""
    print("=" * 80)
    print("EXAMPLE: Using Trained Models for Inference")
    print("=" * 80 + "\n")
    
    # Load models
    models = load_models()
    
    # Display model performance
    print("Model Performance:")
    metrics = models['metrics']
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}\n")
    
    # Example transactions with encoded values
    # Note: In a real scenario, you would pass raw string values and the function
    # would encode them. Here we use pre-encoded values for demonstration.
    examples = [
        {
            'name': 'High-value transaction',
            'data': {
                'amount': 500.0,
                'transaction_count': 3,
                'account_age_days': 1200,
                'transaction_hour': 2,
                'merchant_category': 1,  # Encoded value (e.g., 'online')
                'device_type': 0,        # Encoded value (e.g., 'desktop')
                'location_match': 0,
                'avg_transaction_amount': 150.0,
                'risk_score': 2.5,
                'amount_bin': 4
            }
        },
        {
            'name': 'Low-value transaction',
            'data': {
                'amount': 25.0,
                'transaction_count': 5,
                'account_age_days': 2000,
                'transaction_hour': 14,
                'merchant_category': 2,  # Encoded value (e.g., 'restaurant')
                'device_type': 1,        # Encoded value (e.g., 'mobile')
                'location_match': 1,
                'avg_transaction_amount': 5.0,
                'risk_score': 0.1,
                'amount_bin': 1
            }
        },
        {
            'name': 'Medium transaction with unusual pattern',
            'data': {
                'amount': 100.0,
                'transaction_count': 10,
                'account_age_days': 500,
                'transaction_hour': 23,
                'merchant_category': 3,  # Encoded value (e.g., 'gas_station')
                'device_type': 2,        # Encoded value (e.g., 'tablet')
                'location_match': 0,
                'avg_transaction_amount': 10.0,
                'risk_score': 1.5,
                'amount_bin': 2
            }
        }
    ]
    
    # Make predictions
    print("Making predictions for example transactions:\n")
    for i, example in enumerate(examples, 1):
        print(f"Example {i}: {example['name']}")
        print(f"  Input features: {example['data']}")
        
        result = predict_transaction(models, example['data'])
        
        print(f"  Predictions:")
        print(f"    - Cluster: {result['cluster']}")
        print(f"    - Classification: {result['classification']}")
        print(f"    - PCA Coordinates: PC1={result['pca_pc1']:.2f}, PC2={result['pca_pc2']:.2f}")
        print()
    
    print("=" * 80)
    print("Inference examples completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
