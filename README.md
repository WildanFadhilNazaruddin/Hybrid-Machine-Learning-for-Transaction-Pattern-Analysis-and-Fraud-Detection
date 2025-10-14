# Hybrid Machine Learning for Transaction Pattern Analysis and Fraud Detection

A machine learning project integrating unsupervised clustering and supervised classification to identify fraud patterns in financial transactions. Using K-Means for label generation and classification models for prediction, this project demonstrates an end-to-end hybrid ML pipeline for fraud detection.

## Overview

This project implements a complete hybrid ML pipeline that combines:
- **Unsupervised Learning**: K-Means clustering for pattern discovery
- **Supervised Learning**: Decision Tree classification for prediction
- **Dimensionality Reduction**: PCA for visualization

## Features

### 1. Exploratory Data Analysis (EDA)
- Display first rows with `head()`
- Dataset information with `info()`
- Statistical summary with `describe()`
- Correlation matrix with heatmap visualization
- Histograms for all numeric features

### 2. Data Preprocessing
- Handle null values (median imputation)
- Remove duplicate rows
- Drop ID columns
- Encode categorical variables (Label Encoding)
- Handle outliers using IQR method
- Feature binning (discretization)
- Feature scaling with StandardScaler

### 3. K-Means Clustering
- Elbow method for optimal cluster selection
- Silhouette analysis
- Model saving and loading
- Cluster label generation

### 4. PCA Visualization
- Dimensionality reduction to 2D
- Cluster visualization
- Explained variance analysis

### 5. Cluster Interpretation
- Inverse transformation of scaled data
- Statistical analysis per cluster
- Export clustered data

### 6. Decision Tree Classification
- Train/test split (80/20)
- Model training and evaluation
- Performance metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
- Feature importance analysis
- Model persistence

## Requirements

- Python 3.8+
- scikit-learn 1.7.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- joblib >= 1.3.0

## Installation

1. Clone the repository:
```bash
git clone https://github.com/WildanFadhilNazaruddin/Hybrid-Machine-Learning-for-Transaction-Pattern-Analysis-and-Fraud-Detection.git
cd Hybrid-Machine-Learning-for-Transaction-Pattern-Analysis-and-Fraud-Detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the complete ML pipeline:
```bash
python ml_pipeline.py
```

The pipeline will:
1. Generate a sample transaction dataset
2. Perform comprehensive EDA
3. Preprocess the data
4. Apply K-Means clustering
5. Visualize clusters using PCA
6. Interpret cluster characteristics
7. Train a Decision Tree classifier
8. Evaluate and save all models

## Output Structure

After running the pipeline, the following directories and files will be created:

```
├── data/
│   └── transactions.csv              # Generated sample dataset
├── models/
│   ├── label_encoders.pkl            # Categorical encoders
│   ├── scaler.pkl                    # StandardScaler model
│   ├── kmeans_model.pkl              # K-Means clustering model
│   ├── pca_model.pkl                 # PCA model
│   ├── decision_tree_model.pkl       # Decision Tree classifier
│   └── evaluation_metrics.pkl        # Model performance metrics
├── output/
│   └── clustered_data.csv            # Data with cluster labels
└── visualizations/
    ├── correlation_matrix.png        # Feature correlation heatmap
    ├── histograms.png                # Distribution plots
    ├── elbow_silhouette.png          # Clustering optimization plots
    ├── pca_clusters.png              # 2D cluster visualization
    └── feature_importance.png        # Feature importance bar chart
```

## Pipeline Stages

### Stage 1: Data Generation
- Creates synthetic transaction data with realistic patterns
- 1000 transactions with multiple features
- Includes both numerical and categorical variables

### Stage 2: EDA
- Statistical analysis and visualization
- Correlation analysis
- Distribution analysis

### Stage 3: Preprocessing
- Data cleaning and transformation
- Feature engineering
- Normalization and encoding

### Stage 4: Clustering
- K-Means with automatic cluster selection
- Quality metrics (Silhouette score)
- Pattern identification

### Stage 5: PCA
- Dimensionality reduction
- Visual interpretation
- Variance analysis

### Stage 6: Interpretation
- Cluster profiling
- Statistical summaries
- Data export

### Stage 7: Classification
- Supervised learning on cluster labels
- Performance evaluation
- Model persistence

## Model Performance

The pipeline typically achieves:
- **Accuracy**: ~98%
- **Precision**: ~98%
- **Recall**: ~98%
- **F1-Score**: ~98%

*Note: Performance metrics may vary based on the generated sample data.*

## Customization

To use your own dataset:

1. Replace the `generate_sample_dataset()` function call in `main()` with:
```python
df = pd.read_csv('your_data.csv')
```

2. Adjust preprocessing parameters based on your data characteristics

3. Tune model hyperparameters:
   - K-Means: `n_clusters_range`
   - Decision Tree: `max_depth`, `min_samples_split`, `min_samples_leaf`

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

Wildan Fadhil Nazaruddin

## Acknowledgments

- Built with scikit-learn 1.7.0
- Implements standard ML best practices
- Follows hybrid learning approach for fraud detection
