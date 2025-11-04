# Hybrid-Machine-Learning-for-Transaction-Pattern-Analysis-and-Fraud-Detection

A machine learning project integrating unsupervised clustering and supervised classification to identify fraud patterns in financial transactions. Using K-Means for label generation and classification models for prediction, this project demonstrates an end-to-end hybrid ML pipeline for fraud detection.

## Overview

This project implements a complete hybrid ML pipeline that combines:
- **Unsupervised Learning**: KMeans clustering to discover transaction patterns
- **Supervised Learning**: Decision Tree classification for pattern prediction

## Features

### 1. Exploratory Data Analysis (EDA)
- Display first rows (`head`)
- Dataset information (`info`)
- Statistical summary (`describe`)
- Correlation matrix analysis
- Feature distribution histograms

### 2. Data Preprocessing
- Null value handling (median imputation)
- Duplicate removal
- ID column removal
- Outlier handling (IQR method with capping)
- Feature binning (amount categories)
- Categorical encoding (Label Encoding)
- Feature scaling (StandardScaler)

### 3. KMeans Clustering
- Elbow method for optimal cluster selection
- Silhouette score analysis
- PCA dimensionality reduction and visualization
- Model persistence (save/load)

### 4. Cluster Interpretation
- Inverse transformation to original scale
- Statistical analysis per cluster
- Data export with cluster labels
- Cluster summary generation

### 5. Decision Tree Classification
- Train-test split (80/20)
- Model training on cluster labels
- Comprehensive evaluation:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
- Feature importance analysis
- Model persistence

## Requirements

```
scikit-learn==1.7.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
yellowbrick>=1.5
setuptools>=68.0.0
```

**Note:** For Python 3.12+, `setuptools>=68.0.0` is required to provide `distutils` compatibility for the `yellowbrick` library.

## Installation

```bash
# Clone repository
git clone https://github.com/WildanFadhilNazaruddin/Hybrid-Machine-Learning-for-Transaction-Pattern-Analysis-and-Fraud-Detection.git
cd Hybrid-Machine-Learning-for-Transaction-Pattern-Analysis-and-Fraud-Detection

# Install dependencies
pip install -r requirements.txt

# Hybrid Machine Learning for Transaction Pattern Analysis and Fraud Detection

## Struktur Proyek

- `data/` — Dataset mentah dan hasil intermediate (preprocessed, clustered)
- `notebooks/` — Notebook eksplorasi dan dokumentasi pipeline
- `scripts/` — Script modular untuk EDA, preprocessing, clustering, dan klasifikasi
- `models/` — Model terlatih dan artefak terkait
- `tests/` — Unit test untuk preprocessing, clustering, dan klasifikasi

## Alur Pengembangan & Branching
- `feat/` — Fitur baru (EDA, preprocessing, modeling)
- `fix/` — Perbaikan bug minor
- `hotfix/` — Perbaikan kritis/produksi
- `docs/` — Dokumentasi (README, penjelasan pipeline)
- `test/` — Unit test fungsi & validasi model
- `refactor/` — Perapian struktur & code cleanup

## Pipeline ML
1. **EDA:**
   Jalankan `python scripts/eda.py` untuk eksplorasi awal data.
2. **Preprocessing:**
   Jalankan `python scripts/preprocessing.py` untuk normalisasi data.
3. **Clustering:**
   Jalankan `python scripts/clustering.py` untuk pseudo-labeling dengan KMeans.
4. **Classification:**
   Jalankan `python scripts/classification.py` untuk klasifikasi Decision Tree.

## Dependensi
- scikit-learn==1.7.0
- pandas
- matplotlib

Instalasi: `pip install -r requirements.txt`

## Testing
Jalankan `pytest tests/` untuk menjalankan seluruh unit test.

### Step 1: Generate Sample Dataset

```bash
python generate_dataset.py
```

This creates `transaction_data.csv` with synthetic transaction data including:
- Transaction amount
- Time of day
- Day of week
- Merchant category
- Transaction type
- Location distance
- Previous transactions count
- Account age
- Account balance

### Step 2: Run the Complete Pipeline

```bash
python ml_pipeline.py
```

The pipeline will execute all stages and generate:

#### Visualizations:
- `eda_correlation_heatmap.png` - Feature correlation matrix
- `eda_histograms.png` - Feature distributions
- `clustering_elbow_silhouette.png` - Cluster optimization metrics
- `clustering_pca_visualization.png` - PCA cluster visualization
- `classification_feature_importance.png` - Feature importance chart

#### Models:
- `kmeans_model.pkl` - Trained KMeans clustering model
- `decision_tree_model.pkl` - Trained Decision Tree classifier
- `scaler.pkl` - StandardScaler for feature normalization
- `pca_model.pkl` - PCA transformation model
- `label_encoders.pkl` - Label encoders for categorical features

#### Data Files:
- `data_with_clusters.csv` - Original data with cluster assignments
- `cluster_summary.csv` - Statistical summary per cluster
- `classification_metrics.csv` - Model performance metrics

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Load Data                              │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│              Exploratory Data Analysis                      │
│  • head() • info() • describe() • correlation • histograms  │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                Data Preprocessing                           │
│  • Handle nulls • Remove duplicates • Drop IDs             │
│  • Handle outliers • Binning • Encoding • Scaling          │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                KMeans Clustering                            │
│  • Elbow method • Silhouette score • PCA • Save model      │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│             Cluster Interpretation                          │
│  • Inverse transform • Cluster statistics • Export data    │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│          Decision Tree Classification                       │
│  • Train-test split • Model training • Evaluation          │
│  • Accuracy • Precision • Recall • F1 • Save model         │
└─────────────────────────────────────────────────────────────┘
```

## Example Output

```
================================================================================
EXPLORATORY DATA ANALYSIS (EDA)
================================================================================

1. First 5 rows:
   [Dataset preview]

2. Dataset Info:
   [Column information]

3. Statistical Summary:
   [Descriptive statistics]

4. Correlation Matrix:
   [Correlation values]

================================================================================
DATA PREPROCESSING
================================================================================

1. Handling null values...
   Filled balance nulls with median: 5024.32
   Null values after: 0

2. Removing duplicates...
   Duplicates after: 0

3. Dropping ID columns...
   Dropping columns: ['transaction_id']

4. Handling outliers (IQR method)...
   Total outliers handled: 150

5. Creating binned features...
   Created 'amount_category' with bins

6. Encoding categorical variables...
   Encoded 5 features

7. Scaling numerical features...
   Scaled 10 numerical features

================================================================================
KMEANS CLUSTERING
================================================================================

1. Elbow Method Analysis...
   K=2: Inertia=8450.23, Silhouette=0.425
   K=3: Inertia=6234.56, Silhouette=0.456
   ...

2. Optimal number of clusters: 3

3. Fitting final KMeans model...
   Final Silhouette Score: 0.456
   Cluster sizes: [320, 345, 335]

4. PCA visualization...
   PCA explained variance: 0.345, 0.223
   Total explained variance: 0.568

================================================================================
DECISION TREE CLASSIFICATION
================================================================================

1. Splitting data (test_size=0.2)...
   Training set: 800 samples
   Test set: 200 samples

2. Training Decision Tree Classifier...
   Model trained successfully

3. Making predictions...

4. Model Evaluation:
   Test Set Performance:
     - Accuracy:  0.9450
     - Precision: 0.9448
     - Recall:    0.9450
     - F1-Score:  0.9447
```

## Project Structure

```
.
├── README.md                              # Project documentation
├── requirements.txt                       # Python dependencies
├── generate_dataset.py                    # Dataset generator
├── ml_pipeline.py                         # Main pipeline script
├── transaction_data.csv                   # Generated dataset (gitignored)
├── *.pkl                                  # Trained models (gitignored)
├── *.png                                  # Visualizations (gitignored)
└── *_clusters.csv                         # Results (gitignored)
```

## Key Technologies

- **scikit-learn 1.7.0**: Machine learning algorithms
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib/seaborn**: Data visualization
- **joblib**: Model serialization

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

Wildan Fadhil Nazaruddin
