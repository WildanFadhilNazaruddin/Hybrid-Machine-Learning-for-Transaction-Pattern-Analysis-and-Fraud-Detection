# Copilot Instructions for AI Coding Agents

## Project Overview
This repository implements a hybrid machine learning pipeline for transaction pattern analysis and fraud detection. The approach combines unsupervised clustering (K-Means) for label generation with supervised classification models for fraud prediction.

## Architecture & Data Flow
- **Data Ingestion:** Raw transaction data is loaded and preprocessed.
- **Unsupervised Labeling:** K-Means clustering is used to group transactions and generate pseudo-labels for downstream tasks.
- **Supervised Classification:** Classification models are trained using the generated labels to predict fraudulent transactions.
- **Evaluation:** Model performance is assessed using appropriate metrics (e.g., accuracy, precision, recall).

## Key Conventions & Patterns
- The project is organized as a single-repo pipeline. Expect scripts or notebooks for each major stage (preprocessing, clustering, classification, evaluation).
- K-Means is the default clustering method for label generation.
- Classification models may include logistic regression, decision trees, or other standard classifiers.
- Data and model artifacts are typically stored in local directories (e.g., `data/`, `models/`).

## Developer Workflows
- **Run the pipeline:** Execute scripts sequentially for each stage, or use a main orchestrator script if present.
- **Testing:** If test scripts exist, run them using standard Python test runners (e.g., `pytest`).
- **Dependencies:** Install requirements via `pip install -r requirements.txt` if the file exists.
- **Debugging:** Use print statements or logging for stepwise inspection; check for Jupyter notebooks for interactive exploration.

## Integration & External Dependencies
- Relies on standard Python ML libraries (e.g., scikit-learn, pandas, numpy).
- No evidence of cloud or external service integration in the current structure.

## Examples & References
- See `README.md` for a high-level summary.
- Look for scripts or notebooks in the root or subdirectories for implementation details.

## Guidance for AI Agents
- Follow the pipeline structure: preprocess → cluster → classify → evaluate.
- Use K-Means for clustering unless otherwise specified.
- Maintain modularity: separate data handling, modeling, and evaluation logic.
- Document any new scripts or workflows clearly in the README or as inline comments.

