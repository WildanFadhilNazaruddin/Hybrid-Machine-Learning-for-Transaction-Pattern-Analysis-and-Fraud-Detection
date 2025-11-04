# Installation Guide

## Prerequisites

- Python 3.8 or higher (tested on Python 3.12)
- pip package manager

## Standard Installation

```bash
# Clone the repository
git clone <repository-url>
cd Hybrid-Machine-Learning-for-Transaction-Pattern-Analysis-and-Fraud-Detection

# Install dependencies
pip install -r requirements.txt
```

Replace `<repository-url>` with the actual repository URL.

## Python 3.12 Compatibility

This project uses the `yellowbrick` library for clustering visualizations (specifically `KElbowVisualizer`). 

**Important:** Yellowbrick versions up to 1.5 use the deprecated `distutils` module, which was removed from Python 3.12. The solution is:

1. **Setuptools provides distutils compatibility**: The `setuptools>=68.0.0` package (included in requirements.txt) provides a `distutils` compatibility layer for Python 3.12+.

2. **No code changes needed**: Once both `yellowbrick>=1.5` and `setuptools>=68.0.0` are installed via `pip install -r requirements.txt`, the import will work automatically.

## Troubleshooting

### ModuleNotFoundError: No module named 'distutils'

If you encounter this error when importing `yellowbrick`:

```python
ModuleNotFoundError: No module named 'distutils'
```

**Solution:**
1. Ensure you have `setuptools>=68.0.0` installed:
   ```bash
   pip install 'setuptools>=68.0.0'
   ```

2. Reinstall yellowbrick:
   ```bash
   pip install --upgrade --force-reinstall yellowbrick
   ```

3. Verify the installation:
   ```python
   import setuptools
   from yellowbrick.cluster import KElbowVisualizer
   print("✓ Yellowbrick imported successfully!")
   ```

### Alternative: Manual distutils compatibility

If you still encounter issues, you can manually enable distutils compatibility in your notebook/script:

```python
import sys
if sys.version_info >= (3, 12):
    import setuptools
    sys.modules['distutils'] = setuptools._distutils

# Now import yellowbrick
from yellowbrick.cluster import KElbowVisualizer
```

## Verifying Installation

After installation, verify that all dependencies are correctly installed:

```bash
python -c "from yellowbrick.cluster import KElbowVisualizer; print('✓ All imports successful!')"
```

## Virtual Environment (Recommended)

For a clean installation, use a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Jupyter Notebook Setup

If using Jupyter notebooks:

```bash
# Install in the same environment
pip install jupyter

# Create kernel
python -m ipykernel install --user --name=hybrid-ml

# Start Jupyter
jupyter notebook
```

Then select the "hybrid-ml" kernel in your notebook.
