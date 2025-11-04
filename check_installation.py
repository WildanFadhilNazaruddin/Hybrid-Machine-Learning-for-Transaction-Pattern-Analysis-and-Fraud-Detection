#!/usr/bin/env python3
"""
Installation Checker for Hybrid ML Project

This script verifies that all required dependencies are installed and compatible
with the current Python version.
"""

import sys
from typing import List, Tuple

def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is compatible."""
    version = sys.version_info
    if version < (3, 8):
        return False, f"❌ Python 3.8+ required, found {version.major}.{version.minor}.{version.micro}"
    return True, f"✓ Python {version.major}.{version.minor}.{version.micro}"

def check_module(module_name: str, import_path: str = None) -> Tuple[bool, str]:
    """Check if a module can be imported."""
    import importlib
    try:
        if import_path:
            # Use importlib for safe dynamic imports
            module = importlib.import_module(import_path)
            getattr(module, module_name.split('.')[-1])
        else:
            importlib.import_module(module_name)
        return True, f"✓ {module_name}"
    except ImportError as e:
        return False, f"❌ {module_name}: {str(e)}"
    except Exception as e:
        return False, f"❌ {module_name}: Unexpected error - {str(e)}"

def check_distutils_compat() -> Tuple[bool, str]:
    """Check if distutils compatibility is available for Python 3.12+."""
    if sys.version_info >= (3, 12):
        try:
            import setuptools
            # Handle potential changes in setuptools API
            if hasattr(setuptools, '_distutils'):
                sys.modules['distutils'] = setuptools._distutils
            else:
                # Fallback: try direct import
                import distutils
                sys.modules['distutils'] = distutils
            from distutils.version import LooseVersion
            return True, "✓ distutils compatibility (via setuptools) for Python 3.12+"
        except (ImportError, AttributeError) as e:
            return False, f"❌ distutils compatibility: {str(e)}"
    else:
        try:
            from distutils.version import LooseVersion
            return True, "✓ distutils (native)"
        except ImportError:
            return False, "❌ distutils not available"

def main():
    """Run all checks."""
    print("=" * 70)
    print("Installation Checker for Hybrid ML Project")
    print("=" * 70)
    print()
    
    checks: List[Tuple[bool, str]] = []
    
    # Check Python version
    checks.append(check_python_version())
    
    # Core dependencies
    print("Checking core dependencies:")
    checks.append(check_module("numpy", "numpy"))
    checks.append(check_module("pandas", "pandas"))
    checks.append(check_module("sklearn", "sklearn"))
    checks.append(check_module("matplotlib", "matplotlib"))
    checks.append(check_module("seaborn", "seaborn"))
    checks.append(check_module("joblib", "joblib"))
    
    # Check distutils compatibility
    print("\nChecking distutils compatibility:")
    checks.append(check_distutils_compat())
    
    # Check yellowbrick specifically
    print("\nChecking yellowbrick:")
    checks.append(check_module("yellowbrick", "yellowbrick"))
    
    # Try importing the specific class that was failing
    print("\nChecking specific imports from yellowbrick:")
    checks.append(check_module("KElbowVisualizer", "yellowbrick.cluster"))
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for success, message in checks:
        print(message)
    
    # Summary
    failed = [msg for success, msg in checks if not success]
    if failed:
        print("\n" + "=" * 70)
        print(f"FAILED: {len(failed)} check(s) failed")
        print("=" * 70)
        print("\nTo fix, run:")
        print("  pip install -r requirements.txt")
        print("\nFor Python 3.12+ distutils issues, ensure setuptools>=68.0.0:")
        print("  pip install 'setuptools>=68.0.0'")
        sys.exit(1)
    else:
        print("\n" + "=" * 70)
        print("SUCCESS: All checks passed! ✓")
        print("=" * 70)
        sys.exit(0)

if __name__ == "__main__":
    main()
