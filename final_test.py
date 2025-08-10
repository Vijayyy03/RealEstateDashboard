#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Final test script to verify that all dependencies are working correctly
with Python 3.11.9 after installation.
"""

import sys
import importlib
import importlib.metadata

# Define a function to get package version
def get_package_version(package_name):
    """Get package version using importlib.metadata."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"

# Define a function to print a header
def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {text} ")
    print("=" * 60)

# Define a function to test importing a package
def test_import(package_name, module_name=None):
    """Test importing a package and return success status."""
    if module_name is None:
        module_name = package_name
        
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", get_package_version(package_name))
        print(f"✓ SUCCESS: {package_name} (version: {version})")
        return True, module
    except ImportError as e:
        print(f"✗ FAILED: {package_name}")
        print(f"  Error: {str(e)}")
        return False, None
    except Exception as e:
        print(f"✗ ERROR: {package_name}")
        print(f"  Error: {str(e)}")
        return False, None

# Test Python version
def test_python_version():
    """Test Python version."""
    print_header("PYTHON VERSION")
    print(f"Python {sys.version}")
    
    # Check if Python version is 3.11.x
    major, minor, micro = sys.version_info[:3]
    if (major, minor) == (3, 11):
        print(f"✓ Python version 3.11.{micro} detected")
        return True
    else:
        print(f"✗ Python version mismatch: expected 3.11.x, got {major}.{minor}.{micro}")
        return False

# Test all dependencies
def test_all_dependencies():
    """Test all dependencies."""
    # Test Python version
    python_ok = test_python_version()
    
    # Test core packages
    print_header("CORE PACKAGES")
    fastapi_ok, _ = test_import("fastapi")
    uvicorn_ok, _ = test_import("uvicorn")
    sqlalchemy_ok, _ = test_import("sqlalchemy")
    pydantic_ok, _ = test_import("pydantic")
    dotenv_ok, _ = test_import("python-dotenv", "dotenv")
    
    # Test data processing packages
    print_header("DATA PROCESSING")
    pandas_ok, _ = test_import("pandas")
    numpy_ok, _ = test_import("numpy")
    geopandas_ok, _ = test_import("geopandas")
    
    # Test machine learning packages
    print_header("MACHINE LEARNING")
    sklearn_ok, _ = test_import("scikit-learn", "sklearn")
    xgboost_ok, _ = test_import("xgboost")
    
    # Test TensorFlow with additional info
    tf_ok, tf = test_import("tensorflow")
    if tf_ok:
        try:
            print(f"  TensorFlow is using: {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}")
        except:
            print("  Could not determine if TensorFlow is using GPU or CPU")
    
    joblib_ok, _ = test_import("joblib")
    
    # Test visualization packages
    print_header("VISUALIZATION")
    streamlit_ok, _ = test_import("streamlit")
    plotly_ok, _ = test_import("plotly")
    altair_ok, _ = test_import("altair")
    
    # Print summary
    print_header("SUMMARY")
    all_ok = all([
        python_ok, fastapi_ok, uvicorn_ok, sqlalchemy_ok, pydantic_ok, dotenv_ok,
        pandas_ok, numpy_ok, geopandas_ok, sklearn_ok, xgboost_ok, tf_ok, joblib_ok,
        streamlit_ok, plotly_ok, altair_ok
    ])
    
    if all_ok:
        print("✓ All dependencies are working correctly with Python 3.11.9!")
    else:
        print("✗ Some dependencies are not working correctly with Python 3.11.9.")
        print("  Please check the output above for details.")
    
    return all_ok

# Main function
def main():
    """Run all tests."""
    print_header("DEPENDENCY TEST RESULTS")
    print("Testing dependencies for compatibility with Python 3.11.9...\n")
    
    test_all_dependencies()
    
    print_header("TEST COMPLETE")

# Run the main function
if __name__ == "__main__":
    main()