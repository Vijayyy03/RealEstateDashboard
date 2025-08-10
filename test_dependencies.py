#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify that all dependencies are working correctly
with Python 3.11.9 after installation.
"""

import sys
import importlib
import importlib.metadata

# Use importlib.metadata instead of pkg_resources to avoid setuptools issues
def get_package_version(package_name):
    """Get package version using importlib.metadata."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {text} ")
    print("=" * 60)


def print_status(package, status, version=None):
    """Print the status of a package import attempt."""
    if status:
        status_text = f"✓ SUCCESS: {package}"
        if version:
            status_text += f" (version: {version})"
        print(status_text)
    else:
        print(f"✗ FAILED: {package}")


def test_python_version():
    """Test Python version."""
    print_header("PYTHON VERSION")
    print(f"Python {sys.version}")
    
    # Check if Python version is 3.11.9
    major, minor, micro = sys.version_info[:3]
    if (major, minor, micro) == (3, 11, 9):
        print("✓ Python version 3.11.9 confirmed")
    else:
        print(f"✗ Python version mismatch: expected 3.11.9, got {major}.{minor}.{micro}")


def test_core_packages():
    """Test core packages."""
    print_header("CORE PACKAGES")
    
    packages = [
        "pip", "setuptools", "wheel", "python-dotenv", "pydantic", 
        "fastapi", "uvicorn", "sqlalchemy"
    ]
    
    for package in packages:
        try:
            # Handle package names with hyphens
            import_name = package.replace("-", "_")
            if import_name == "python_dotenv":
                import_name = "dotenv"
                
            # Import the package
            module = importlib.import_module(import_name)
            
            # Get version
            version = getattr(module, "__version__", get_package_version(package))
                
            print_status(package, True, version)
        except ImportError as e:
            print_status(package, False)
            print(f"  Error: {str(e)}")


def test_data_processing():
    """Test data processing packages."""
    print_header("DATA PROCESSING")
    
    packages = ["pandas", "numpy", "geopandas"]
    
    for package in packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", get_package_version(package))
            print_status(package, True, version)
        except ImportError as e:
            print_status(package, False)
            print(f"  Error: {str(e)}")


def test_ml_packages():
    """Test machine learning packages."""
    print_header("MACHINE LEARNING")
    
    packages = [
        ("scikit-learn", "sklearn"),
        ("xgboost", "xgboost"),
        ("tensorflow", "tensorflow"),
        ("joblib", "joblib")
    ]
    
    for pkg_name, import_name in packages:
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, "__version__", get_package_version(pkg_name))
            print_status(pkg_name, True, version)
            
            # Additional test for TensorFlow
            if import_name == "tensorflow":
                import tensorflow as tf
                print(f"  TensorFlow is using: {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}")
                
        except ImportError as e:
            print_status(pkg_name, False)
            print(f"  Error: {str(e)}")
        except Exception as e:
            print_status(pkg_name, False)
            print(f"  Error: {str(e)}")


def test_visualization():
    """Test visualization packages."""
    print_header("VISUALIZATION")
    
    packages = [
        ("streamlit", "streamlit"),
        ("plotly", "plotly"),
        ("altair", "altair")
    ]
    
    for pkg_name, import_name in packages:
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, "__version__", get_package_version(pkg_name))
            print_status(pkg_name, True, version)
        except ImportError as e:
            print_status(pkg_name, False)
            print(f"  Error: {str(e)}")


def main():
    """Run all tests."""
    print_header("DEPENDENCY TEST RESULTS")
    print("Testing dependencies for compatibility with Python 3.11.9...\n")
    
    test_python_version()
    test_core_packages()
    test_data_processing()
    test_ml_packages()
    test_visualization()
    
    print_header("TEST COMPLETE")


if __name__ == "__main__":
    main()