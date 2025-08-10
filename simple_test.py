#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple test script to verify that key dependencies are working correctly
with Python 3.11.9 after installation.
"""

import sys
import os


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {text} ")
    print("=" * 60)


def test_python_version():
    """Test Python version."""
    print_header("PYTHON VERSION")
    print(f"Python {sys.version}")
    
    # Check if Python version is 3.11.9
    major, minor, micro = sys.version_info[:3]
    if (major, minor) == (3, 11):
        print(f"✓ Python version 3.11.{micro} detected")
    else:
        print(f"✗ Python version mismatch: expected 3.11.x, got {major}.{minor}.{micro}")


def test_import(package_name, module_name=None):
    """Test importing a package and return success status."""
    if module_name is None:
        module_name = package_name
        
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"✓ SUCCESS: {package_name} (version: {version})")
        return True
    except ImportError as e:
        print(f"✗ FAILED: {package_name}")
        print(f"  Error: {str(e)}")
        return False
    except Exception as e:
        print(f"✗ ERROR: {package_name}")
        print(f"  Error: {str(e)}")
        return False


def test_core_packages():
    """Test core packages."""
    print_header("CORE PACKAGES")
    
    # Test key packages without importing setuptools directly
    test_import("fastapi")
    test_import("uvicorn")
    test_import("sqlalchemy")
    test_import("pydantic")
    test_import("python-dotenv", "dotenv")


def test_data_processing():
    """Test data processing packages."""
    print_header("DATA PROCESSING")
    
    test_import("pandas")
    test_import("numpy")
    test_import("geopandas")


def test_ml_packages():
    """Test machine learning packages."""
    print_header("MACHINE LEARNING")
    
    test_import("sklearn")
    test_import("xgboost")
    
    # Test TensorFlow with additional info
    if test_import("tensorflow"):
        import tensorflow as tf
        print(f"  TensorFlow is using: {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}")
    
    test_import("joblib")


def test_visualization():
    """Test visualization packages."""
    print_header("VISUALIZATION")
    
    test_import("streamlit")
    test_import("plotly")
    test_import("altair")


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