# Real Estate Project Setup Guide

## Python 3.11.9 Compatibility Setup

This guide will help you set up a fresh virtual environment with Python 3.11.9 and install all the required dependencies for the Real Estate project.

## Prerequisites

- Python 3.11.9 installed on your system
  - Download from: [Python 3.11.9](https://www.python.org/downloads/release/python-3119/)
- Git (if you need to clone the repository)

## Setup Instructions

### 1. Set Up Virtual Environment

We've provided a PowerShell script that automates the setup process. To use it:

```powershell
# Run the setup script
.\setup_venv.ps1
```

This script will:
- Verify Python 3.11.9 is installed
- Create a new virtual environment named `venv-py3119`
- Install all dependencies from the updated `requirements.txt` file

### 2. Manual Setup (Alternative)

If you prefer to set up manually or are not using Windows:

```bash
# Create a virtual environment
python3.11 -m venv venv-py3119

# Activate the virtual environment
# On Windows
.\venv-py3119\Scripts\activate
# On macOS/Linux
source venv-py3119/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Testing the Setup

After setting up the environment, you can verify that all dependencies are working correctly:

```bash
# Activate the virtual environment if not already activated
.\venv-py3119\Scripts\activate  # Windows
# OR
source venv-py3119/bin/activate  # macOS/Linux

# Run the test script
python test_dependencies.py
```

The test script will check:
- Python version
- Core packages (pip, setuptools, fastapi, etc.)
- Data processing libraries (pandas, numpy, etc.)
- Machine learning libraries (scikit-learn, tensorflow, xgboost)
- Visualization libraries (streamlit, plotly, etc.)

## Running the Application

After confirming all dependencies are working correctly, you can run the application:

```bash
# Make sure the virtual environment is activated
.\venv-py3119\Scripts\activate  # Windows
# OR
source venv-py3119/bin/activate  # macOS/Linux

# Run the Streamlit application
streamlit run app.py
```

## Dependency Changes

The following changes were made to ensure compatibility with Python 3.11.9:

- `tensorflow`: Updated to `>=2.15.0,<2.18.0` (from `>=2.17.0`)
- `scikit-learn`: Updated to `>=1.5.2,<1.8.0` (from `>=1.5.2`)
- `xgboost`: Updated to `>=2.1.2,<3.0.0` (from `>=2.1.2`)

These changes ensure that the libraries will install versions compatible with Python 3.11.9 while maintaining stability.

## Troubleshooting

If you encounter any issues during setup or running the application:

1. **Dependency Conflicts**: Try installing dependencies one by one to identify conflicts
2. **TensorFlow Issues**: TensorFlow may require additional setup for GPU support
3. **Import Errors**: Ensure you're using the correct virtual environment

For persistent issues, check the library documentation or open an issue in the project repository.