import streamlit as st
import pandas as pd
import numpy as np

# Set page title
st.set_page_config(page_title="Simple Streamlit Test", layout="wide")

# Title and introduction
st.title("Simple Streamlit Test")
st.markdown("This is a simple test application to verify that Streamlit is working correctly with Python 3.11.9.")

# Display Python version
st.header("Python Version")
import sys
st.code(f"Python {sys.version}")

# Display dependency versions
st.header("Dependency Versions")
st.write(f"Streamlit: {st.__version__}")
st.write(f"Pandas: {pd.__version__}")
st.write(f"NumPy: {np.__version__}")

# Create a sample dataframe
st.header("Sample Data")
data = pd.DataFrame({
    'x': np.random.normal(0, 1, 10),
    'y': np.random.normal(0, 1, 10)
})
st.dataframe(data)

# Create a simple chart
st.header("Simple Chart")
st.line_chart(data)

# Final success message
st.success("Streamlit is working correctly with Python 3.11.9!")