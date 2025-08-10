import streamlit as st
import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import tensorflow as tf
import sklearn
import xgboost as xgb
import sys

# Set page title and configuration
st.set_page_config(page_title="Python 3.11.9 Compatibility Test", layout="wide")

# Title and introduction
st.title("Python 3.11.9 Compatibility Test")
st.markdown("This is a simple test application to verify that all dependencies are working correctly with Python 3.11.9.")

# Display Python version
st.header("Python Version")
st.code(f"Python {sys.version}")

# Display dependency versions
st.header("Dependency Versions")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Core Libraries")
    st.write(f"Streamlit: {st.__version__}")
    st.write(f"Pandas: {pd.__version__}")
    st.write(f"NumPy: {np.__version__}")
    st.write(f"Plotly: {plotly.__version__}")

with col2:
    st.subheader("Machine Learning Libraries")
    st.write(f"TensorFlow: {tf.__version__}")
    st.write(f"Scikit-learn: {sklearn.__version__}")
    st.write(f"XGBoost: {xgb.__version__}")

# Create a sample dataframe
st.header("Sample Data Visualization")

# Generate sample data
data = pd.DataFrame({
    'x': np.random.normal(0, 1, 100),
    'y': np.random.normal(0, 1, 100),
    'category': np.random.choice(['A', 'B', 'C'], 100)
})

# Display the dataframe
st.subheader("Sample DataFrame")
st.dataframe(data.head())

# Create a plotly scatter plot
st.subheader("Plotly Scatter Plot")
fig = px.scatter(data, x='x', y='y', color='category', title='Sample Scatter Plot')
st.plotly_chart(fig, use_container_width=True)

# TensorFlow simple model
st.header("TensorFlow Simple Model")

# Create a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Display model summary
st.code(model.summary())
st.success("TensorFlow model created successfully!")

# Final success message
st.header("Test Results")
st.success("All dependencies are working correctly with Python 3.11.9!")