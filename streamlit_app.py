import streamlit as tf
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from data_loader import load_gold_data
from model import prepare_time_series_features, train_forecaster

# Configure page layout
st.set_page_config(page_title="Gold Price Forecasting AI", layout="wide")

st.title("💰 Gold Price Forecasting AI")
st.write("This app automatically extracts 50 years of data from Kaggle and builds a supervised ML forecaster.")

# 1. Load Data
with st.spinner("Downloading and processing data from Kaggle..."):
    try:
        df = load_gold_data()
        
    except Exception as e:
        st.error(f"Error initializing data: {e}")
        st.stop()

# Layout Sidebars and Filters
st.sidebar.header("Model Parameters")
lag_days = st.sidebar.slider("Historical Memory Lag (Days)", min_value=3, max_value=30, value=7)

# 2. Extract Features and Train Model
X, y, df_features = prepare_time_series_features(df, lag_days=lag_days)
model, metrics = train_forecaster(X, y)






# Grab the most recent row values to feed as prediction features
latest_lags = df['Value'].iloc[-lag_days:].values[::-1].reshape(1, -1)
predicted_price = model.predict(latest_lags)[0]

# Display Result Card
st.info(f"### **Predicted Gold Price:** `${predicted_price:,.2f}`")

# Raw Data Inspect tools
if st.checkbox("Show Raw Dataset Snippet"):
    st.dataframe(df.tail(20))