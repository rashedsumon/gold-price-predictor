import streamlit as tf
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from data_loader import load_gold_data
from model import prepare_time_series_features, train_forecaster

# Configure page layout
st.set_page_config(page_title="Gold Price Forecasting AI", layout="wide")

st.title("💰 Gold Price Time-Series Forecasting Dashboard")
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

# 3. Main Interface KPI Cards
kpi1, kpi2, kpi3 = st.columns(3)
with kpi1:
    st.metric("Latest Historic Price", f"${df['Value'].iloc[-1]:,.2f}", f"As of {df['Date'].iloc[-1].strftime('%Y-%m-%d')}")
with kpi2:
    st.metric("Model Prediction Error (MAE)", f"${metrics['MAE']:.2f}")
with kpi3:
    st.metric("Model Confidence ($R^2$)", f"{metrics['R2']*100:.1f}%")

# 4. Interactive Plotly Visualization
st.subheader("Historical Timeline of Gold Prices")
fig = px.line(df, x='Date', y='Value', labels={'Value': 'Price (USD)', 'Date': 'Year'}, title="Gold Price Trend (1970 - 2020)")
st.plotly_chart(fig, use_container_width=True)

# 5. Live Prediction Interface
st.markdown("---")
st.subheader("🔮 Run Live AI Forecast")
st.write(f"Based on the last {lag_days} days of recorded data, here is the predicted market open price for the next trading cycle:")

# Grab the most recent row values to feed as prediction features
latest_lags = df['Value'].iloc[-lag_days:].values[::-1].reshape(1, -1)
predicted_price = model.predict(latest_lags)[0]

# Display Result Card
st.info(f"### **Predicted Gold Price:** `${predicted_price:,.2f}`")

# Raw Data Inspect tools
if st.checkbox("Show Raw Dataset Snippet"):
    st.dataframe(df.tail(20))