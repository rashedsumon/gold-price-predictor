import streamlit as st
import numpy as np
import pandas as pd
from data_loader import load_gold_data
from model import prepare_time_series_features, train_forecaster

# Configure page layout
st.set_page_config(page_title="Gold Price Predictor", layout="centered")

st.title("💰 Gold Price Prediction AI")
st.write("Adjust your parameters below and trigger the AI model to calculate the next market price prediction.")

# Automatically handle data pipeline setup behind the scenes
@st.cache_data(show_spinner=False)
def get_cached_data():
    return load_gold_data()

try:
    df = get_cached_data()
except Exception as e:
    st.error(f"Error fetching data from Kaggle: {e}")
    st.stop()

# ==========================================
# SECTION 1: MODEL PARAMETERS
# ==========================================
st.header("1. Model Parameters")
st.markdown("Configure how many historical days of data the machine learning model should look back on to find patterns.")

# Slider to adjust the lag feature configuration
lag_days = st.slider(
    label="Historical Memory Window (Days to look back)", 
    min_value=3, 
    max_value=30, 
    value=7,
    help="The number of past consecutive days used as features to predict tomorrow's value."
)

# Extract features and train model dynamically based on slider choice
X, y, _ = prepare_time_series_features(df, lag_days=lag_days)
model, metrics = train_forecaster(X, y)

st.markdown("---")

# ==========================================
# SECTION 2: PREDICTED GOLD PRICE
# ==========================================
st.header("2. Predicted Gold Price")
st.markdown(f"Press the button below to process the last **{lag_days} days** of historical gold entries through the trained model.")

# Action Button to trigger prediction output
if st.button("🔮 Predict Next Price", type="primary"):
    with st.spinner("Running AI matrix calculations..."):
        # Slice the most recent rows to form the predictor matrix (reverse to match lag order)
        latest_historical_lags = df['Value'].iloc[-lag_days:].values[::-1].reshape(1, -1)
        
        # Pass the input array to our scikit-learn model
        predicted_price = model.predict(latest_historical_lags)[0]
        
    # Display results to the user
    st.success("Analysis Complete!")
    
    # Layout result alongside model performance confidence metrics
    col1, col2 = st.columns([2, 1])
    with col1:
        st.metric(
            label="Predicted Gold Open Price (USD)", 
            value=f"${predicted_price:,.2f}"
        )
    with col2:
        st.metric(
            label="Model Confidence ($R^2$)", 
            value=f"{metrics['R2']*100:.1f}%"
        )
        
    st.info(f"💡 *Note: This prediction is generated using data up to the latest dataset milestone ({df['Date'].iloc[-1].strftime('%Y-%m-%d')}).*")
else:
    st.info("👈 Click the button above to view the calculation output.")