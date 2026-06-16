import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def prepare_time_series_features(df, lag_days=7):
    """
    Transforms univariate time series data into a supervised learning windowed dataset.
    Uses the previous `lag_days` to predict the next day's price.
    """
    df_features = df.copy()
    
    # Create lag features
    for i in range(1, lag_days + 1):
        df_features[f'Lag_{i}'] = df_features['Value'].shift(i)
        
    df_features = df_features.dropna().reset_index(drop=True)
    
    # Feature matrix (X) and Target vector (y)
    feature_cols = [f'Lag_{i}' for i in range(1, lag_days + 1)]
    X = df_features[feature_cols].values
    y = df_features['Value'].values
    
    return X, y, df_features

def train_forecaster(X, y):
    """
    Trains a Linear Regression model on historical lag features.
    """
    # Split chronologically to simulate realistic forecasting evaluation
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Initialize and fit model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate performance
    predictions = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "R2": r2_score(y_test, predictions)
    }
    
    return model, metrics