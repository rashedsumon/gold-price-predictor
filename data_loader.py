import os
import glob
import pandas as pd
import kagglehub

def load_gold_data():
    """
    Downloads the latest version of the gold price dataset via kagglehub,
    locates the CSV file, and returns it as a parsed pandas DataFrame.
    """
    # Download dataset files using kagglehub
    path = kagglehub.dataset_download("arashnic/learn-time-series-forecasting-from-gold-price")
    
    # Find the CSV file inside the downloaded path
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in the downloaded dataset path: {path}")
    
    # Read the first available CSV file
    df = pd.read_csv(csv_files[0])
    
    # Data Cleaning and Normalization
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df = df.dropna()
    
    return df

if __name__ == "__main__":
    # Test execution
    print("Testing Kaggle data download...")
    data = load_gold_data()
    print(f"Success! Loaded dataset with shape {data.shape}")
    print(data.head())