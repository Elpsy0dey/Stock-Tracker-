import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# List of top 10 tech stocks by market cap
tech_stocks = ['NVDA', 'MSFT', 'AAPL', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AVGO', 'ORCL', 'PLTR']

# Set the time period for backtesting - using the last 3 years
end_date = datetime.now()
start_date = end_date - timedelta(days=3*365)  # 3 years of data

print(f"Debug period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

# Download historical data for each stock individually
print("Downloading sample data...")
stock_data = {}

# Download just one ticker for testing
ticker = tech_stocks[0]  # NVDA
print(f"Downloading data for {ticker}")

# Try method 1: download individual ticker directly
df1 = yf.download(ticker, start=start_date, end=end_date)
print("\nMETHOD 1: Direct download")
print(f"  Shape: {df1.shape}")
print(f"  Type: {type(df1)}")
print(f"  Columns: {df1.columns.tolist()}")
print(f"  Index type: {type(df1.index)}")

# Try comparing Close vs Open directly
print("\nTesting comparison method 1:")
try:
    # Try direct comparison
    comparison = df1['Close'] > df1['Open']
    print(f"  Direct comparison results: {comparison[:5]}")
except Exception as e:
    print(f"  Error in direct comparison: {e}")
    
# Calculate Bollinger Bands
print("\nAdding Bollinger Bands:")
rolling_mean = df1['Close'].rolling(window=20).mean()
rolling_std = df1['Close'].rolling(window=20).std()
df1['BB_Middle'] = rolling_mean
df1['BB_Upper'] = rolling_mean + (rolling_std * 2)
df1['BB_Lower'] = rolling_mean - (rolling_std * 2)

print("Bollinger Bands columns added")
print(f"BB_Middle shape: {df1['BB_Middle'].shape}")
print(f"BB_Upper shape: {df1['BB_Upper'].shape}")

# Try comparison with Bollinger Bands
print("\nTesting Bollinger Bands comparison:")
try:
    i = 50  # Pick a row where we have sufficient data
    print(f"At index {i}:")
    print(f"  Close: {df1['Close'].iloc[i]}")
    print(f"  BB_Upper: {df1['BB_Upper'].iloc[i]}")
    comparison = df1['Close'].iloc[i] > df1['BB_Upper'].iloc[i]
    print(f"  Close > BB_Upper? {comparison}")
except Exception as e:
    print(f"  Error comparing with iloc: {e}")

# Try with .loc and a date
try:
    sample_date = df1.index[50]
    print(f"\nAt date {sample_date}:")
    print(f"  Close: {df1.loc[sample_date, 'Close']}")
    print(f"  BB_Upper: {df1.loc[sample_date, 'BB_Upper']}")
    comparison = df1.loc[sample_date, 'Close'] > df1.loc[sample_date, 'BB_Upper']
    print(f"  Close > BB_Upper? {comparison}")
except Exception as e:
    print(f"  Error comparing with loc: {e}")

# Try getting the values as floats
try:
    print("\nTrying with explicit float conversion:")
    close_val = float(df1['Close'].iloc[i])
    bb_val = float(df1['BB_Upper'].iloc[i])
    print(f"  Close as float: {close_val}")
    print(f"  BB_Upper as float: {bb_val}")
    comparison = close_val > bb_val
    print(f"  Close > BB_Upper? {comparison}")
except Exception as e:
    print(f"  Error converting to float: {e}")
    
# Try with .item() method
try:
    print("\nTrying with .item() method:")
    close_val = df1['Close'].iloc[i].item() if hasattr(df1['Close'].iloc[i], 'item') else df1['Close'].iloc[i]
    bb_val = df1['BB_Upper'].iloc[i].item() if hasattr(df1['BB_Upper'].iloc[i], 'item') else df1['BB_Upper'].iloc[i]
    print(f"  Close with item(): {close_val}")
    print(f"  BB_Upper with item(): {bb_val}")
    comparison = close_val > bb_val
    print(f"  Close > BB_Upper? {comparison}")
except Exception as e:
    print(f"  Error using item() method: {e}")

print("\nDebug complete!") 