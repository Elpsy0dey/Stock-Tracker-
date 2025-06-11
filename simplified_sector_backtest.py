import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys
import traceback

print("Script started...")

# Create results directory if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')
if not os.path.exists('results/sectors'):
    os.makedirs('results/sectors')
print("Created results directories")

# Define stocks from 5 different market segments (1 stock per segment for speed)
market_segments = {
    'Technology': ['AAPL'],
    'Healthcare': ['JNJ'],
    'Financial': ['JPM'],
    'Consumer': ['PG'],
    'Energy': ['XOM']
}

# Flatten the list of stocks
all_stocks = [stock for segment, stocks in market_segments.items() for stock in stocks]

# Map stocks back to their sectors for reporting
stock_to_sector = {}
for sector, stocks in market_segments.items():
    for stock in stocks:
        stock_to_sector[stock] = sector

print(f"Testing {len(all_stocks)} stocks across {len(market_segments)} market segments")

# Set the time period for backtesting - using 1 year instead of 3 for speed
end_date = datetime.now()
start_date = end_date - timedelta(days=365)  # 1 year of data
print(f"Testing period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

# Download historical data for each stock individually
print("Downloading historical data...")
stock_data = {}
for ticker in all_stocks:
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        print(f"Downloaded {ticker} data: {len(df)} days")
        stock_data[ticker] = df
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")

# Calculate a simplified set of technical indicators
def calculate_indicators(df):
    # RSI (14-day)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Moving Averages
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    
    # Bollinger Bands
    rolling_mean = df['Close'].rolling(window=20).mean()
    rolling_std = df['Close'].rolling(window=20).std()
    df['BB_Middle'] = rolling_mean
    df['BB_Upper'] = rolling_mean + (rolling_std * 2)
    df['BB_Lower'] = rolling_mean - (rolling_std * 2)
    
    return df

print("Calculating indicators...")
for ticker in all_stocks:
    if ticker in stock_data:
        stock_data[ticker] = calculate_indicators(stock_data[ticker])
        stock_data[ticker].dropna(inplace=True)
        print(f"Calculated indicators for {ticker}")

# Generate simplified trading signals
def generate_signals(stock_data):
    results = {}
    
    for ticker, df in stock_data.items():
        print(f"Generating signals for {ticker}")
        results[ticker] = {
            'swing_signals': [],
            'breakout_signals': [],
            'dates': []
        }
        
        # Start from earliest date with complete indicator data
        for i in range(1, len(df)):
            current_date = df.index[i]
            
            # Simplified swing trading signal - RSI crosses above 70 or below 30
            swing_signal = 0
            if df['RSI'].iloc[i-1] < 30 and df['RSI'].iloc[i] >= 30:
                swing_signal = 1  # Buy signal
            elif df['RSI'].iloc[i-1] > 70 and df['RSI'].iloc[i] <= 70:
                swing_signal = -1  # Sell signal
            
            # Simplified breakout signal - Price crosses above/below 50-day MA
            breakout_signal = 0
            if df['Close'].iloc[i-1] <= df['SMA50'].iloc[i-1] and df['Close'].iloc[i] > df['SMA50'].iloc[i]:
                breakout_signal = 1  # Buy signal
            elif df['Close'].iloc[i-1] >= df['SMA50'].iloc[i-1] and df['Close'].iloc[i] < df['SMA50'].iloc[i]:
                breakout_signal = -1  # Sell signal
            
            results[ticker]['swing_signals'].append(swing_signal)
            results[ticker]['breakout_signals'].append(breakout_signal)
            results[ticker]['dates'].append(current_date)
        
        # Count number of signals
        swing_count = sum(1 for s in results[ticker]['swing_signals'] if s != 0)
        breakout_count = sum(1 for s in results[ticker]['breakout_signals'] if s != 0)
        print(f"{ticker}: {swing_count} swing signals, {breakout_count} breakout signals")
    
    return results

print("Generating trading signals...")
signals = generate_signals(stock_data)

# Simple backtest function
def run_backtest(stock_data, signals, risk_pct=5):
    results = {}
    
    for ticker in signals.keys():
        print(f"Backtesting {ticker}")
        
        # Extract price data and signals
        df = stock_data[ticker]
        dates = signals[ticker]['dates']
        swing = signals[ticker]['swing_signals']
        breakout = signals[ticker]['breakout_signals']
        
        # Create a dataframe for backtesting
        backtest_df = pd.DataFrame(index=dates)
        backtest_df['price'] = df.loc[dates, 'Close'].values
        backtest_df['swing'] = swing
        backtest_df['breakout'] = breakout
        
        # Initialize results tracking
        results[ticker] = {
            'swing': {
                'trades': 0,
                'wins': 0,
                'return_pct': 0
            },
            'breakout': {
                'trades': 0,
                'wins': 0,
                'return_pct': 0
            }
        }
        
        # Backtest swing strategy
        position = 0  # 0: no position, 1: long, -1: short
        entry_price = 0
        entry_day = 0  # Initialize entry day variable
        equity = 10000  # Starting capital
        
        for i in range(len(backtest_df)):
            signal = backtest_df['swing'].iloc[i]
            price = backtest_df['price'].iloc[i]
            
            # Close position if we have one and get opposing signal or 10 days passed
            if position != 0 and (i - entry_day >= 10 or (signal != 0 and signal != position)):
                results[ticker]['swing']['trades'] += 1
                trade_return = 0
                
                if position == 1:  # Long position
                    trade_return = (price / entry_price - 1) * 100
                else:  # Short position
                    trade_return = (entry_price / price - 1) * 100
                
                if trade_return > 0:
                    results[ticker]['swing']['wins'] += 1
                
                results[ticker]['swing']['return_pct'] += trade_return
                position = 0
            
            # Open new position if we have signal and no current position
            if position == 0 and signal != 0:
                position = signal
                entry_price = price
                entry_day = i
        
        # Repeat for breakout strategy
        position = 0
        entry_price = 0
        entry_day = 0  # Initialize entry day variable for breakout strategy
        
        for i in range(len(backtest_df)):
            signal = backtest_df['breakout'].iloc[i]
            price = backtest_df['price'].iloc[i]
            
            # Close position if we have one and get opposing signal or 60 days passed
            if position != 0 and (i - entry_day >= 60 or (signal != 0 and signal != position)):
                results[ticker]['breakout']['trades'] += 1
                trade_return = 0
                
                if position == 1:  # Long position
                    trade_return = (price / entry_price - 1) * 100
                else:  # Short position
                    trade_return = (entry_price / price - 1) * 100
                
                if trade_return > 0:
                    results[ticker]['breakout']['wins'] += 1
                
                results[ticker]['breakout']['return_pct'] += trade_return
                position = 0
            
            # Open new position if we have signal and no current position
            if position == 0 and signal != 0:
                position = signal
                entry_price = price
                entry_day = i
    
    return results

print("Running backtest...")
try:
    backtest_results = run_backtest(stock_data, signals)

    # Print simple results
    print("\n=== BACKTEST RESULTS ===\n")
    
    # Swing trading results
    print("--- SWING TRADING RESULTS ---")
    print(f"{'Ticker':<6} {'Sector':<10} {'Return %':<10} {'Win %':<10} {'Trades':<6}")
    print("-" * 45)
    
    for ticker, result in backtest_results.items():
        sector = stock_to_sector[ticker]
        trades = result['swing']['trades']
        win_rate = (result['swing']['wins'] / trades * 100) if trades > 0 else 0
        avg_return = result['swing']['return_pct'] / trades if trades > 0 else 0
        
        print(f"{ticker:<6} {sector:<10} {avg_return:<10.2f} {win_rate:<10.2f} {trades:<6}")
    
    # Breakout results
    print("\n--- BREAKOUT RESULTS ---")
    print(f"{'Ticker':<6} {'Sector':<10} {'Return %':<10} {'Win %':<10} {'Trades':<6}")
    print("-" * 45)
    
    for ticker, result in backtest_results.items():
        sector = stock_to_sector[ticker]
        trades = result['breakout']['trades']
        win_rate = (result['breakout']['wins'] / trades * 100) if trades > 0 else 0
        avg_return = result['breakout']['return_pct'] / trades if trades > 0 else 0
        
        print(f"{ticker:<6} {sector:<10} {avg_return:<10.2f} {win_rate:<10.2f} {trades:<6}")
    
    # Create simple bar chart comparing sectors
    plt.figure(figsize=(10, 6))
    
    # Gather sector data
    sector_swing_returns = {}
    sector_breakout_returns = {}
    
    for ticker, result in backtest_results.items():
        sector = stock_to_sector[ticker]
        trades_swing = result['swing']['trades']
        trades_breakout = result['breakout']['trades']
        
        if sector not in sector_swing_returns:
            sector_swing_returns[sector] = []
        if sector not in sector_breakout_returns:
            sector_breakout_returns[sector] = []
        
        if trades_swing > 0:
            sector_swing_returns[sector].append(result['swing']['return_pct'] / trades_swing)
        if trades_breakout > 0:
            sector_breakout_returns[sector].append(result['breakout']['return_pct'] / trades_breakout)
    
    # Calculate sector averages
    sectors = list(market_segments.keys())
    swing_avgs = [np.mean(sector_swing_returns.get(sector, [0])) for sector in sectors]
    breakout_avgs = [np.mean(sector_breakout_returns.get(sector, [0])) for sector in sectors]
    
    # Create plot
    x = np.arange(len(sectors))
    width = 0.35
    
    plt.bar(x - width/2, swing_avgs, width, label='Swing Trade')
    plt.bar(x + width/2, breakout_avgs, width, label='Breakout')
    
    plt.xlabel('Sector')
    plt.ylabel('Average Return per Trade (%)')
    plt.title('Trading Strategy Performance by Sector')
    plt.xticks(x, sectors)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.savefig('results/sectors/sector_comparison.png')
    print("\nPlot saved to results/sectors/sector_comparison.png")

except Exception as e:
    print(f"Error in backtest: {e}")
    traceback.print_exc()

print("\nBacktest completed.") 