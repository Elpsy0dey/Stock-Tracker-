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

# Define stocks from 5 different market segments (4 stocks per segment)
market_segments = {
    'Technology': ['AAPL', 'MSFT', 'NVDA', 'GOOGL'],
    'Healthcare': ['JNJ', 'PFE', 'UNH', 'MRK'],
    'Financial': ['JPM', 'BAC', 'V', 'MA'],
    'Consumer': ['PG', 'WMT', 'COST', 'KO'],
    'Energy': ['XOM', 'CVX', 'COP', 'EOG']
}

# Flatten the list of stocks
all_stocks = [stock for segment, stocks in market_segments.items() for stock in stocks]

# Map stocks back to their sectors for reporting
stock_to_sector = {}
for sector, stocks in market_segments.items():
    for stock in stocks:
        stock_to_sector[stock] = sector

print(f"Testing {len(all_stocks)} stocks across {len(market_segments)} market segments")

# Set the time period for backtesting - using 3 years
end_date = datetime.now()
start_date = end_date - timedelta(days=3*365)
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

# Calculate technical indicators for each stock
def calculate_indicators(df):
    """Calculate technical indicators"""
    print("Calculating indicators...")
    
    # 1. RSI (14-day)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 2. Moving Averages
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # 3. MACD
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # 4. Bollinger Bands
    rolling_mean = df['Close'].rolling(window=20).mean()
    rolling_std = df['Close'].rolling(window=20).std()
    
    df['BB_Middle'] = rolling_mean
    df['BB_Upper'] = rolling_mean + (rolling_std * 2)
    df['BB_Lower'] = rolling_mean - (rolling_std * 2)
    
    # 5. ATR (Average True Range)
    tr1 = abs(df['High'] - df['Low'])
    tr2 = abs(df['High'] - df['Close'].shift(1))
    tr3 = abs(df['Low'] - df['Close'].shift(1))
    
    df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    # 6. Simple ADX (Average Directional Index)
    high_diff = df['High'].diff()
    low_diff = df['Low'].diff()
    
    pos_directional = high_diff.where(high_diff > 0, 0)
    neg_directional = -low_diff.where(low_diff < 0, 0)
    
    df['ADX'] = (abs(pos_directional - neg_directional) / (pos_directional + neg_directional + 0.0001)).rolling(window=14).mean() * 100
    
    return df

print("Calculating technical indicators...")
for ticker in all_stocks:
    if ticker in stock_data:
        print(f"Processing {ticker} indicators...")
        stock_data[ticker] = calculate_indicators(stock_data[ticker])
        # Drop NaN values resulting from indicator calculations
        stock_data[ticker].dropna(inplace=True)
        print(f"Calculated indicators for {ticker}")

# Implement the strategies
def implement_strategies(stock_data):
    """Implement the swing trading and breakout strategies"""
    print("Implementing trading strategies...")
    results = {}
    
    for ticker, df in stock_data.items():
        print(f"Generating signals for {ticker}...")
        results[ticker] = {
            'swing_signals': [],
            'breakout_signals': [],
            'dates': []
        }
        
        # Start from the earliest date with complete indicator data
        signal_count = {'swing': 0, 'breakout': 0}
        
        for i in range(1, len(df)):
            current_date = df.index[i]
            
            # Strategy 1: Swing Trading (1-2 weeks)
            swing_signal = 0  # 0: no signal, 1: buy, -1: sell
            
            # Resistance Reversal (sell signal)
            if df['RSI'].iloc[i-1] > 70 and df['RSI'].iloc[i] < 70 and df['Close'].iloc[i] > df['BB_Upper'].iloc[i]:
                swing_signal = -1  # Sell signal
                signal_count['swing'] += 1
            
            # Support Bounce (buy signal)
            elif df['RSI'].iloc[i-1] < 30 and df['RSI'].iloc[i] > 30 and df['Close'].iloc[i] < df['BB_Lower'].iloc[i]:
                swing_signal = 1  # Buy signal
                signal_count['swing'] += 1
            
            # Mean-Reversion Setup
            elif df['Close'].iloc[i-1] > df['BB_Upper'].iloc[i-1] and df['Close'].iloc[i] < df['BB_Upper'].iloc[i] and df['RSI'].iloc[i] > 60:
                swing_signal = -1  # Sell signal (expecting reversion to mean)
                signal_count['swing'] += 1
            
            elif df['Close'].iloc[i-1] < df['BB_Lower'].iloc[i-1] and df['Close'].iloc[i] > df['BB_Lower'].iloc[i] and df['RSI'].iloc[i] < 40:
                swing_signal = 1  # Buy signal (expecting reversion to mean)
                signal_count['swing'] += 1
                
            # Strategy 2: Breakout/Trend Signals (1-6 months)
            breakout_signal = 0  # 0: no signal, 1: buy, -1: sell
            
            # Calculate average volume for comparison
            if i >= 50:
                avg_volume = df['Volume'].iloc[i-50:i].mean()
            else:
                avg_volume = df['Volume'].iloc[0:i].mean() if i > 0 else df['Volume'].iloc[0]
                
            # Volume spike
            volume_spike = df['Volume'].iloc[i] > 1.5 * avg_volume
            
            # Bullish breakout pattern
            if df['Close'].iloc[i] > df['SMA50'].iloc[i] and df['SMA50'].iloc[i] > df['SMA200'].iloc[i] and df['Close'].iloc[i-1] <= df['SMA50'].iloc[i-1] and volume_spike:
                breakout_signal = 1  # Buy signal - breakout above 50-day MA with volume
                signal_count['breakout'] += 1
                
            # Golden Cross (longer-term bullish)
            elif df['SMA50'].iloc[i-1] <= df['SMA200'].iloc[i-1] and df['SMA50'].iloc[i] > df['SMA200'].iloc[i]:
                breakout_signal = 1  # Buy signal - 50MA crosses above 200MA
                signal_count['breakout'] += 1
                
            # Death Cross (longer-term bearish)
            elif df['SMA50'].iloc[i-1] >= df['SMA200'].iloc[i-1] and df['SMA50'].iloc[i] < df['SMA200'].iloc[i]:
                breakout_signal = -1  # Sell signal - 50MA crosses below 200MA
                signal_count['breakout'] += 1
            
            results[ticker]['swing_signals'].append(swing_signal)
            results[ticker]['breakout_signals'].append(breakout_signal)
            results[ticker]['dates'].append(current_date)
        
        print(f"{ticker}: {signal_count['swing']} swing signals, {signal_count['breakout']} breakout signals")
    
    return results

print("Generating trading signals...")
strategy_results = implement_strategies(stock_data)

# Backtest the strategies
def backtest_strategies(stock_data, signals):
    """Backtest both strategies across all stocks"""
    print("Starting backtest...")
    results = {}
    
    for ticker in signals.keys():
        print(f"Backtesting {ticker}...")
        
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
                'return_pct': 0,
                'max_drawdown': 0,
                'equity_curve': []
            },
            'breakout': {
                'trades': 0,
                'wins': 0,
                'return_pct': 0,
                'max_drawdown': 0,
                'equity_curve': []
            }
        }
        
        # Backtest swing strategy (1-2 weeks horizon)
        position = 0  # 0: no position, 1: long, -1: short
        entry_price = 0
        entry_day = 0
        equity = 10000  # Starting capital
        equity_curve = [equity]
        max_equity = equity
        
        for i in range(1, len(backtest_df)):
            signal = backtest_df['swing'].iloc[i]
            price = backtest_df['price'].iloc[i]
            
            # Exit positions after 10 trading days (approx. 2 weeks) or on opposing signal
            if position != 0 and (i - entry_day >= 10 or (signal != 0 and signal != position)):
                results[ticker]['swing']['trades'] += 1
                trade_return = 0
                
                if position == 1:  # Long position
                    trade_return = (price / entry_price - 1) * 5  # 5% position size (returns x5)
                    equity += trade_return * 200  # 5% of 10000 = 500, so returns apply to 500
                else:  # Short position
                    trade_return = (entry_price / price - 1) * 5  # 5% position size (returns x5)
                    equity += trade_return * 200
                
                # Track win/loss
                if trade_return > 0:
                    results[ticker]['swing']['wins'] += 1
                
                results[ticker]['swing']['return_pct'] += trade_return
                position = 0
                
                # Update equity curve and track max drawdown
                equity_curve.append(equity)
                if equity > max_equity:
                    max_equity = equity
                drawdown = (max_equity - equity) / max_equity * 100
                if drawdown > results[ticker]['swing']['max_drawdown']:
                    results[ticker]['swing']['max_drawdown'] = drawdown
                
            # Enter new position if we don't have one
            elif position == 0 and signal != 0:
                position = signal
                entry_price = price
                entry_day = i
            
            # If no change in position, just copy the last equity value
            else:
                equity_curve.append(equity_curve[-1])
        
        results[ticker]['swing']['equity_curve'] = equity_curve
        
        # Backtest breakout strategy (1-6 month horizon)
        position = 0
        entry_price = 0
        entry_day = 0
        equity = 10000
        equity_curve = [equity]
        max_equity = equity
        
        for i in range(1, len(backtest_df)):
            signal = backtest_df['breakout'].iloc[i]
            price = backtest_df['price'].iloc[i]
            
            # Exit positions after 60 trading days (approx. 3 months) or on opposing signal
            if position != 0 and (i - entry_day >= 60 or (signal != 0 and signal != position)):
                results[ticker]['breakout']['trades'] += 1
                trade_return = 0
                
                if position == 1:
                    trade_return = (price / entry_price - 1) * 10  # 10% position size (returns x10)
                    equity += trade_return * 1000  # 10% of 10000 = 1000
                else:
                    trade_return = (entry_price / price - 1) * 10  # 10% position size (returns x10)
                    equity += trade_return * 1000
                
                # Track win/loss
                if trade_return > 0:
                    results[ticker]['breakout']['wins'] += 1
                
                results[ticker]['breakout']['return_pct'] += trade_return
                position = 0
                
                # Update equity curve and track max drawdown
                equity_curve.append(equity)
                if equity > max_equity:
                    max_equity = equity
                drawdown = (max_equity - equity) / max_equity * 100
                if drawdown > results[ticker]['breakout']['max_drawdown']:
                    results[ticker]['breakout']['max_drawdown'] = drawdown
                
            # Enter new position if we don't have one
            elif position == 0 and signal != 0:
                position = signal
                entry_price = price
                entry_day = i
            
            # If no change in position, just copy the last equity value
            else:
                equity_curve.append(equity_curve[-1])
        
        results[ticker]['breakout']['equity_curve'] = equity_curve
        
        # Calculate final returns
        swing_final_equity = results[ticker]['swing']['equity_curve'][-1]
        breakout_final_equity = results[ticker]['breakout']['equity_curve'][-1]
        
        swing_total_return = (swing_final_equity - 10000) / 10000 * 100
        breakout_total_return = (breakout_final_equity - 10000) / 10000 * 100
        
        print(f"{ticker} Swing: {results[ticker]['swing']['trades']} trades, {swing_total_return:.2f}% return")
        print(f"{ticker} Breakout: {results[ticker]['breakout']['trades']} trades, {breakout_total_return:.2f}% return")
    
    return results

print("Running backtest...")
try:
    backtest_results = backtest_strategies(stock_data, strategy_results)
    print("Backtest completed")
except Exception as e:
    print(f"Error during backtesting: {e}")
    traceback.print_exc()

# Analyze and visualize results
def analyze_results(backtest_results, stock_to_sector):
    """Analyze backtest results and create visualizations"""
    print("Analyzing results...")
    
    # Calculate sector-level metrics
    sector_metrics = {}
    
    for ticker, results in backtest_results.items():
        sector = stock_to_sector[ticker]
        
        if sector not in sector_metrics:
            sector_metrics[sector] = {
                'swing': {
                    'returns': [],
                    'win_rates': [],
                    'trades': 0,
                    'max_drawdowns': []
                },
                'breakout': {
                    'returns': [],
                    'win_rates': [],
                    'trades': 0,
                    'max_drawdowns': []
                }
            }
        
        # Swing metrics
        swing = results['swing']
        if swing['trades'] > 0:
            sector_metrics[sector]['swing']['trades'] += swing['trades']
            sector_metrics[sector]['swing']['win_rates'].append(swing['wins'] / swing['trades'] * 100)
            
            # Calculate total return from equity curve
            total_return = (swing['equity_curve'][-1] - 10000) / 10000 * 100
            sector_metrics[sector]['swing']['returns'].append(total_return)
            sector_metrics[sector]['swing']['max_drawdowns'].append(swing['max_drawdown'])
        
        # Breakout metrics
        breakout = results['breakout']
        if breakout['trades'] > 0:
            sector_metrics[sector]['breakout']['trades'] += breakout['trades']
            sector_metrics[sector]['breakout']['win_rates'].append(breakout['wins'] / breakout['trades'] * 100)
            
            # Calculate total return from equity curve
            total_return = (breakout['equity_curve'][-1] - 10000) / 10000 * 100
            sector_metrics[sector]['breakout']['returns'].append(total_return)
            sector_metrics[sector]['breakout']['max_drawdowns'].append(breakout['max_drawdown'])
    
    # Print results by sector and strategy
    print("\n=== STRATEGY RESULTS BY SECTOR ===\n")
    
    for strategy in ['swing', 'breakout']:
        print(f"\n--- {strategy.upper()} STRATEGY ---\n")
        print(f"{'Sector':<12} {'Avg Return':<12} {'Avg Win Rate':<12} {'Trades':<8} {'Avg Max DD':<10}")
        print("-" * 60)
        
        for sector, metrics in sector_metrics.items():
            strat_metrics = metrics[strategy]
            
            if strat_metrics['trades'] > 0:
                avg_return = sum(strat_metrics['returns']) / len(strat_metrics['returns'])
                avg_win_rate = sum(strat_metrics['win_rates']) / len(strat_metrics['win_rates'])
                avg_max_dd = sum(strat_metrics['max_drawdowns']) / len(strat_metrics['max_drawdowns'])
                
                print(f"{sector:<12} {avg_return:<12.2f} {avg_win_rate:<12.2f} {strat_metrics['trades']:<8} {avg_max_dd:<10.2f}")
    
    # Print top performing stocks for each strategy
    print("\n=== TOP PERFORMING STOCKS ===\n")
    
    # Collect all stock performances
    stock_performances = []
    
    for ticker, results in backtest_results.items():
        swing_return = (results['swing']['equity_curve'][-1] - 10000) / 10000 * 100 if results['swing']['trades'] > 0 else 0
        breakout_return = (results['breakout']['equity_curve'][-1] - 10000) / 10000 * 100 if results['breakout']['trades'] > 0 else 0
        
        stock_performances.append({
            'ticker': ticker,
            'sector': stock_to_sector[ticker],
            'swing_return': swing_return,
            'breakout_return': breakout_return,
            'swing_trades': results['swing']['trades'],
            'breakout_trades': results['breakout']['trades']
        })
    
    # Sort by swing return
    top_swing = sorted(stock_performances, key=lambda x: x['swing_return'], reverse=True)[:5]
    
    print("Top 5 Stocks - Swing Trading:")
    print(f"{'Rank':<5} {'Ticker':<6} {'Sector':<12} {'Return %':<10} {'Trades':<8}")
    print("-" * 45)
    
    for i, stock in enumerate(top_swing):
        print(f"{i+1:<5} {stock['ticker']:<6} {stock['sector']:<12} {stock['swing_return']:<10.2f} {stock['swing_trades']:<8}")
    
    # Sort by breakout return
    top_breakout = sorted(stock_performances, key=lambda x: x['breakout_return'], reverse=True)[:5]
    
    print("\nTop 5 Stocks - Breakout Strategy:")
    print(f"{'Rank':<5} {'Ticker':<6} {'Sector':<12} {'Return %':<10} {'Trades':<8}")
    print("-" * 45)
    
    for i, stock in enumerate(top_breakout):
        print(f"{i+1:<5} {stock['ticker']:<6} {stock['sector']:<12} {stock['breakout_return']:<10.2f} {stock['breakout_trades']:<8}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # 1. Sector comparison chart
    plt.figure(figsize=(12, 8))
    
    sectors = list(sector_metrics.keys())
    x = np.arange(len(sectors))
    width = 0.35
    
    # Calculate average returns by sector
    swing_avgs = []
    breakout_avgs = []
    
    for sector in sectors:
        if sector_metrics[sector]['swing']['returns']:
            swing_avgs.append(np.mean(sector_metrics[sector]['swing']['returns']))
        else:
            swing_avgs.append(0)
            
        if sector_metrics[sector]['breakout']['returns']:
            breakout_avgs.append(np.mean(sector_metrics[sector]['breakout']['returns']))
        else:
            breakout_avgs.append(0)
    
    # Create the bar chart
    plt.bar(x - width/2, swing_avgs, width, label='Swing Trading')
    plt.bar(x + width/2, breakout_avgs, width, label='Breakout')
    
    plt.title('Average Returns by Market Sector', fontsize=16)
    plt.xlabel('Sector', fontsize=14)
    plt.ylabel('Average Return (%)', fontsize=14)
    plt.xticks(x, sectors)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for i, v in enumerate(swing_avgs):
        plt.text(i - width/2, v + 1, f"{v:.1f}%", ha='center', fontsize=10)
        
    for i, v in enumerate(breakout_avgs):
        plt.text(i + width/2, v + 1, f"{v:.1f}%", ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/sectors/sector_comparison.png')
    
    # 2. Top stocks chart
    plt.figure(figsize=(14, 10))
    
    # Combine top performers from both strategies, removing duplicates
    all_top_tickers = list(set([s['ticker'] for s in top_swing + top_breakout]))
    all_top_data = {}
    
    for ticker in all_top_tickers:
        for stock in stock_performances:
            if stock['ticker'] == ticker:
                all_top_data[ticker] = {
                    'swing': stock['swing_return'],
                    'breakout': stock['breakout_return'],
                    'sector': stock['sector']
                }
                break
    
    # Sort by combined performance
    sorted_tickers = sorted(all_top_data.keys(), 
                          key=lambda x: all_top_data[x]['swing'] + all_top_data[x]['breakout'], 
                          reverse=True)
    
    # Prepare data for chart
    swing_returns = [all_top_data[t]['swing'] for t in sorted_tickers]
    breakout_returns = [all_top_data[t]['breakout'] for t in sorted_tickers]
    labels = [f"{t} ({all_top_data[t]['sector']})" for t in sorted_tickers]
    
    x = np.arange(len(sorted_tickers))
    width = 0.35
    
    plt.bar(x - width/2, swing_returns, width, label='Swing Trading')
    plt.bar(x + width/2, breakout_returns, width, label='Breakout')
    
    plt.title('Top Performing Stocks Across Strategies', fontsize=16)
    plt.xlabel('Stock (Sector)', fontsize=14)
    plt.ylabel('Return (%)', fontsize=14)
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('results/sectors/top_stocks.png')
    
    # 3. Create a chart showing win rates by sector
    plt.figure(figsize=(12, 8))
    
    # Calculate average win rates by sector
    swing_winrates = []
    breakout_winrates = []
    
    for sector in sectors:
        if sector_metrics[sector]['swing']['win_rates']:
            swing_winrates.append(np.mean(sector_metrics[sector]['swing']['win_rates']))
        else:
            swing_winrates.append(0)
            
        if sector_metrics[sector]['breakout']['win_rates']:
            breakout_winrates.append(np.mean(sector_metrics[sector]['breakout']['win_rates']))
        else:
            breakout_winrates.append(0)
    
    # Create the bar chart
    plt.bar(x - width/2, swing_winrates, width, label='Swing Trading')
    plt.bar(x + width/2, breakout_winrates, width, label='Breakout')
    
    plt.title('Average Win Rates by Market Sector', fontsize=16)
    plt.xlabel('Sector', fontsize=14)
    plt.ylabel('Win Rate (%)', fontsize=14)
    plt.xticks(x, sectors)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for i, v in enumerate(swing_winrates):
        plt.text(i - width/2, v + 1, f"{v:.1f}%", ha='center', fontsize=10)
        
    for i, v in enumerate(breakout_winrates):
        plt.text(i + width/2, v + 1, f"{v:.1f}%", ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/sectors/sector_winrates.png')
    
    print("Results analysis complete. Charts saved to 'results/sectors/' directory.")

print("Analyzing results...")
try:
    analyze_results(backtest_results, stock_to_sector)
except Exception as e:
    print(f"Error analyzing results: {e}")
    traceback.print_exc()

print("\nMulti-sector backtesting completed!") 