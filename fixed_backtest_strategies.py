import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# Create results directory if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

# List of top 10 tech stocks by market cap
tech_stocks = ['NVDA', 'MSFT', 'AAPL', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AVGO', 'ORCL', 'PLTR']

# Set the time period for backtesting - using the last 3 years
end_date = datetime.now()
start_date = end_date - timedelta(days=3*365)  # 3 years of data

print(f"Backtesting period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

# Download historical data for each stock individually
print("Downloading historical data...")
stock_data = {}
for ticker in tech_stocks:
    try:
        # Download data for single ticker
        df = yf.download(ticker, start=start_date, end=end_date)
        
        # Fix MultiIndex if present in column names
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        
        print(f"Downloaded data for {ticker}: {len(df)} trading days")
        stock_data[ticker] = df
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")

# Calculate technical indicators for each stock
def calculate_indicators(df):
    """Calculate technical indicators mentioned in the paper"""
    # 1. Momentum Oscillators
    # RSI (14-day)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Money Flow Index (MFI)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    
    positive_mf = positive_flow.rolling(window=14).sum()
    negative_mf = negative_flow.rolling(window=14).sum()
    
    mf_ratio = positive_mf / negative_mf
    df['MFI'] = 100 - (100 / (1 + mf_ratio))
    
    # 2. Trend Indicators
    # Moving Averages (50/200-day SMA/EMA)
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # MACD
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # ATR - Average True Range (needed before ADX)
    tr1 = abs(df['High'] - df['Low'])
    tr2 = abs(df['High'] - df['Close'].shift(1))
    tr3 = abs(df['Low'] - df['Close'].shift(1))
    
    df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    # Average Directional Index (ADX) - now TR is already calculated
    high_diff = df['High'].diff()
    low_diff = df['Low'].diff()
    
    pos_directional = high_diff.where(high_diff > 0, 0)
    neg_directional = -low_diff.where(low_diff < 0, 0)
    
    # Simplified ADX
    df['ADX'] = (abs(pos_directional - neg_directional) / (pos_directional + neg_directional + 0.0001)).rolling(window=14).mean() * 100
    
    # 3. Volatility Bands
    # Bollinger Bands
    rolling_mean = df['Close'].rolling(window=20).mean()
    rolling_std = df['Close'].rolling(window=20).std()
    
    df['BB_Middle'] = rolling_mean
    df['BB_Upper'] = rolling_mean + (rolling_std * 2)
    df['BB_Lower'] = rolling_mean - (rolling_std * 2)
    
    # 4. Volume/Accumulation
    # On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # Accumulation/Distribution Index
    clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 0.0001)
    clv = clv.fillna(0)
    df['ADI'] = (clv * df['Volume']).cumsum()
    
    return df

print("Calculating technical indicators...")
for ticker in tech_stocks:
    if ticker in stock_data:
        stock_data[ticker] = calculate_indicators(stock_data[ticker])
        # Drop NaN values resulting from indicator calculations
        stock_data[ticker].dropna(inplace=True)
        print(f"Calculated indicators for {ticker}")

# Implement the strategies from the paper
def implement_strategies(stock_data):
    """Implement the trading strategies recommended in the paper"""
    results = {}
    
    for ticker, df in stock_data.items():
        results[ticker] = {
            'swing_trade_signals': [],
            'breakout_signals': [],
            'dates': []
        }
        
        # Start from the earliest date with complete indicator data
        for i in range(1, len(df)):
            current_date = df.index[i]
            
            # Strategy 1: Swing Trading (1-2 weeks)
            swing_signal = 0  # 0: no signal, 1: buy, -1: sell
            
            # Resistance Reversal (sell signal)
            if df['RSI'].iloc[i-1] > 70 and df['RSI'].iloc[i] < 70 and \
               df['Close'].iloc[i] > df['BB_Upper'].iloc[i] and \
               df['MACD_Hist'].iloc[i] < 0:
                swing_signal = -1  # Sell signal
            
            # Support Bounce (buy signal)
            elif df['RSI'].iloc[i-1] < 30 and df['RSI'].iloc[i] > 30 and \
                 df['Close'].iloc[i] < df['BB_Lower'].iloc[i] and \
                 df['MACD_Hist'].iloc[i] > 0:
                swing_signal = 1  # Buy signal
            
            # Mean-Reversion Setup
            elif df['Close'].iloc[i-1] > df['BB_Upper'].iloc[i-1] and \
                 df['Close'].iloc[i] < df['BB_Upper'].iloc[i] and \
                 df['RSI'].iloc[i] > 60:
                swing_signal = -1  # Sell signal (expecting reversion to mean)
            
            elif df['Close'].iloc[i-1] < df['BB_Lower'].iloc[i-1] and \
                 df['Close'].iloc[i] > df['BB_Lower'].iloc[i] and \
                 df['RSI'].iloc[i] < 40:
                swing_signal = 1  # Buy signal (expecting reversion to mean)
                
            # Strategy 2: Breakout/Trend Signals (1-6 months)
            breakout_signal = 0  # 0: no signal, 1: buy, -1: sell
            
            # Calculate average volume for comparison
            if i >= 50:
                avg_volume = df['Volume'].iloc[i-50:i].mean()
            else:
                avg_volume = df['Volume'].iloc[0:i].mean() if i > 0 else df['Volume'].iloc[0]
                
            # Breakouts on High Volume
            volume_spike = df['Volume'].iloc[i] > 2 * avg_volume
            
            # Bullish breakout pattern
            if df['Close'].iloc[i] > df['SMA50'].iloc[i] and \
               df['SMA50'].iloc[i] > df['SMA200'].iloc[i] and \
               df['Close'].iloc[i-1] <= df['SMA50'].iloc[i-1] and \
               volume_spike and \
               df['RSI'].iloc[i] > 50:
                breakout_signal = 1  # Buy signal - breakout above 50-day MA with volume
                
            # Golden Cross (longer-term bullish)
            elif df['SMA50'].iloc[i-1] <= df['SMA200'].iloc[i-1] and \
                 df['SMA50'].iloc[i] > df['SMA200'].iloc[i] and \
                 df['ADX'].iloc[i] > 25:
                breakout_signal = 1  # Buy signal - 50MA crosses above 200MA
                
            # Death Cross (longer-term bearish)
            elif df['SMA50'].iloc[i-1] >= df['SMA200'].iloc[i-1] and \
                 df['SMA50'].iloc[i] < df['SMA200'].iloc[i] and \
                 df['ADX'].iloc[i] > 25:
                breakout_signal = -1  # Sell signal - 50MA crosses below 200MA
            
            results[ticker]['swing_trade_signals'].append(swing_signal)
            results[ticker]['breakout_signals'].append(breakout_signal)
            results[ticker]['dates'].append(current_date)
    
    return results

print("Implementing trading strategies...")
strategy_results = implement_strategies(stock_data)

# Backtest the strategies
def backtest_strategies(stock_data, strategy_results, initial_capital=10000):
    """Backtest the implemented strategies"""
    backtest_results = {}
    
    for ticker in strategy_results.keys():
        df = stock_data[ticker]
        signals = strategy_results[ticker]
        
        # Initialize backtest variables
        backtest_results[ticker] = {
            'swing_trade': {
                'capital': initial_capital,
                'positions': 0,
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'returns': []
            },
            'breakout': {
                'capital': initial_capital,
                'positions': 0,
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'returns': []
            }
        }
        
        # Convert dates and signals to a DataFrame for easier manipulation
        signal_df = pd.DataFrame({
            'date': signals['dates'],
            'swing_signal': signals['swing_trade_signals'],
            'breakout_signal': signals['breakout_signals']
        })
        
        # Set up backtest dataframe with prices
        backtest_df = pd.DataFrame(index=signal_df['date'])
        backtest_df['close'] = df.loc[signal_df['date']]['Close'].values
        backtest_df['swing_signal'] = signal_df['swing_signal'].values
        backtest_df['breakout_signal'] = signal_df['breakout_signal'].values
        
        # Swing Trade Strategy Backtest (short-term)
        swing = backtest_results[ticker]['swing_trade']
        position_price = 0
        position_type = 0  # 0: none, 1: long, -1: short
        position_day = 0
        position_size = 0
        
        for i in range(1, len(backtest_df)):
            signal = backtest_df['swing_signal'].iloc[i]
            current_price = backtest_df['close'].iloc[i]
            prev_price = backtest_df['close'].iloc[i-1]
            
            # Exit positions after 10 trading days (approx. 2 weeks) or on opposing signal
            if position_type != 0 and (i - position_day >= 10 or (signal != 0 and signal != position_type)):
                swing['trades'] += 1
                
                # Calculate profit/loss based on position type
                if position_type == 1:  # Long position
                    profit = (current_price / position_price - 1) * position_size
                    swing['capital'] += position_size + profit
                else:  # Short position
                    profit = (position_price / current_price - 1) * position_size
                    swing['capital'] += position_size + profit
                
                if profit > 0:
                    swing['wins'] += 1
                else:
                    swing['losses'] += 1
                
                position_type = 0
                swing['positions'] = 0
                
            # Enter new position if we don't have one
            if signal != 0 and position_type == 0:
                # Risk management: Use only 5% of capital per trade
                position_size = swing['capital'] * 0.05
                position_price = current_price
                position_type = signal
                position_day = i
                swing['positions'] = position_size
                swing['capital'] -= position_size
            
            # Record daily capital value
            if position_type == 1:  # Long position
                position_value = (current_price / position_price) * swing['positions']
            elif position_type == -1:  # Short position
                position_value = (position_price / current_price) * swing['positions']
            else:
                position_value = 0
                
            daily_return = swing['capital'] + position_value
            swing['returns'].append(daily_return)
        
        # Breakout Strategy Backtest (longer-term)
        breakout = backtest_results[ticker]['breakout']
        position_price = 0
        position_type = 0  # 0: none, 1: long, -1: short
        position_day = 0
        position_size = 0
        
        for i in range(1, len(backtest_df)):
            signal = backtest_df['breakout_signal'].iloc[i]
            current_price = backtest_df['close'].iloc[i]
            prev_price = backtest_df['close'].iloc[i-1]
            
            # Exit positions after 60 trading days (approx. 3 months) or on opposing signal
            if position_type != 0 and (i - position_day >= 60 or (signal != 0 and signal != position_type)):
                breakout['trades'] += 1
                
                # Calculate profit/loss based on position type
                if position_type == 1:  # Long position
                    profit = (current_price / position_price - 1) * position_size
                    breakout['capital'] += position_size + profit
                else:  # Short position
                    profit = (position_price / current_price - 1) * position_size
                    breakout['capital'] += position_size + profit
                
                if profit > 0:
                    breakout['wins'] += 1
                else:
                    breakout['losses'] += 1
                
                position_type = 0
                breakout['positions'] = 0
                
            # Enter new position if we don't have one
            if signal != 0 and position_type == 0:
                # Risk management: Use only 10% of capital per trade
                position_size = breakout['capital'] * 0.10
                position_price = current_price
                position_type = signal
                position_day = i
                breakout['positions'] = position_size
                breakout['capital'] -= position_size
            
            # Record daily capital value
            if position_type == 1:  # Long position
                position_value = (current_price / position_price) * breakout['positions']
            elif position_type == -1:  # Short position
                position_value = (position_price / current_price) * breakout['positions']
            else:
                position_value = 0
                
            daily_return = breakout['capital'] + position_value
            breakout['returns'].append(daily_return)
    
    return backtest_results

print("Backtesting strategies...")
backtest_results = backtest_strategies(stock_data, strategy_results)

# Calculate performance metrics
def calculate_performance(backtest_results):
    """Calculate performance metrics for backtested strategies"""
    performance = {}
    
    for ticker, results in backtest_results.items():
        performance[ticker] = {}
        
        for strategy, data in results.items():
            if len(data['returns']) == 0:
                continue
                
            initial_capital = 10000
            final_capital = data['returns'][-1]
            total_return = (final_capital - initial_capital) / initial_capital * 100
            
            # Calculate annualized return
            days = len(data['returns'])
            annualized_return = ((1 + total_return / 100) ** (252 / days) - 1) * 100
            
            # Calculate max drawdown
            peak = data['returns'][0]
            max_drawdown = 0
            
            for value in data['returns']:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak * 100
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            # Calculate win rate
            if data['trades'] > 0:
                win_rate = data['wins'] / data['trades'] * 100
            else:
                win_rate = 0
            
            # Calculate Sharpe ratio (simplified, assuming risk-free rate of 0)
            returns_array = np.array(data['returns'])
            daily_returns = np.diff(returns_array) / returns_array[:-1]
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 0 and np.std(daily_returns) > 0 else 0
            
            performance[ticker][strategy] = {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'trades': data['trades'],
                'sharpe_ratio': sharpe_ratio
            }
    
    return performance

print("Calculating performance metrics...")
performance = calculate_performance(backtest_results)

# Print and visualize results
def print_results(performance):
    """Print the performance metrics for all strategies and stocks"""
    print("\n=== BACKTEST RESULTS ===\n")
    
    # Print strategy performance summary
    for strategy in ['swing_trade', 'breakout']:
        print(f"\n--- {strategy.upper()} STRATEGY SUMMARY ---\n")
        
        # Collect results for this strategy across all stocks
        returns = []
        annualized_returns = []
        max_drawdowns = []
        win_rates = []
        sharpe_ratios = []
        
        for ticker, strats in performance.items():
            if strategy in strats:
                returns.append(strats[strategy]['total_return'])
                annualized_returns.append(strats[strategy]['annualized_return'])
                max_drawdowns.append(strats[strategy]['max_drawdown'])
                win_rates.append(strats[strategy]['win_rate'])
                sharpe_ratios.append(strats[strategy]['sharpe_ratio'])
        
        # Calculate averages
        avg_return = sum(returns) / len(returns) if returns else 0
        avg_annual_return = sum(annualized_returns) / len(annualized_returns) if annualized_returns else 0
        avg_max_drawdown = sum(max_drawdowns) / len(max_drawdowns) if max_drawdowns else 0
        avg_win_rate = sum(win_rates) / len(win_rates) if win_rates else 0
        avg_sharpe = sum(sharpe_ratios) / len(sharpe_ratios) if sharpe_ratios else 0
        
        print(f"Average Total Return: {avg_return:.2f}%")
        print(f"Average Annualized Return: {avg_annual_return:.2f}%")
        print(f"Average Max Drawdown: {avg_max_drawdown:.2f}%")
        print(f"Average Win Rate: {avg_win_rate:.2f}%")
        print(f"Average Sharpe Ratio: {avg_sharpe:.2f}")
        
        # Print individual stock results
        print("\nIndividual Stock Performance:")
        print(f"{'Ticker':<6} {'Total Return':<15} {'Annual Return':<15} {'Max Drawdown':<15} {'Win Rate':<15} {'Trades':<8} {'Sharpe':<8}")
        print("-" * 80)
        
        for ticker, strats in performance.items():
            if strategy in strats:
                perf = strats[strategy]
                print(f"{ticker:<6} {perf['total_return']:<15.2f} {perf['annualized_return']:<15.2f} "
                      f"{perf['max_drawdown']:<15.2f} {perf['win_rate']:<15.2f} {perf['trades']:<8} "
                      f"{perf['sharpe_ratio']:<8.2f}")

def visualize_results(backtest_results, performance):
    """Visualize the backtest results"""
    # Create comparison charts
    
    # 1. Compare strategies across stocks
    fig, axes = plt.subplots(2, 1, figsize=(12, 16))
    
    # Total returns by strategy
    tickers = list(performance.keys())
    swing_returns = [performance[ticker]['swing_trade']['total_return'] if 'swing_trade' in performance[ticker] else 0 for ticker in tickers]
    breakout_returns = [performance[ticker]['breakout']['total_return'] if 'breakout' in performance[ticker] else 0 for ticker in tickers]
    
    x = np.arange(len(tickers))
    width = 0.35
    
    axes[0].bar(x - width/2, swing_returns, width, label='Swing Trade')
    axes[0].bar(x + width/2, breakout_returns, width, label='Breakout')
    axes[0].set_ylabel('Total Return (%)')
    axes[0].set_title('Total Returns by Strategy')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tickers)
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Win rates by strategy
    swing_winrates = [performance[ticker]['swing_trade']['win_rate'] if 'swing_trade' in performance[ticker] else 0 for ticker in tickers]
    breakout_winrates = [performance[ticker]['breakout']['win_rate'] if 'breakout' in performance[ticker] else 0 for ticker in tickers]
    
    axes[1].bar(x - width/2, swing_winrates, width, label='Swing Trade')
    axes[1].bar(x + width/2, breakout_winrates, width, label='Breakout')
    axes[1].set_ylabel('Win Rate (%)')
    axes[1].set_title('Win Rates by Strategy')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tickers)
    axes[1].legend()
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('results/strategy_comparison.png')
    
    # 2. Equity curves for top performing stocks
    # Find top 3 stocks for each strategy
    swing_perf = [(ticker, perf['swing_trade']['total_return']) 
                  for ticker, perf in performance.items() 
                  if 'swing_trade' in perf]
    breakout_perf = [(ticker, perf['breakout']['total_return']) 
                     for ticker, perf in performance.items() 
                     if 'breakout' in perf]
    
    swing_top3 = sorted(swing_perf, key=lambda x: x[1], reverse=True)[:3]
    breakout_top3 = sorted(breakout_perf, key=lambda x: x[1], reverse=True)[:3]
    
    # Plot equity curves
    fig, axes = plt.subplots(2, 1, figsize=(12, 16))
    
    for ticker, _ in swing_top3:
        returns = backtest_results[ticker]['swing_trade']['returns']
        axes[0].plot(range(len(returns)), returns, label=ticker)
    
    axes[0].set_title('Equity Curves - Top 3 Swing Trade Performers')
    axes[0].set_xlabel('Trading Days')
    axes[0].set_ylabel('Account Value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    for ticker, _ in breakout_top3:
        returns = backtest_results[ticker]['breakout']['returns']
        axes[1].plot(range(len(returns)), returns, label=ticker)
    
    axes[1].set_title('Equity Curves - Top 3 Breakout Performers')
    axes[1].set_xlabel('Trading Days')
    axes[1].set_ylabel('Account Value')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/equity_curves.png')
    
    # 3. Create a summary visualization
    # Calculate overall averages
    avg_swing_return = sum([perf['swing_trade']['total_return'] for ticker, perf in performance.items() if 'swing_trade' in perf]) / len(swing_perf) if swing_perf else 0
    avg_breakout_return = sum([perf['breakout']['total_return'] for ticker, perf in performance.items() if 'breakout' in perf]) / len(breakout_perf) if breakout_perf else 0
    
    avg_swing_winrate = sum([perf['swing_trade']['win_rate'] for ticker, perf in performance.items() if 'swing_trade' in perf]) / len(swing_perf) if swing_perf else 0
    avg_breakout_winrate = sum([perf['breakout']['win_rate'] for ticker, perf in performance.items() if 'breakout' in perf]) / len(breakout_perf) if breakout_perf else 0
    
    avg_swing_sharpe = sum([perf['swing_trade']['sharpe_ratio'] for ticker, perf in performance.items() if 'swing_trade' in perf]) / len(swing_perf) if swing_perf else 0
    avg_breakout_sharpe = sum([perf['breakout']['sharpe_ratio'] for ticker, perf in performance.items() if 'breakout' in perf]) / len(breakout_perf) if breakout_perf else 0
    
    avg_swing_maxdd = sum([perf['swing_trade']['max_drawdown'] for ticker, perf in performance.items() if 'swing_trade' in perf]) / len(swing_perf) if swing_perf else 0
    avg_breakout_maxdd = sum([perf['breakout']['max_drawdown'] for ticker, perf in performance.items() if 'breakout' in perf]) / len(breakout_perf) if breakout_perf else 0
    
    # Create summary plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    metrics = ['Total Return (%)', 'Win Rate (%)', 'Sharpe Ratio', 'Max Drawdown (%)']
    swing_values = [avg_swing_return, avg_swing_winrate, avg_swing_sharpe, avg_swing_maxdd]
    breakout_values = [avg_breakout_return, avg_breakout_winrate, avg_breakout_sharpe, avg_breakout_maxdd]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax.bar(x - width/2, swing_values, width, label='Swing Trade')
    ax.bar(x + width/2, breakout_values, width, label='Breakout')
    
    ax.set_title('Strategy Performance Summary')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for i, v in enumerate(swing_values):
        ax.text(i - width/2, v + 0.5, f'{v:.2f}', ha='center')
    
    for i, v in enumerate(breakout_values):
        ax.text(i + width/2, v + 0.5, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('results/summary_metrics.png')
    
    print("\nCharts saved to 'results' directory")

print("Printing results...")
print_results(performance)

print("Visualizing results...")
visualize_results(backtest_results, performance)

print("\nBacktesting complete! Results have been saved to the 'results' directory.") 