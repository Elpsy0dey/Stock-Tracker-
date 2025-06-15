"""
Data utilities for the Trading Portfolio Tracker
"""

import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import requests
import re
import io
from config.settings import *

def load_trades_from_file(uploaded_file) -> Tuple[bool, pd.DataFrame, str]:
    """
    Load trades from uploaded CSV or Excel file
    
    Returns:
        Tuple of (success, dataframe, error_message)
    """
    try:
        # Determine file type and read accordingly
        if uploaded_file.name.lower().endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Standardize column names
        column_mapping = {
            'Trade Date': 'trade_date',
            'Settlement Date': 'settlement_date',
            'Symbol': 'symbol',
            'Side': 'side',
            'Trade Identifier': 'trade_id',
            'Units': 'units',
            'Avg. Price': 'avg_price',
            'Value': 'value',
            'Fees': 'fees',
            'GST': 'gst',
            'Total Value': 'total_value',
            'Currency': 'currency',
            'AUD/USD rate': 'aud_usd_rate'
        }
        
        # Rename columns if they exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # Check for required columns
        required_columns = ['trade_date', 'symbol', 'side', 'units', 'avg_price']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            error_msg = f"Missing required columns: {', '.join(missing_columns)}"
            return False, pd.DataFrame(), error_msg
        
        # Convert dates
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        if 'settlement_date' in df.columns:
            df['settlement_date'] = pd.to_datetime(df['settlement_date'])
        
        # Clean symbol names (extract ticker from full name)
        df['ticker'] = df['symbol'].str.split(' - ').str[0]
        
        # Handle numeric columns
        numeric_columns = ['units', 'avg_price', 'value', 'fees', 'total_value']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce')
        
        # Fill NaN values
        df['fees'] = df['fees'].fillna(0)
        if 'gst' in df.columns:
            df['gst'] = df['gst'].fillna(0)
        
        return True, df, ""
        
    except Exception as e:
        return False, pd.DataFrame(), f"Error loading file: {str(e)}"

def get_stock_data(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch stock data from Yahoo Finance
    
    Args:
        symbol: Stock ticker symbol
        period: Data period ('1y', '2y', '5y', '10y', 'ytd', 'max')
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
    
    Returns:
        DataFrame with OHLCV data
    """
    try:
        # Map custom periods to Yahoo Finance periods
        period_mapping = {
            '1y': '1y',
            '2y': '2y',
            '5y': '5y'
        }
        
        yf_period = period_mapping.get(period, period)
        
        stock = yf.Ticker(symbol)
        hist = stock.history(period=yf_period, interval=interval)
        
        if hist.empty:
            return pd.DataFrame()
        
        # No custom filtering needed - using standard Yahoo Finance periods
        
        # Ensure we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in hist.columns:
                return pd.DataFrame()
        
        return hist
    
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()

def get_current_prices(symbols: List[str]) -> Dict[str, float]:
    """
    Get current market prices for multiple symbols
    
    Args:
        symbols: List of stock ticker symbols
    
    Returns:
        Dictionary of symbol -> current price
    """
    prices = {}
    
    for symbol in symbols:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1d")
            
            if not hist.empty:
                prices[symbol] = float(hist['Close'].iloc[-1])
            else:
                # Try to get info as fallback
                info = stock.info
                if 'regularMarketPrice' in info:
                    prices[symbol] = float(info['regularMarketPrice'])
                else:
                    prices[symbol] = 0.0
        except:
            prices[symbol] = 0.0
    
    return prices

def get_stock_info(symbol: str) -> Dict:
    """
    Get comprehensive stock information
    
    Args:
        symbol: Stock ticker symbol
    
    Returns:
        Dictionary with stock information
    """
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # Clean and standardize the info
        cleaned_info = {
            'symbol': symbol,
            'company_name': info.get('longName', symbol),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'market_cap': info.get('marketCap', 0),
            'current_price': info.get('regularMarketPrice', 0),
            'previous_close': info.get('previousClose', 0),
            'volume': info.get('volume', 0),
            'avg_volume': info.get('averageVolume', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'forward_pe': info.get('forwardPE', 0),
            'price_to_book': info.get('priceToBook', 0),
            'dividend_yield': info.get('dividendYield', 0),
            'beta': info.get('beta', 0),
            '52_week_high': info.get('fiftyTwoWeekHigh', 0),
            '52_week_low': info.get('fiftyTwoWeekLow', 0),
            'moving_avg_50': info.get('fiftyDayAverage', 0),
            'moving_avg_200': info.get('twoHundredDayAverage', 0)
        }
        
        return cleaned_info
    
    except Exception as e:
        return {'symbol': symbol, 'error': str(e)}

def validate_stock_symbol(symbol: str) -> bool:
    """
    Validate if a stock symbol exists and has data
    
    Args:
        symbol: Stock ticker symbol
    
    Returns:
        True if valid, False otherwise
    """
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="5d")
        return not hist.empty
    except:
        return False

def calculate_returns(prices: pd.Series, periods: List[int] = [1, 5, 10, 20]) -> Dict[str, float]:
    """
    Calculate returns over different periods
    
    Args:
        prices: Series of price data
        periods: List of periods to calculate returns for
    
    Returns:
        Dictionary of period -> return percentage
    """
    returns = {}
    
    for period in periods:
        if len(prices) > period:
            current_price = prices.iloc[-1]
            past_price = prices.iloc[-(period + 1)]
            return_pct = ((current_price - past_price) / past_price) * 100
            returns[f'{period}d_return'] = return_pct
        else:
            returns[f'{period}d_return'] = 0.0
    
    return returns

def clean_numeric_column(series: pd.Series) -> pd.Series:
    """
    Clean numeric column by removing currency symbols and converting to float
    
    Args:
        series: Pandas series to clean
    
    Returns:
        Cleaned numeric series
    """
    return pd.to_numeric(
        series.astype(str).str.replace('$', '').str.replace(',', ''), 
        errors='coerce'
    )

def format_currency(value: float, decimals: int = 2) -> str:
    """
    Format value as currency
    
    Args:
        value: Numeric value
        decimals: Number of decimal places
    
    Returns:
        Formatted currency string
    """
    return f"${value:,.{decimals}f}"

def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format value as percentage
    
    Args:
        value: Numeric value (as decimal, e.g., 0.05 for 5%)
        decimals: Number of decimal places
    
    Returns:
        Formatted percentage string
    """
    return f"{value:.{decimals}f}%"

def get_trading_dates(start_date: datetime, end_date: datetime) -> List[datetime]:
    """
    Get list of trading dates (excluding weekends)
    
    Args:
        start_date: Start date
        end_date: End date
    
    Returns:
        List of trading dates
    """
    dates = []
    current_date = start_date
    
    while current_date <= end_date:
        # Skip weekends (Monday = 0, Sunday = 6)
        if current_date.weekday() < 5:
            dates.append(current_date)
        current_date += timedelta(days=1)
    
    return dates

def calculate_portfolio_metrics(returns: pd.Series) -> Dict[str, float]:
    """
    Calculate comprehensive portfolio metrics
    
    Args:
        returns: Series of portfolio returns
    
    Returns:
        Dictionary of metrics
    """
    if returns.empty or len(returns) < 2:
        return {}
    
    # Basic statistics
    total_return = (1 + returns).prod() - 1
    avg_return = returns.mean()
    volatility = returns.std()
    
    # Risk-adjusted metrics
    sharpe_ratio = avg_return / volatility if volatility > 0 else 0
    
    # Drawdown analysis
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Win rate
    positive_returns = returns[returns > 0]
    win_rate = len(positive_returns) / len(returns) * 100
    
    return {
        'total_return': total_return * 100,
        'avg_return': avg_return * 100,
        'volatility': volatility * 100,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown * 100,
        'win_rate': win_rate
    }

def resample_data(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Resample OHLCV data to different frequency
    
    Args:
        df: DataFrame with OHLCV data
        freq: Frequency string ('D', 'W', 'M', etc.)
    
    Returns:
        Resampled DataFrame
    """
    if df.empty:
        return df
    
    ohlc_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    
    # Only resample columns that exist
    resample_dict = {}
    for col, method in ohlc_dict.items():
        if col in df.columns:
            resample_dict[col] = method
    
    if not resample_dict:
        return df
    
    resampled = df.resample(freq).agg(resample_dict)
    return resampled.dropna()

def get_sp500_tickers() -> List[str]:
    """
    Get list of S&P 500 ticker symbols
    
    Returns:
        List of ticker symbols
    """
    try:
        # This is a simplified list - in production, you might want to fetch from a reliable source
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        sp500_table = tables[0]
        return sp500_table['Symbol'].tolist()
    except:
        # Fallback to a sample of major S&P 500 stocks
        return [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'NVDA', 'META', 'BRK-B',
            'UNH', 'JNJ', 'JPM', 'V', 'PG', 'HD', 'MA', 'PFE', 'BAC', 'ABBV',
            'KO', 'AVGO', 'PEP', 'TMO', 'COST', 'WMT', 'DIS', 'ABT', 'VZ',
            'ADBE', 'NFLX', 'CRM', 'XOM', 'NKE', 'DHR', 'LIN', 'BMY', 'ORCL'
        ]

def load_data_from_google_sheet(sheet_url: str, sheet_tab: str = None) -> Tuple[bool, pd.DataFrame, str]:
    """
    Load data from a Google Sheet URL
    
    Args:
        sheet_url: URL of the Google Sheet (must be publicly accessible or shared with anyone with the link)
        sheet_tab: Optional name of the specific sheet tab to load
        
    Returns:
        Tuple of (success, dataframe, error_message)
    """
    try:
        # Extract the sheet ID from the URL
        sheet_id_match = re.search(r'/spreadsheets/d/([a-zA-Z0-9-_]+)', sheet_url)
        if not sheet_id_match:
            return False, pd.DataFrame(), "Invalid Google Sheets URL format"
        
        sheet_id = sheet_id_match.group(1)
        
        # Create the export URL (CSV format)
        export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        
        # Add sheet tab parameter if specified
        if sheet_tab:
            export_url += f"&gid={sheet_tab}"
        
        # Make a request to check if the sheet is accessible
        response = requests.get(export_url)
        if response.status_code != 200:
            if response.status_code == 404:
                return False, pd.DataFrame(), "Google Sheet not found. Make sure the URL is correct."
            elif response.status_code == 403:
                return False, pd.DataFrame(), "Access denied. Make sure the Google Sheet is publicly accessible or shared with anyone with the link."
            else:
                return False, pd.DataFrame(), f"Error accessing Google Sheet (HTTP {response.status_code})"
        
        # Read the CSV data from the response content
        df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
        
        if df.empty:
            return False, pd.DataFrame(), "The Google Sheet appears to be empty"
        
        # Standardize column names
        column_mapping = {
            'Trade Date': 'trade_date',
            'Settlement Date': 'settlement_date',
            'Symbol': 'symbol',
            'Side': 'side',
            'Trade Identifier': 'trade_id',
            'Units': 'units',
            'Avg. Price': 'avg_price',
            'Value': 'value',
            'Fees': 'fees',
            'GST': 'gst',
            'Total Value': 'total_value',
            'Currency': 'currency',
            'AUD/USD rate': 'aud_usd_rate'
        }
        
        # Rename columns if they exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # Check for required columns
        required_columns = ['trade_date', 'symbol', 'side', 'units', 'avg_price']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            error_msg = f"Missing required columns: {', '.join(missing_columns)}"
            return False, pd.DataFrame(), error_msg
        
        # Convert dates
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        if 'settlement_date' in df.columns:
            df['settlement_date'] = pd.to_datetime(df['settlement_date'])
        
        # Clean symbol names (extract ticker from full name)
        df['ticker'] = df['symbol'].str.split(' - ').str[0]
        
        # Handle numeric columns
        numeric_columns = ['units', 'avg_price', 'value', 'fees', 'total_value']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce')
        
        # Fill NaN values
        df['fees'] = df['fees'].fillna(0)
        if 'gst' in df.columns:
            df['gst'] = df['gst'].fillna(0)
        
        return True, df, ""
        
    except Exception as e:
        return False, pd.DataFrame(), f"Error loading Google Sheet: {str(e)}" 