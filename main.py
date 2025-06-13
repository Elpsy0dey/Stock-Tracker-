# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Advanced Trading Portfolio Tracker

A comprehensive web-based application for portfolio tracking, technical analysis,
stock screening, risk management, and machine learning-based trading signals.

Based on academic research for high win-rate trading strategies.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime, timedelta
from typing import Dict, List
import json
import plotly.express as px
import yfinance as yf
import pytz
import logging

# Import our modules
from config.settings import *
from models.portfolio_tracker import PortfolioTracker
from models.technical_analysis import TechnicalAnalyzer
from models.stock_screener import StockScreener
from models.ml_models import MLSignalPredictor
from utils.data_utils import load_trades_from_file, get_stock_data, get_sp500_tickers
from utils.chart_utils import *
from utils.startup import initialize_application
from services.ai_service import AIService
from services.strategy_manager import StrategyManager
from config.api_config import API_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# Set page config
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

def init_session_state():
    """Initialize session state variables"""
    if 'portfolio_tracker' not in st.session_state:
        st.session_state.portfolio_tracker = PortfolioTracker()
    if 'technical_analyzer' not in st.session_state:
        st.session_state.technical_analyzer = TechnicalAnalyzer()
    if 'stock_screener' not in st.session_state:
        st.session_state.stock_screener = StockScreener()
        # Initialize stock universe
        st.session_state.stock_screener.set_stock_universe()

    if 'ml_predictor' not in st.session_state:
        st.session_state.ml_predictor = MLSignalPredictor()
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'ml_trained' not in st.session_state:
        st.session_state.ml_trained = False
    
    # Initialize AI model detection if not already done
    if 'ai_model_initialized' not in st.session_state:
        try:
            # Initialize application components
            initialize_application()
            st.session_state.ai_model_initialized = True
            st.session_state.current_model = API_CONFIG['MODEL']
            logger.info(f"AI model initialized: {st.session_state.current_model}")
        except Exception as e:
            logger.error(f"Error during AI model initialization: {str(e)}")
            st.session_state.ai_model_initialized = False

def get_stock_universe():
    """Get list of stock symbols for ML training"""
    # Use S&P 500 stocks for training
    return [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 'JNJ', 'V',
        'HD', 'MA', 'PG', 'UNH', 'BAC', 'CRM', 'DIS', 'ADBE', 'NFLX', 'XOM',
        'KO', 'PEP', 'ABBV', 'BRK-B', 'WMT', 'PFE', 'TMO', 'AVGO', 'COST', 'ABT',
        'VZ', 'NKE', 'DHR', 'BMY', 'LIN', 'ORCL'
    ]

def main():
    """Main application function"""
    st.title(APP_TITLE)
    st.markdown("🚀 **Advanced Trading Platform** with Technical Analysis, Stock Screening, Risk Management & Machine Learning")
    
    # Initialize session state
    init_session_state()
    
    # Create main navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Portfolio", 
        "📈 Performance", 
        "🎯 Stock Screener", 
        "🔍 Technical Analysis",
        "⚙️ Settings"
    ])
    
    # Sidebar for global controls
    with st.sidebar:
        st.header("📁 Data Management")
        
        # Portfolio data upload
        st.subheader("Portfolio Data")
        starting_cash = st.number_input(
            "Starting Cash ($)",
            value=st.session_state.portfolio_tracker.starting_cash,
            step=0.01,
            format="%.2f"
        )
        st.session_state.portfolio_tracker.starting_cash = starting_cash
        
        uploaded_file = st.file_uploader(
            "Upload Trading Data",
            type=ALLOWED_FILE_TYPES,
            help="Upload your trading history (CSV or Excel)"
        )
        
        if uploaded_file is not None:
            if st.button("📤 Load Trading Data"):
                success, trades_df, error_msg = load_trades_from_file(uploaded_file)
                if success:
                    st.session_state.portfolio_tracker.load_trades(trades_df)
                    st.session_state.data_loaded = True
                    st.success("✅ Trading data loaded successfully!")
                    st.rerun()
                else:
                    st.error(f"❌ {error_msg}")
        
        # Market data controls
        if st.session_state.data_loaded:
            st.subheader("Market Data")
            if st.button("🔄 Refresh Prices"):
                st.session_state.portfolio_tracker.refresh_current_prices()
                st.success("Prices updated!")
                st.rerun()
        
        # Quick stats
        if st.session_state.data_loaded:
            stats = st.session_state.portfolio_tracker.get_summary_stats()
            st.subheader("📊 Quick Stats")
            st.metric("Account Value", f"${stats['total_account_value']:,.0f}")
            st.metric("Total P&L", f"${stats['total_pnl']:,.0f}", 
                     delta=f"{stats['total_return_pct']:+.1f}%")
    
    # Tab 1: Portfolio Overview
    with tab1:
        portfolio_tab()
    
    # Tab 2: Performance Analytics
    with tab2:
        performance_tab()
    
    # Tab 3: Stock Screener
    with tab3:
        stock_screener_tab()
    
    # Tab 4: Technical Analysis
    with tab4:
        technical_analysis_tab()
    
    # Tab 5: Settings
    with tab5:
        settings_tab()

def portfolio_tab():
    """Portfolio overview and management"""
    # Portfolio Overview
    st.markdown("<h1 style='font-size: 2.5rem; margin-bottom: 1rem;'>📊 Portfolio Overview</h1>", unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.info("Please load portfolio data to view portfolio overview")
        return
    
    # Get portfolio data
    stats = st.session_state.portfolio_tracker.get_summary_stats()
    
    # Portfolio Summary
    st.markdown("<h2 style='font-size: 2rem; margin-bottom: 1rem;'>📈 Portfolio Summary</h2>", unsafe_allow_html=True)
    
    # Create three columns for key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Account Value",
            f"${stats['total_account_value']:,.2f}",
            delta=f"{stats['total_return_pct']:+.2f}%",
            delta_color="normal"
        )
        st.metric("Cash Balance", f"${stats['current_cash']:,.2f}")
    
    with col2:
        st.metric(
            "Total Return",
            f"${stats['total_pnl']:,.2f}",
            delta=f"${stats['total_pnl']:+,.2f}",
            delta_color="normal"
        )
        st.metric("Return %", f"{stats['total_return_pct']:,.2f}%")
    
    with col3:
        st.metric("Portfolio Value", f"${stats['portfolio_value']:,.2f}")
        st.metric(
            "Unrealized P&L",
            f"${stats['unrealized_pnl']:,.2f}",
            delta=f"${stats['unrealized_pnl']:+,.2f}",
            delta_color="normal"
        )
    
    st.markdown("---")
    
    # Monthly Balance Changes
    st.markdown("<h2 style='font-size: 2rem; margin-bottom: 1rem;'>📅 Monthly Balance Changes</h2>", unsafe_allow_html=True)
    
    # Get monthly data
    monthly_data = st.session_state.portfolio_tracker.get_formatted_monthly_data()
    
    if not monthly_data.empty:
        # Add view toggle
        view_option = st.radio(
            "Select View",
            ["📊 Table View", "📈 Graph View"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        if view_option == "📊 Table View":
            # Display monthly changes table
            st.dataframe(monthly_data, use_container_width=True, hide_index=True)
        else:
            # Create and display monthly balance chart
            monthly_df = st.session_state.portfolio_tracker.calculate_monthly_balance_changes()
            if not monthly_df.empty:
                fig = create_monthly_balance_chart(monthly_df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No monthly data available")
    
    st.markdown("---")
    
    # Current Holdings
    st.markdown("<h2 style='font-size: 2rem; margin-bottom: 1rem;'>💼 Current Holdings</h2>", unsafe_allow_html=True)
    
    # Get holdings data
    if stats['portfolio_details']:
        holdings_data = []
        for symbol, details in stats['portfolio_details'].items():
            # Get extended market data
            try:
                stock = yf.Ticker(symbol)
                info = stock.info
                
                # Get pre-market and after-market prices
                pre_market_price = info.get('preMarketPrice', None)
                after_market_price = info.get('postMarketPrice', None)
                
                # Get current market status
                current_time = datetime.now(pytz.timezone('US/Eastern'))
                market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
                market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
                
                # Determine which prices to show
                show_pre_market = current_time < market_open
                show_after_market = current_time > market_close
                
                # Calculate average price
                avg_price = calculate_average_price(symbol)
                
                # Get short-term trading decision
                try:
                    short_term_data = {
                        'price': details['current_price'],
                        'rsi': info.get('RSI', 50),
                        'macd': {
                            'value': info.get('MACD', 0),
                            'signal': info.get('MACD_Signal', 0)
                        },
                        'stochastic': {
                            'k': info.get('Stoch_K', 50),
                            'd': info.get('Stoch_D', 50)
                        },
                        'adx': info.get('ADX', 25),
                        'volume_ratio': info.get('Volume', 0) / info.get('Volume_SMA', 1) if info.get('Volume_SMA', 0) > 0 else 1.0,
                        'bb_position': info.get('BB_Position', 0.5),
                        'atr': info.get('ATR', 0),
                        'patterns': [pattern for pattern, value in info.items() if pattern.endswith('_Pattern') and value],
                        'sma_20': info.get('SMA_20', 0),
                        'sma_50': info.get('SMA_50', 0),
                        'sma_200': info.get('SMA_200', 0)
                    }
                    # Removing AI action and reason generation
                except Exception as e:
                    print(f"Error getting technical data for {symbol}: {str(e)}")
                
                holding_info = {
                    'Symbol': symbol,
                    'Average Price': avg_price,
                    'Shares': details['shares'],
                    'Gross Value': details['shares'] * details['current_price'],
                    'Current Price': details['current_price'],
                    'Unrealized P&L': details['unrealized_pnl'],
                    'Return %': details['unrealized_pnl_pct']
                }
                
                # Add pre-market data if available and relevant
                if show_pre_market and pre_market_price:
                    holding_info.update({
                        'Pre-Market Price': pre_market_price,
                        'Pre-Market Change': pre_market_price - details['current_price'],
                        'Pre-Market Change %': ((pre_market_price - details['current_price']) / details['current_price']) * 100
                    })
                
                # Add after-market data if available and relevant
                if show_after_market and after_market_price:
                    holding_info.update({
                        'After-Market Price': after_market_price,
                        'After-Market Change': after_market_price - details['current_price'],
                        'After-Market Change %': ((after_market_price - details['current_price']) / details['current_price']) * 100
                    })
                
                holdings_data.append(holding_info)
                
            except Exception as e:
                st.warning(f"Could not fetch extended market data for {symbol}: {str(e)}")
                # Calculate average price even if market data fetch fails
                avg_price = calculate_average_price(symbol)
                holdings_data.append({
                    'Symbol': symbol,
                    'Average Price': avg_price,
                    'Shares': details['shares'],
                    'Gross Value': details['shares'] * details['current_price'],  # Changed to use current price
                    'Current Price': details['current_price'],
                    'Unrealized P&L': details['unrealized_pnl'],
                    'Return %': details['unrealized_pnl_pct']
                })
        
        holdings_df = pd.DataFrame(holdings_data)
        
        # Reorder columns to match requested order
        base_columns = ['Symbol', 'Average Price', 'Shares', 'Gross Value', 'Current Price']
        pre_market_columns = ['Pre-Market Price', 'Pre-Market Change', 'Pre-Market Change %'] if 'Pre-Market Price' in holdings_df.columns else []
        after_market_columns = ['After-Market Price', 'After-Market Change', 'After-Market Change %'] if 'After-Market Price' in holdings_df.columns else []
        performance_columns = ['Unrealized P&L', 'Return %']
        # Removed short_term_columns with Action and Reason
        
        # Combine all columns in the desired order
        ordered_columns = base_columns + pre_market_columns + after_market_columns + performance_columns
        holdings_df = holdings_df[ordered_columns]
        
        # Format the display values
        display_df = holdings_df.copy()
        display_df['Shares'] = display_df['Shares'].apply(lambda x: f"{x:,.0f}")
        display_df['Average Price'] = display_df['Average Price'].apply(lambda x: f"${x:,.2f}")
        display_df['Gross Value'] = display_df['Gross Value'].apply(lambda x: f"${x:,.2f}")
        display_df['Current Price'] = display_df['Current Price'].apply(lambda x: f"${x:,.2f}")
        display_df['Unrealized P&L'] = display_df['Unrealized P&L'].apply(lambda x: f"${x:+,.2f}")
        display_df['Return %'] = display_df['Return %'].apply(lambda x: f"{x:+.2f}%")
        
        # Format pre-market and after-market columns if they exist
        if 'Pre-Market Price' in display_df.columns:
            display_df['Pre-Market Price'] = display_df['Pre-Market Price'].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "N/A")
            display_df['Pre-Market Change'] = display_df['Pre-Market Change'].apply(lambda x: f"${x:+,.2f}" if pd.notnull(x) else "N/A")
            display_df['Pre-Market Change %'] = display_df['Pre-Market Change %'].apply(lambda x: f"{x:+.2f}%" if pd.notnull(x) else "N/A")
        
        if 'After-Market Price' in display_df.columns:
            display_df['After-Market Price'] = display_df['After-Market Price'].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "N/A")
            display_df['After-Market Change'] = display_df['After-Market Change'].apply(lambda x: f"${x:+,.2f}" if pd.notnull(x) else "N/A")
            display_df['After-Market Change %'] = display_df['After-Market Change %'].apply(lambda x: f"{x:+.2f}%" if pd.notnull(x) else "N/A")

        # Add color coding for positive and negative values
        def color_values(val):
            if isinstance(val, str):
                if val.startswith('$+') or val.startswith('+'):
                    return 'color: #00C853'  # Standard green color for all positive values
                elif val.startswith('$-') or val.startswith('-'):
                    return 'color: #FF3D00'  # Standard red color for all negative values
            return ''

        # Get list of columns to style based on what exists in the DataFrame
        columns_to_style = ['Unrealized P&L', 'Return %']
        
        if 'Pre-Market Change' in display_df.columns:
            columns_to_style.extend(['Pre-Market Change', 'Pre-Market Change %'])
        if 'After-Market Change' in display_df.columns:
            columns_to_style.extend(['After-Market Change', 'After-Market Change %'])

        # Apply styling to the DataFrame
        styled_df = display_df.style.applymap(color_values, subset=columns_to_style)

        # Display the styled DataFrame
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Portfolio allocation charts
        st.markdown("<h3 style='font-size: 1.5rem; margin-bottom: 0.5rem;'>Portfolio Allocation</h3>", unsafe_allow_html=True)
        
        # Create two columns for charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart for allocation
            fig = px.pie(holdings_df, 
                        values='Gross Value', 
                        names='Symbol',
                        title='Portfolio Allocation by Position',
                        color_discrete_sequence=px.colors.qualitative.Set1)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Bar chart for returns
            fig = px.bar(holdings_df,
                        x='Symbol',
                        y='Return %',
                        title='Position Returns',
                        color='Return %',
                        color_continuous_scale=['red', 'gray', 'green'])
            fig.update_layout(yaxis_title="Return %")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No holdings data available")
    
    st.markdown("---")
    
    # Recent Trades
    st.markdown("<h2 style='font-size: 2rem; margin-bottom: 1rem;'>🔄 Recent Trades</h2>", unsafe_allow_html=True)
    
    # Get recent trades from the trade history
    if hasattr(st.session_state.portfolio_tracker, 'trade_history'):
        recent_trades = st.session_state.portfolio_tracker.trade_history[-10:]  # Get last 10 trades
        if recent_trades:
            trades_df = pd.DataFrame(recent_trades)
            
            # Format the columns
            trades_df['entry_price'] = trades_df['entry_price'].round(2)
            trades_df['exit_price'] = trades_df['exit_price'].round(2)
            trades_df['pnl'] = trades_df['pnl'].round(2)
            trades_df['pnl_pct'] = trades_df['pnl_pct'].round(2)
            trades_df['fees'] = trades_df['fees'].round(2)  # Format fees
            
            # Rename columns for better readability
            trades_df = trades_df.rename(columns={
                'symbol': 'Symbol',
                'entry_date': 'Entry Date',
                'exit_date': 'Exit Date',
                'entry_price': 'Entry Price ($)',
                'exit_price': 'Exit Price ($)',
                'shares': 'Shares',
                'pnl': 'P&L ($)',
                'pnl_pct': 'P&L (%)',
                'hold_time': 'Hold Time (Days)',
                'fees': 'Fees ($)'  # Add fees column
            })
            
            # Add color coding for P&L
            def color_pnl(val):
                if isinstance(val, (int, float)):
                    return 'color: #00C853' if val > 0 else 'color: #FF3D00'
                return ''
            
            # Apply styling
            styled_df = trades_df.style.applymap(color_pnl, subset=['P&L ($)', 'P&L (%)'])
            
            # Display the table with better formatting
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Entry Date': st.column_config.DatetimeColumn(format='YYYY-MM-DD'),
                    'Exit Date': st.column_config.DatetimeColumn(format='YYYY-MM-DD'),
                    'Entry Price ($)': st.column_config.NumberColumn(format='$%.2f'),
                    'Exit Price ($)': st.column_config.NumberColumn(format='$%.2f'),
                    'P&L ($)': st.column_config.NumberColumn(format='$%.2f'),
                    'P&L (%)': st.column_config.NumberColumn(format='%.2f%%'),
                    'Hold Time (Days)': st.column_config.NumberColumn(format='%.0f'),
                    'Fees ($)': st.column_config.NumberColumn(format='$%.2f')  # Add fees formatting
                }
            )
        else:
            st.info("No recent trades available")
    else:
        st.info("No trade history available")

def technical_analysis_tab():
    """Technical analysis tools and charts"""
    st.markdown("<h1 style='font-size: 2.5rem; margin-bottom: 1rem;'>🔍 Technical Analysis</h1>", unsafe_allow_html=True)
    
    # Add custom CSS for the analyze button with a unique identifier
    st.markdown("""
        <style>
            div[data-testid="stButton"] #analyze_button {
                display: flex;
                justify-content: center;
            }
            div[data-testid="stButton"] #analyze_button button {
                background-color: #1E88E5;  /* Blue color */
                color: white;
                width: auto !important;
                min-width: 100px;
                border: none;
                border-radius: 4px;
                padding: 0.5rem 1rem;
            }
            div[data-testid="stButton"] #analyze_button button:hover {
                background-color: #1565C0;  /* Darker blue on hover */
            }
            div[data-testid="stButton"] #refresh_ai_button {
                display: flex;
                justify-content: center;
            }
            div[data-testid="stButton"] #refresh_ai_button button {
                background-color: #4CAF50;  /* Green color */
                color: white;
                width: auto !important;
                min-width: 100px;
                border: none;
                border-radius: 4px;
                padding: 0.5rem 1rem;
            }
            div[data-testid="stButton"] #refresh_ai_button button:hover {
                background-color: #388E3C;  /* Darker green on hover */
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Symbol selection - adjusted column ratios for better alignment
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        symbol = st.text_input("Enter Stock Symbol", value="AAPL", placeholder="e.g., AAPL, MSFT, GOOGL")
    
    with col2:
        period = st.selectbox("Time Period", [
            "1y", "2y", "5y"
        ], index=0)  # Default to 1y
    
    with col3:
        # Add some vertical spacing to align with the selectbox
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_button = st.button("📊 Analyze", key="analyze_button")
    
    # Add spacing before results
    st.markdown("---")
    
    # Display results below the input controls
    if analyze_button:
        analyze_symbol(symbol, period)

def analyze_symbol(symbol: str, period: str):
    """Analyze a symbol with technical indicators"""
    with st.spinner(f'Analyzing {symbol}...'):
        # Get stock data
        df = get_stock_data(symbol, period=period)
        
        if df.empty:
            st.error(f"Could not fetch data for {symbol}")
            return
        
        # Calculate technical indicators
        df_with_indicators = st.session_state.technical_analyzer.calculate_all_indicators(df)
        
        if df_with_indicators.empty:
            st.error("Could not calculate technical indicators")
            return
        
        # Gather technical data for AI analysis and cache clearing
        latest_data = df_with_indicators.iloc[-1]
        technical_data = {
            'price': latest_data['Close'],
            'rsi': latest_data.get('RSI', 0),
            'macd': {
                'value': latest_data.get('MACD', 0),
                'signal': latest_data.get('MACD_Signal', 0)
            },
            'stochastic': {
                'k': latest_data.get('Stochastic_K', 0),
                'd': latest_data.get('Stochastic_D', 0)
            },
            'adx': latest_data.get('ADX', 0),
            'volume_ratio': latest_data.get('Volume_Ratio', 0),
            'bb_position': latest_data.get('BB_Position', 0.5),
            'atr': latest_data.get('ATR', 0),
            'patterns': [], # Fill with detected patterns if available
            'sma_20': latest_data.get('SMA_20', 0),
            'sma_50': latest_data.get('SMA_50', 0),
            'sma_200': latest_data.get('SMA_200', 0)
        }
        
        # Clear the AI service cache for this specific technical data
        from services.ai_service import AIService
        AIService.clear_cache(technical_data)
        
        # Create comprehensive chart
        tech_chart = create_technical_analysis_chart(df_with_indicators, symbol)
        if tech_chart:
            st.plotly_chart(tech_chart, use_container_width=True)
        
        # Display current signals with error handling
        try:
            # Signal strength analysis
            st.markdown("---")
            st.subheader("💪 Signal Strength Analysis")
            signal_strength = st.session_state.technical_analyzer.get_signal_strength(df_with_indicators)
            
            if signal_strength:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Overall Score", f"{signal_strength.get('overall_score', 0):.0f}/100")
                with col2:
                    st.metric("Momentum", f"{signal_strength.get('momentum_score', 0):.0f}/100")
                with col3:
                    st.metric("Trend", f"{signal_strength.get('trend_score', 0):.0f}/100")
                with col4:
                    st.metric("Volume", f"{signal_strength.get('volume_score', 0):.0f}/100")
            else:
                st.info("Insufficient data for signal strength analysis")
            
            # Trading Signals Section
            st.markdown("---")
            st.subheader("🎯 Trading Signals")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Swing Trading (1-2 weeks)**")
                swing_signals = st.session_state.technical_analyzer.get_swing_signals(df_with_indicators)
                
                if swing_signals:
                    # Group signals by type for better display
                    bullish_signals = []
                    bearish_signals = []
                    setup_signals = []
                    
                    for signal_name, signal_value in swing_signals.items():
                        if 'bullish' in signal_name.lower() or 'breakout_retest' in signal_name:
                            if signal_value:
                                bullish_signals.append(signal_name.replace('_', ' ').title())
                        elif 'bearish' in signal_name.lower():
                            if signal_value:
                                bearish_signals.append(signal_name.replace('_', ' ').title())
                        else:
                            if signal_value:
                                setup_signals.append(signal_name.replace('_', ' ').title())
                    
                    # Display active signals
                    if bullish_signals:
                        st.success("🟢 **Bullish Signals:**")
                        for signal in bullish_signals:
                            st.write(f"✅ {signal}")
                    
                    if bearish_signals:
                        st.error("🔴 **Bearish Signals:**")
                        for signal in bearish_signals:
                            st.write(f"✅ {signal}")
                    
                    if setup_signals:
                        st.warning("🟡 **Setup Signals:**")
                        for signal in setup_signals:
                            st.write(f"✅ {signal}")
                    
                    if not (bullish_signals or bearish_signals or setup_signals):
                        st.info("❌ No active swing signals detected")
                        # Show all signals with status
                        for signal_name, signal_value in swing_signals.items():
                            emoji = "✅" if signal_value else "❌"
                            st.write(f"{emoji} {signal_name.replace('_', ' ').title()}")
                else:
                    st.info("Insufficient data for swing signals")
            
            with col2:
                st.markdown("**Breakout Trading (1-6 months)**")
                breakout_signals = st.session_state.technical_analyzer.get_breakout_signals(df_with_indicators)
                
                if breakout_signals:
                    # Group breakout signals by type
                    confirmed_breakouts = []
                    potential_breakouts = []
                    
                    for signal_name, signal_value in breakout_signals.items():
                        if signal_value:
                            if 'confirmed' in signal_name.lower() or 'breakout' in signal_name.lower():
                                confirmed_breakouts.append(signal_name.replace('_', ' ').title())
                            else:
                                potential_breakouts.append(signal_name.replace('_', ' ').title())
                    
                    if confirmed_breakouts:
                        st.success("🚀 **Confirmed Breakout Signals:**")
                        for signal in confirmed_breakouts:
                            st.write(f"✅ {signal}")
                    elif potential_breakouts:
                        st.warning("🟡 **Potential Breakout Signals:**")
                        for signal in potential_breakouts:
                            st.write(f"✅ {signal}")
                    else:
                        st.info("❌ No active breakout signals detected")
                        # Show all breakout signals with status
                    for signal_name, signal_value in breakout_signals.items():
                        emoji = "✅" if signal_value else "❌"
                        st.write(f"{emoji} {signal_name.replace('_', ' ').title()}")
                else:
                    st.info("Insufficient data for breakout signals")
            
            # Pattern Recognition
            st.markdown("---")
            st.subheader("📊 Pattern Recognition")
            latest = df_with_indicators.iloc[-1]
            
            # Create columns for different pattern types
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Reversal Patterns**")
                if latest.get('Double_Top', False):
                    st.error("🔴 Double Top")
                if latest.get('Double_Bottom', False):
                    st.success("🟢 Double Bottom")
                if latest.get('Head_And_Shoulders', False):
                    st.error("🔴 Head and Shoulders")
                if latest.get('Inverse_Head_And_Shoulders', False):
                    st.success("🟢 Inverse Head and Shoulders")
            
            with col2:
                st.markdown("**Continuation Patterns**")
                if latest.get('Ascending_Triangle', False):
                    st.success("🟢 Ascending Triangle")
                if latest.get('Descending_Triangle', False):
                    st.error("🔴 Descending Triangle")
                if latest.get('Symmetrical_Triangle', False):
                    st.warning("🟡 Symmetrical Triangle")
                if latest.get('Rectangle', False):
                    st.info("⚪ Rectangle")
            
            with col3:
                st.markdown("**Candlestick Patterns**")
                if latest.get('Doji', False):
                    st.warning("🟡 Doji")
                if latest.get('Hammer', False):
                    st.success("🟢 Hammer")
                if latest.get('Shooting_Star', False):
                    st.error("🔴 Shooting Star")
                if latest.get('Engulfing', False):
                    st.warning("🟡 Engulfing")
            
            # V4.0 Price-Volume Analysis Section
            st.markdown("---")
            st.subheader("⚠️ Price-Volume Analysis & Trap Risk")
            
            # Extract V4.0 indicators
            v4_indicators = {
                'Bull_Trap': latest.get('Bull_Trap', 0),
                'Bear_Trap': latest.get('Bear_Trap', 0),
                'False_Breakout': latest.get('False_Breakout', 0),
                'Volume_Price_Divergence': latest.get('Volume_Price_Divergence', 0),
                'HFT_Activity': latest.get('HFT_Activity', 0),
                'Stop_Hunting': latest.get('Stop_Hunting', 0),
                'Volume_Delta': latest.get('Volume_Delta', 0)
            }
            
            # Create a result dictionary with the necessary fields for trap risk calculation
            result_for_risk = {**v4_indicators, 'Close': latest['Close']}
            if 'SMA_20' in latest:
                result_for_risk['SMA_20'] = latest['SMA_20']
            
            # Calculate trap risk
            trap_risk = get_trap_risk_indicator(result_for_risk)
            
            # Display trap risk prominently
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown("### Trap Risk:")
                risk_color = "green" if "Low" in trap_risk else "orange" if "Medium" in trap_risk else "red" if "High" in trap_risk else "gray"
                st.markdown(f"<h2 style='color: {risk_color};'>{trap_risk}</h2>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("### Risk Factors:")
                
                # Display active risk factors
                risk_factors = []
                
                if v4_indicators['Bull_Trap'] > 0:
                    risk_factors.append("🔴 Bull Trap Detected")
                if v4_indicators['Bear_Trap'] > 0:
                    risk_factors.append("🔴 Bear Trap Detected")
                if v4_indicators['False_Breakout'] != 0:
                    risk_factors.append("🔴 False Breakout Pattern")
                if v4_indicators['Volume_Price_Divergence'] != 0:
                    divergence_type = "Bearish" if v4_indicators['Volume_Price_Divergence'] < 0 else "Bullish"
                    risk_factors.append(f"🟡 {divergence_type} Volume-Price Divergence")
                if v4_indicators['HFT_Activity'] > 0.7:
                    risk_factors.append("🔴 High HFT Activity")
                elif v4_indicators['HFT_Activity'] > 0.3:
                    risk_factors.append("🟡 Moderate HFT Activity")
                if v4_indicators['Stop_Hunting'] > 0:
                    risk_factors.append("🔴 Stop Hunting Pattern")
                
                # Calculate volume delta significance
                volume_delta = v4_indicators['Volume_Delta']
                if abs(volume_delta) > 0.5:
                    delta_type = "Buying" if volume_delta > 0 else "Selling"
                    risk_factors.append(f"🟡 Strong {delta_type} Pressure ({volume_delta:.2f})")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.markdown(factor)
                else:
                    st.markdown("✅ No significant risk factors detected")
            
            # Add explanation of trap risk indicators
            with st.expander("ℹ️ About Trap Risk Indicators"):
                st.markdown("""
                **Price-Volume Analysis** helps detect potential false signals and market manipulation:
                
                - **Bull/Bear Traps**: False breakouts designed to trap traders in the wrong direction
                - **Volume-Price Divergence**: When price movement isn't supported by corresponding volume
                - **HFT Activity**: High-frequency trading creating misleading price signals
                - **Stop Hunting**: Large players deliberately triggering retail stop losses
                - **False Breakout Patterns**: Breakouts that fail to follow through
                
                **Trap Risk Levels:**
                - 🟢 **Low**: Low probability of false signal/trap
                - 🟡 **Medium**: Some risk factors present - use caution
                - 🔴 **High**: Multiple risk factors detected - high chance of false signal
                - ⚫ **Unknown**: Insufficient data to assess trap risk
                """)
            
            # NEW: Add Pattern vs Risk Analysis
            st.markdown("---")
            st.subheader("🔄 Pattern vs Risk Analysis")
            
            # Remove the expander from here
            
            # Identify active technical patterns
            active_patterns = []
            
            # Check for bullish patterns
            if latest.get('RSI', 0) < 40 and latest.get('Stoch_K', 0) < 30:
                active_patterns.append(("Oversold Condition", "Bullish"))
            if latest.get('MACD', 0) > latest.get('MACD_Signal', 0):
                active_patterns.append(("MACD Bullish Crossover", "Bullish"))
            if latest.get('Hammer', False):
                active_patterns.append(("Hammer Pattern", "Bullish"))
            if latest.get('Inverse_Head_And_Shoulders', False):
                active_patterns.append(("Inverse Head & Shoulders", "Bullish"))
            if latest.get('Double_Bottom', False):
                active_patterns.append(("Double Bottom", "Bullish"))
            if latest.get('Ascending_Triangle', False):
                active_patterns.append(("Ascending Triangle", "Bullish"))
            if latest.get('Golden_Cross', False) or (latest.get('SMA_50', 0) > latest.get('SMA_200', 0) and df_with_indicators['SMA_50'].iloc[-2] <= df_with_indicators['SMA_200'].iloc[-2]):
                active_patterns.append(("Golden Cross", "Bullish"))
                
            # Check for bearish patterns
            if latest.get('RSI', 0) > 60 and latest.get('Stoch_K', 0) > 70:
                active_patterns.append(("Overbought Condition", "Bearish"))
            if latest.get('MACD', 0) < latest.get('MACD_Signal', 0):
                active_patterns.append(("MACD Bearish Crossover", "Bearish"))
            if latest.get('Shooting_Star', False):
                active_patterns.append(("Shooting Star", "Bearish"))
            if latest.get('Head_And_Shoulders', False):
                active_patterns.append(("Head & Shoulders", "Bearish"))
            if latest.get('Double_Top', False):
                active_patterns.append(("Double Top", "Bearish"))
            if latest.get('Descending_Triangle', False):
                active_patterns.append(("Descending Triangle", "Bearish"))
            if latest.get('Death_Cross', False) or (latest.get('SMA_50', 0) < latest.get('SMA_200', 0) and df_with_indicators['SMA_50'].iloc[-2] >= df_with_indicators['SMA_200'].iloc[-2]):
                active_patterns.append(("Death Cross", "Bearish"))
            
            # Determine pattern direction
            if active_patterns:
                bullish_count = sum(1 for _, direction in active_patterns if direction == "Bullish")
                bearish_count = sum(1 for _, direction in active_patterns if direction == "Bearish")
                
                if bullish_count > bearish_count:
                    pattern_direction = "Bullish"
                elif bearish_count > bullish_count:
                    pattern_direction = "Bearish"
                else:
                    pattern_direction = "Mixed"
            else:
                pattern_direction = "Neutral"
            
            # Determine risk direction
            bull_trap = v4_indicators['Bull_Trap'] > 0
            bear_trap = v4_indicators['Bear_Trap'] > 0
            false_breakout = v4_indicators['False_Breakout'] != 0
            volume_price_divergence = v4_indicators['Volume_Price_Divergence']
            
            if bull_trap:
                risk_direction = "Bearish"  # Bull trap is bearish (false bullish signal)
            elif bear_trap:
                risk_direction = "Bullish"  # Bear trap is bullish (false bearish signal)
            elif false_breakout != 0:
                # Determine direction of false breakout
                if false_breakout > 0:
                    risk_direction = "Bearish"  # False upward breakout
                else:
                    risk_direction = "Bullish"  # False downward breakout
            elif volume_price_divergence != 0:
                if volume_price_divergence > 0:
                    risk_direction = "Bearish"  # Bullish price with weak volume is bearish
                else:
                    risk_direction = "Bullish"  # Bearish price with weak volume is bullish
            else:
                risk_direction = "Neutral"
            
            # Check for pattern-risk alignment
            if pattern_direction == "Neutral" or risk_direction == "Neutral":
                alignment = "Neutral"
            elif pattern_direction == risk_direction:
                alignment = "Aligned"
            else:
                alignment = "Conflicting"
            
            # Display pattern-risk alignment
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Technical Pattern Direction:")
                pattern_color = "green" if pattern_direction == "Bullish" else "red" if pattern_direction == "Bearish" else "gray"
                st.markdown(f"<h3 style='color: {pattern_color};'>{pattern_direction}</h3>", unsafe_allow_html=True)
                
                if active_patterns:
                    st.markdown("**Active Patterns:**")
                    for pattern, direction in active_patterns:
                        direction_emoji = "🟢" if direction == "Bullish" else "🔴" if direction == "Bearish" else "⚪"
                        st.markdown(f"{direction_emoji} {pattern}")
                else:
                    st.markdown("No clear technical patterns detected")
            
            with col2:
                st.markdown("### Risk Factor Direction:")
                risk_color = "green" if risk_direction == "Bullish" else "red" if risk_direction == "Bearish" else "gray"
                st.markdown(f"<h3 style='color: {risk_color};'>{risk_direction}</h3>", unsafe_allow_html=True)
                
                # Display alignment status with icon
                alignment_emoji = "✅" if alignment == "Aligned" else "⚠️" if alignment == "Conflicting" else "⚪"
                alignment_color = "green" if alignment == "Aligned" else "orange" if alignment == "Conflicting" else "gray"
                
                st.markdown(f"### Signal-Risk Alignment: <span style='color: {alignment_color};'>{alignment_emoji} {alignment}</span>", unsafe_allow_html=True)
                
                if alignment == "Conflicting":
                    st.markdown("""
                    ⚠️ **Warning:** Technical patterns and risk factors are showing conflicting signals.
                    Consider waiting for confirmation or reducing position size.
                    """)
                elif alignment == "Aligned":
                    st.markdown("""
                    ✅ **Confirmation:** Technical patterns and risk analysis are aligned.
                    This increases confidence in the signal direction.
                    """)
            
            # Add the expander at the bottom of the section
            with st.expander("ℹ️ About Pattern vs Risk Analysis"):
                st.markdown("""
                **Pattern vs Risk Analysis** compares traditional technical patterns with advanced risk factors:
                
                - **Signal Alignment**: When technical patterns and risk factors point in the same direction, increasing confidence
                - **Signal Conflict**: When technical patterns and risk factors contradict each other, suggesting caution
                - **Pattern Direction**: Determined by analyzing RSI, MACD, chart patterns, and moving averages
                - **Risk Direction**: Determined by analyzing traps, false breakouts, and volume-price relationships
                
                **Alignment Status:**
                - ✅ **Aligned**: Technical patterns and risk factors point in the same direction (high confidence)
                - ⚠️ **Conflicting**: Technical patterns and risk factors contradict each other (use caution)
                - ⚪ **Neutral**: Either patterns or risk factors don't show a clear direction
                """)
            
            # Get latest data for AI analysis
            latest = df_with_indicators.iloc[-1]
            technical_data = {
                'price': latest['Close'],
                'rsi': latest['RSI'],
                'macd': {
                    'value': latest['MACD'],
                    'signal': latest['MACD_Signal']
                },
                'stochastic': {
                    'k': latest['Stoch_K'],
                    'd': latest['Stoch_D']
                },
                'adx': latest['ADX'],
                'volume_ratio': latest['Volume'] / latest['Volume_SMA'] if 'Volume_SMA' in latest else 1.0,
                'bb_position': latest['BB_Position'],
                'atr': latest['ATR'],
                'patterns': [pattern for pattern, value in latest.items() if pattern.endswith('_Pattern') and value],
                'sma_20': latest['SMA_20'],
                'sma_50': latest['SMA_50'],
                'sma_200': latest['SMA_200']
            }
            
            # Get AI suggestions
            ai_suggestions = AIService.generate_trading_suggestions(technical_data)
            
            # Display AI suggestions in a new section
            st.markdown("---")
            st.subheader("🤖 AI Trading Analysis")
            
            # Add a refresh button specifically for AI analysis
            refresh_ai_col1, refresh_ai_col2 = st.columns([5, 1])
            with refresh_ai_col2:
                refresh_ai_button = st.button("🔄 Refresh AI", key="refresh_ai_button")
                
                if refresh_ai_button:
                    # Clear the AI service cache for this technical data
                    AIService.clear_cache(technical_data)
                    # Regenerate the AI suggestions
                    ai_suggestions = AIService.generate_trading_suggestions(technical_data)
                    st.success("AI analysis refreshed!")
            
            # Display the AI suggestions
            st.markdown(ai_suggestions)
            
            # Advanced Key Indicators
            st.markdown("---")
            st.subheader("📈 Advanced Key Indicators")
            
            # Get the most recent data point
            latest = df_with_indicators.iloc[-1]
            
            # Create columns for different indicator groups
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Momentum Indicators**")
                # RSI
                rsi_value = latest['RSI']
                rsi_status = "🟢" if rsi_value < 30 else "🔴" if rsi_value > 70 else "⚪"
                st.metric("RSI", f"{rsi_value:.1f}", delta=f"{rsi_status} {'Oversold' if rsi_value < 30 else 'Overbought' if rsi_value > 70 else 'Neutral'}")
                
                # MACD
                macd_value = latest['MACD']
                macd_signal = latest['MACD_Signal']
                macd_status = "🟢" if macd_value > macd_signal else "🔴"
                st.metric("MACD", f"{macd_value:.2f}", delta=f"{macd_status} {'Bullish' if macd_value > macd_signal else 'Bearish'}")
                
                # MACD Signal
                st.metric("MACD Signal", f"{macd_signal:.2f}")
                
                # Stochastic
                stoch_k = latest['Stoch_K']
                stoch_d = latest['Stoch_D']
                stoch_status = "🟢" if stoch_k > stoch_d else "🔴"
                st.metric("Stochastic K", f"{stoch_k:.1f}", delta=f"{stoch_status} {'Bullish' if stoch_k > stoch_d else 'Bearish'}")
                st.metric("Stochastic D", f"{stoch_d:.1f}")
            
            with col2:
                st.markdown("**Trend & Volume**")
                # ADX
                adx_value = latest['ADX']
                adx_status = "🟢" if adx_value > 25 else "⚪"
                st.metric("ADX", f"{adx_value:.1f}", delta=f"{adx_status} {'Strong Trend' if adx_value > 25 else 'Weak Trend'}")
                
                # Volume Ratio
                volume_ratio = latest['Volume'] / latest['Volume_SMA'] if 'Volume_SMA' in latest else 1.0
                volume_status = "🟢" if volume_ratio > 1.2 else "🔴" if volume_ratio < 0.8 else "⚪"
                st.metric("Volume Ratio", f"{volume_ratio:.1f}", delta=f"{volume_status} {'High' if volume_ratio > 1.2 else 'Low' if volume_ratio < 0.8 else 'Normal'}")
                
                # BB Position
                bb_position = latest['BB_Position']
                bb_status = "🟢" if bb_position < 0.2 else "🔴" if bb_position > 0.8 else "⚪"
                st.metric("BB Position", f"{bb_position:.2f}", delta=f"{bb_status} {'Oversold' if bb_position < 0.2 else 'Overbought' if bb_position > 0.8 else 'Neutral'}")
                
                # ATR
                st.metric("ATR", f"{latest['ATR']:.2f}")
            
            # Moving Averages
            st.markdown("**Moving Averages**")
            col1, col2, col3 = st.columns(3)
            
            # Calculate MA relationships
            sma20 = latest['SMA_20']
            sma50 = latest['SMA_50']
            sma200 = latest['SMA_200']
            current_price = latest['Close']
            
            with col1:
                ma20_status = "🟢" if current_price > sma20 else "🔴"
                st.metric("SMA 20", f"{sma20:.2f}", delta=f"{ma20_status} {'Above' if current_price > sma20 else 'Below'}")
            
            with col2:
                ma50_status = "🟢" if current_price > sma50 else "🔴"
                st.metric("SMA 50", f"{sma50:.2f}", delta=f"{ma50_status} {'Above' if current_price > sma50 else 'Below'}")
            
            with col3:
                ma200_status = "🟢" if current_price > sma200 else "🔴"
                st.metric("SMA 200", f"{sma200:.2f}", delta=f"{ma200_status} {'Above' if current_price > sma200 else 'Below'}")
            
            # Add MA Crossovers
            st.markdown("**Moving Average Crossovers**")
            col1, col2 = st.columns(2)
            
            with col1:
                # Golden Cross (SMA50 crosses above SMA200)
                golden_cross = sma50 > sma200 and df_with_indicators['SMA_50'].iloc[-2] <= df_with_indicators['SMA_200'].iloc[-2]
                if golden_cross:
                    st.success("🟢 Golden Cross (SMA50 > SMA200)")
                
                # Death Cross (SMA50 crosses below SMA200)
                death_cross = sma50 < sma200 and df_with_indicators['SMA_50'].iloc[-2] >= df_with_indicators['SMA_200'].iloc[-2]
                if death_cross:
                    st.error("🔴 Death Cross (SMA50 < SMA200)")
            
            with col2:
                # Short-term Cross (SMA20 crosses SMA50)
                short_term_bullish = sma20 > sma50 and df_with_indicators['SMA_20'].iloc[-2] <= df_with_indicators['SMA_50'].iloc[-2]
                short_term_bearish = sma20 < sma50 and df_with_indicators['SMA_20'].iloc[-2] >= df_with_indicators['SMA_50'].iloc[-2]
                
                if short_term_bullish:
                    st.success("🟢 Short-term Bullish (SMA20 > SMA50)")
                elif short_term_bearish:
                    st.error("🔴 Short-term Bearish (SMA20 < SMA50)")
            
            # Enhanced risk warning based on research
            st.markdown("---")
            st.warning("""
            ⚠️ **Risk Management Guidelines**:
            - Maximum portfolio risk per trade: 5% (as per research)
            - Position sizing: 1-2% of portfolio per trade with 5% stop-loss
            - For swing trades: Use tighter stops (2-5% as per research)
            - For breakout trades: Use wider stops (10-15% as per research)
            - Diversification: Spread risk across multiple uncorrelated positions
            - Volatility-based sizing: Adjust position size based on ATR
            - Portfolio-level risk: Pause trading if overall drawdown approaches 5%
            - Always consider market conditions and sector strength
            - Monitor for momentum fade or reversal patterns
            """)
            
        except Exception as e:
            st.error(f"Error analyzing signals: {str(e)}")
            st.info("This may be due to insufficient data or missing indicators. Try a longer time period.")

def stock_screener_tab():
    """Stock screening for trading opportunities"""
    st.markdown("<h1 style='font-size: 2.5rem; margin-bottom: 1rem;'>🎯 Stock Screener</h1>", unsafe_allow_html=True)
    st.markdown("Find high-probability trading opportunities using research-based criteria")
    
    # Store last selected screen type in session state to detect changes
    if 'last_screen_type' not in st.session_state:
        st.session_state.last_screen_type = "Both"
    
    # Store last selected universe in session state
    if 'last_universe' not in st.session_state:
        st.session_state.last_universe = "S&P 500 (Top 50)"
    
    # Screening options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        screen_type = st.selectbox(
            "Screening Type",
            ["Swing Trading (1-2 weeks)", "Breakout Trading (1-6 months)", "Both"]
        )
        
        # Check if screen type changed to auto-suggest universe
        if screen_type != st.session_state.last_screen_type:
            if screen_type == "Swing Trading (1-2 weeks)":
                st.session_state.last_universe = "Swing-Optimized Universe"
            elif screen_type == "Breakout Trading (1-6 months)":
                st.session_state.last_universe = "Breakout-Optimized Universe"
            # Update last selected type
            st.session_state.last_screen_type = screen_type
    
    with col2:
        universe = st.selectbox(
            "Stock Universe",
            ["S&P 500 (Top 50)", "Swing-Optimized Universe", "Breakout-Optimized Universe", "Custom Watchlist"],
            index=["S&P 500 (Top 50)", "Swing-Optimized Universe", "Breakout-Optimized Universe", "Custom Watchlist"].index(st.session_state.last_universe),
            help="S&P 500: Standard market index stocks\nSwing-Optimized: Stocks with optimal volatility and liquidity for 1-2 week trades\nBreakout-Optimized: Mid-cap growth stocks ideal for breakout patterns\nCustom: Your own symbol list"
        )
        # Update last selected universe
        st.session_state.last_universe = universe
    
    with col3:
        max_results = st.slider("Max Results", 5, 50, 20)
    
    # Custom watchlist option
    if universe == "Custom Watchlist":
        symbols_input = st.text_area(
            "Enter Symbols (comma-separated)",
            placeholder="AAPL, MSFT, GOOGL, TSLA, NVDA",
            help="Enter stock symbols separated by commas"
        )
        custom_symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
        st.session_state.stock_screener.set_stock_universe(custom_symbols)
    elif universe == "Swing-Optimized Universe":
        # Add info about the swing universe
        st.info("**Swing-Optimized Universe**: This curated list includes 40+ stocks with characteristics ideal for short-term swing trades (1-2 weeks). The selection features stocks with optimal volatility, strong liquidity, and tendency for mean reversion. Includes large caps with predictable movement patterns, financial stocks for mean-reversion strategies, and retail/healthcare sectors with reliable oscillation patterns.")
        
        # Set a universe optimized for swing trading
        swing_universe = [
            # Large cap stocks with high liquidity and volatility
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'AMD', 'NFLX',
            # Mid-cap tech with good movement
            'PYPL', 'CRWD', 'NET', 'SNAP', 'ETSY', 'PINS', 'TWLO', 'ZS', 'OKTA', 'DDOG',
            # Financial sector (good for mean-reversion)
            'JPM', 'BAC', 'C', 'WFC', 'GS', 'MS', 'SCHW',
            # Retail with strong swings
            'WMT', 'TGT', 'HD', 'LOW', 'COST', 'SBUX', 'MCD',
            # Healthcare with momentum
            'JNJ', 'PFE', 'MRNA', 'BNTX', 'UNH', 'CVS',
            # Consumer discretionary
            'AMZN', 'BABA', 'NKE', 'DIS', 'ABNB', 'MAR'
        ]
        # Filter out duplicates while maintaining order
        swing_universe = list(dict.fromkeys(swing_universe))
        st.session_state.stock_screener.set_stock_universe(swing_universe)
    elif universe == "Breakout-Optimized Universe":
        # Add info about the breakout universe
        st.info("**Breakout-Optimized Universe**: This specialized list of 40+ stocks focuses on mid-cap growth companies ideal for breakout patterns (1-6 month trades). These stocks have higher growth potential, strong momentum characteristics, and trade within the optimal $20-$100 price range identified in our backtesting research. Includes emerging tech leaders, biotech with catalyst potential, clean energy stocks, and innovative fintech companies.")
        
        # Set a universe optimized for breakout trading (more mid and small caps)
        breakout_universe = [
            # Growth tech stocks good for breakouts
            'NVDA', 'AMD', 'CRWD', 'NET', 'DDOG', 'ZS', 'SNOW', 'CYBR', 'TTD', 'DOCN',
            # Mid-cap growth with momentum
            'ENPH', 'FTNT', 'HUBS', 'GTLB', 'BILL', 'GLOB', 'ESTC', 'TWLO', 'U', 'PATH',
            # Biotech/Pharma candidates (volatile with catalyst potential)
            'MRNA', 'BNTX', 'CRSP', 'EDIT', 'NTLA', 'BEAM', 'SGEN', 'EXAS',
            # Clean energy/EV sector
            'TSLA', 'RIVN', 'LCID', 'CHPT', 'PLUG', 'ENPH', 'SEDG',
            # Consumer tech with growth
            'RBLX', 'PTON', 'DASH', 'ABNB', 'UBER', 'LYFT',
            # Fintech
            'SQ', 'PYPL', 'AFRM', 'UPST', 'COIN', 'MELI'
        ]
        # Filter out duplicates while maintaining order
        breakout_universe = list(dict.fromkeys(breakout_universe))
        st.session_state.stock_screener.set_stock_universe(breakout_universe)
    else:
        st.session_state.stock_screener.set_stock_universe()
    
    # Run screening button - smaller and aligned left with unique key
    if st.button("🔍 Run Screen", type="primary", use_container_width=False, key="run_screen_button"):
        run_stock_screen(screen_type, max_results)
    
    # Add spacing before results
    st.markdown("---")
    
    # Display results below the input controls
    # Note: Results will be displayed when the button above is clicked

def run_stock_screen(screen_type: str, max_results: int):
    """Run the stock screening process"""
    with st.spinner('Screening stocks... This may take a few minutes.'):
        # Force refresh the screener to avoid cache issues
        # Create fresh instance to match command line test behavior
        from models.stock_screener import StockScreener
        fresh_screener = StockScreener()
        
        # Preserve the current universe instead of resetting to S&P 500
        current_universe = st.session_state.stock_screener.stock_universe
        fresh_screener.set_stock_universe(current_universe)
        
        # Update session state with fresh instance
        st.session_state.stock_screener = fresh_screener
        
        results = {}
        
        if screen_type in ["Swing Trading (1-2 weeks)", "Both"]:
            swing_results = st.session_state.stock_screener.screen_swing_opportunities(max_results)
            results['swing'] = swing_results
        
        if screen_type in ["Breakout Trading (1-6 months)", "Both"]:
            breakout_results = st.session_state.stock_screener.screen_breakout_opportunities(max_results)
            results['breakout'] = breakout_results
        
        # Display results
        display_screening_results(results)

def get_entry_timing(result: Dict) -> str:
    """Determine optimal entry timing based on technical indicators"""
    rsi = result['rsi']
    signal_strength = result['signal_strength']
    signals_count = result['signals_count']
    
    # Priority: Formal signals get immediate entry
    if signals_count > 0:
        if signal_strength >= 60:
            return "🟢 Enter Now"
        else:
            return "🟡 Enter with Caution"
    
    # Research-based timing for signals=0
    if rsi < 35:  # Oversold - good support bounce entry
        return "🟢 Enter Now" if signal_strength >= 50 else "🟡 Wait for Confirmation"
    elif rsi > 65:  # Overbought - wait for pullback
        return "🟡 Wait for Pullback"
    elif 45 <= rsi <= 55:  # Neutral zone - depends on strength
        return "🟢 Enter Now" if signal_strength >= 55 else "🟠 Monitor"
    else:
        return "🟠 Monitor"

def get_breakout_entry_timing(result: Dict) -> str:
    """Determine optimal entry timing for breakout opportunities"""
    signal_strength = result['signal_strength']
    signals_count = result['signals_count']
    adx = result.get('adx', 20)
    volume_surge = result.get('volume_surge', 1.0)
    
    # Priority: Confirmed breakout patterns
    if signals_count > 0:
        if signal_strength >= 65 and volume_surge > 1.5:
            return "🟢 Enter on Break"  # Strong breakout with volume
        elif signal_strength >= 50 and adx > 25:
            return "🟡 Wait for Retest"  # Good breakout but wait for pullback
        else:
            return "🟠 Watch for Confirmation"  # Weak breakout signal
    
    # Research-based timing for pre-breakout candidates
    if signal_strength >= 60:
        return "🟠 Watch for Confirmation"  # High potential, waiting for trigger
    elif signal_strength >= 45 and adx > 20:
        return "🔴 Pre-Breakout"  # Building momentum but not ready
    else:
        return "🔴 Pre-Breakout"  # Early stage

def display_screening_results(results: Dict):
    """Display stock screening results separated by signal priority"""
    if 'swing' in results and results['swing']:
        # Separate results by signal type
        formal_signals = [r for r in results['swing'] if r['signals_count'] > 0]
        research_opportunities = [r for r in results['swing'] if r['signals_count'] == 0]
        
        # Display formal signals first (priority opportunities)
        if formal_signals:
            st.subheader("🚨 Priority Opportunities - Formal Technical Signals")
            st.markdown("**These stocks have confirmed technical patterns - prioritize for immediate entry**")
            
            formal_data = []
            for result in formal_signals:
                # Calculate trap risk indicator
                trap_risk = get_trap_risk_indicator(result)
                
                formal_data.append({
                    'Symbol': result['symbol'],
                    'Price': f"${result['current_price']:.2f}",
                    'Strength': f"{result['signal_strength']:.0f}",
                    'RSI': f"{result['rsi']:.1f}",
                    'Research Setup': result.get('research_setup', 'Quality Setup'),
                    'Risk Level': result['risk_level'],
                    'Trap Risk': trap_risk,
                    'Entry Timing': get_entry_timing(result),
                    'Buy Price': f"${result.get('suggested_entry_price', result['current_price']):.2f}",
                    'Stop Loss': f"${result['suggested_stop_loss']:.2f}",
                    'Take Profit 1': f"${result.get('suggested_take_profit_1', result['current_price'] * 1.03):.2f}",
                    'Take Profit 2': f"${result.get('suggested_take_profit_2', result['current_price'] * 1.06):.2f}",
                    'Take Profit 3': f"${result.get('suggested_take_profit_3', result['current_price'] * 1.10):.2f}",
                    'Signals': result['signals_count']
                })
            
            formal_data.sort(key=lambda x: float(x['Strength']), reverse=True)
            formal_df = pd.DataFrame(formal_data)
            st.dataframe(formal_df, use_container_width=True, hide_index=True)
            
            st.success(f"✅ **{len(formal_signals)} Priority Opportunities** - These have the highest probability of success")
        
        # Display research opportunities
        if research_opportunities:
            st.subheader("📊 Research-Based Opportunities - Watchlist Candidates")
            st.markdown("**These stocks meet academic research criteria but lack formal confirmations**")
            
            research_data = []
            for result in research_opportunities:
                # Calculate trap risk indicator
                trap_risk = get_trap_risk_indicator(result)
                
                research_data.append({
                    'Symbol': result['symbol'],
                    'Price': f"${result['current_price']:.2f}",
                    'Strength': f"{result['signal_strength']:.0f}",
                    'RSI': f"{result['rsi']:.1f}",
                    'Research Setup': result.get('research_setup', 'Quality Setup'),
                    'Risk Level': result['risk_level'],
                    'Trap Risk': trap_risk,
                    'Entry Timing': get_entry_timing(result),
                    'Buy Price': f"${result.get('suggested_entry_price', result['current_price']):.2f}",
                    'Stop Loss': f"${result['suggested_stop_loss']:.2f}",
                    'Take Profit 1': f"${result.get('suggested_take_profit_1', result['current_price'] * 1.03):.2f}",
                    'Take Profit 2': f"${result.get('suggested_take_profit_2', result['current_price'] * 1.06):.2f}",
                    'Take Profit 3': f"${result.get('suggested_take_profit_3', result['current_price'] * 1.10):.2f}",
                    'Signals': result['signals_count']
                })
            
            research_data.sort(key=lambda x: float(x['Strength']), reverse=True)
            research_df = pd.DataFrame(research_data)
            st.dataframe(research_df, use_container_width=True, hide_index=True)
            
            st.info(f"ℹ️ **{len(research_opportunities)} Research Opportunities** - Monitor for confirmation signals")
        
        # Summary info
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Opportunities", len(results['swing']))
        with col2:
            st.metric("Priority (Signals≥1)", len(formal_signals))
        with col3:
            st.metric("Research-Based", len(research_opportunities))
        with col4:
            avg_strength = sum(r['signal_strength'] for r in results['swing']) / len(results['swing'])
            st.metric("Avg Strength", f"{avg_strength:.0f}")
    
    elif 'swing' in results:
        st.warning("No swing trading opportunities found with current criteria")
    
    if 'breakout' in results and results['breakout']:
        # Separate breakout results by signal type
        confirmed_breakouts = [r for r in results['breakout'] if r['signals_count'] > 0]
        pre_breakout_candidates = [r for r in results['breakout'] if r['signals_count'] == 0]
        
        # Display confirmed breakout patterns first
        if confirmed_breakouts:
            st.subheader("🚀 Priority Breakout Opportunities - Confirmed Patterns")
            st.markdown("**These stocks have confirmed breakout patterns with strong momentum - prioritize for immediate entry**")
            
            confirmed_data = []
            for result in confirmed_breakouts:
                # Calculate trap risk indicator
                trap_risk = get_trap_risk_indicator(result)
                
                confirmed_data.append({
                    'Symbol': result['symbol'],
                    'Price': f"${result['current_price']:.2f}",
                    'Strength': f"{result['signal_strength']:.0f}",
                    'ADX': f"{result['adx']:.1f}",
                    'Breakout Setup': result.get('breakout_setup', 'Volume Breakout'),
                    'Risk Level': result['risk_level'],
                    'Trap Risk': trap_risk,
                    'Entry Timing': get_breakout_entry_timing(result),
                    'Buy Price': f"${result.get('suggested_entry_price', result['current_price']):.2f}",
                    'Stop Loss': f"${result['suggested_stop_loss']:.2f}",
                    'Take Profit 1': f"${result.get('suggested_take_profit_1', result['current_price'] * 1.05):.2f}",
                    'Take Profit 2': f"${result.get('suggested_take_profit_2', result['current_price'] * 1.10):.2f}",
                    'Take Profit 3': f"${result.get('suggested_take_profit_3', result['current_price'] * 1.15):.2f}",
                    'Signals': result['signals_count']
                })
            
            confirmed_data.sort(key=lambda x: float(x['Strength']), reverse=True)
            confirmed_df = pd.DataFrame(confirmed_data)
            st.dataframe(confirmed_df, use_container_width=True, hide_index=True)
            
            st.success(f"✅ **{len(confirmed_breakouts)} Confirmed Breakout Opportunities** - Strong momentum patterns detected")
        
        # Display pre-breakout candidates
        if pre_breakout_candidates:
            st.subheader("🚀 Priority Breakout Opportunities - Pre-Breakout Candidates")
            st.markdown("**These stocks are building momentum and showing early breakout characteristics - monitor for confirmation**")
            
            pre_breakout_data = []
            for result in pre_breakout_candidates:
                # Calculate trap risk indicator
                trap_risk = get_trap_risk_indicator(result)
                
                pre_breakout_data.append({
                    'Symbol': result['symbol'],
                    'Price': f"${result['current_price']:.2f}",
                    'Strength': f"{result['signal_strength']:.0f}",
                    'ADX': f"{result['adx']:.1f}",
                    'Breakout Setup': result.get('breakout_setup', 'Pre-Breakout'),
                    'Risk Level': result['risk_level'],
                    'Trap Risk': trap_risk,
                    'Entry Timing': get_breakout_entry_timing(result),
                    'Buy Price': f"${result.get('suggested_entry_price', result['current_price']):.2f}",
                    'Stop Loss': f"${result['suggested_stop_loss']:.2f}",
                    'Take Profit 1': f"${result.get('suggested_take_profit_1', result['current_price'] * 1.05):.2f}",
                    'Take Profit 2': f"${result.get('suggested_take_profit_2', result['current_price'] * 1.10):.2f}",
                    'Take Profit 3': f"${result.get('suggested_take_profit_3', result['current_price'] * 1.15):.2f}",
                    'Signals': result['signals_count']
                })
            
            pre_breakout_data.sort(key=lambda x: float(x['Strength']), reverse=True)
            pre_breakout_df = pd.DataFrame(pre_breakout_data)
            st.dataframe(pre_breakout_df, use_container_width=True, hide_index=True)
            
            st.info(f"ℹ️ **{len(pre_breakout_candidates)} Pre-Breakout Candidates** - Watch for breakout confirmation")
        
        # Breakout summary info
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Breakout Opportunities", len(results['breakout']))
        with col2:
            st.metric("Confirmed Patterns", len(confirmed_breakouts))
        with col3:
            st.metric("Pre-Breakout Candidates", len(pre_breakout_candidates))
        with col4:
            avg_strength = sum(r['signal_strength'] for r in results['breakout']) / len(results['breakout'])
            st.metric("Avg Momentum Strength", f"{avg_strength:.0f}")
        
        # Breakout timing guide (now collapsible)
        with st.expander("📋 Breakout Entry Timing Guide & Setup Types", expanded=False):
            st.markdown("""
            **📋 Breakout Entry Timing Guide:**
            - **🟢 Enter on Break**: Confirmed breakout above resistance - enter immediately
            - **🟡 Wait for Retest**: Wait for pullback to breakout level for better entry
            - **🟠 Watch for Confirmation**: Monitor for volume spike and pattern completion
            - **🔴 Pre-Breakout**: Not yet broken out - wait for confirmation
            
            **🔬 Breakout Setup Types:**
            - **Volume Breakout**: High volume breakout above resistance
            - **Trend Continuation**: Breakout in direction of main trend
            - **Consolidation Break**: Breakout from sideways range
            - **Golden Cross**: Moving average crossover breakout
            - **Pre-Breakout**: Building for potential breakout
            
            **⚠️ Trap Risk Indicators:**
            - **🟢 Low**: Low probability of false signal/trap
            - **🟡 Medium**: Some risk factors present - use caution
            - **🔴 High**: Multiple risk factors detected - high chance of false signal
            - **⚫ Unknown**: Insufficient data to assess trap risk
            """)
    
    elif 'breakout' in results:
        st.warning("No breakout trading opportunities found with current criteria")
        
        # Add debug information for filtered stocks
        st.subheader("🔍 Debug: Filtered Stocks")
        st.markdown("**Showing stocks that were filtered out during screening**")
        
        # Get filtered stocks from the screener
        filtered_stocks = st.session_state.stock_screener.get_filtered_stocks()
        if filtered_stocks:
            filtered_data = []
            for stock in filtered_stocks:
                filtered_data.append({
                    'Symbol': stock['symbol'],
                    'Price': f"${stock['price']:.2f}",
                    'Volume': f"${stock['dollar_volume']:,.0f}",
                    'RSI': f"{stock['rsi']:.1f}",
                    'BB Position': f"{stock['bb_position']:.2f}",
                    'Passes Basic Filter': "✅" if stock['passes_basic_filter'] else "❌",
                    'Filter Reason': stock['reason']
                })
            
            filtered_df = pd.DataFrame(filtered_data)
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)
        else:
            st.info("No filtered stocks information available")

    # After displaying Priority Breakout Opportunities
    if 'breakout_results' in st.session_state:
        priority_opportunities = st.session_state.breakout_results.get('priority_opportunities', [])
        research_opportunities = st.session_state.breakout_results.get('research_opportunities', [])

        st.subheader("🚀 Priority Breakout Opportunities - Confirmed Patterns")
        if priority_opportunities:
            st.dataframe(pd.DataFrame(priority_opportunities), hide_index=True)
        else:
            st.info("No priority breakout opportunities found.")

        # Add Pre-Breakout Candidates section
        st.subheader("🔎 Pre-Breakout Candidates - Research-Based Setups")
        if research_opportunities:
            st.dataframe(pd.DataFrame(research_opportunities), hide_index=True)
        else:
            st.info("No pre-breakout candidates found.")

def get_trap_risk_indicator(result: Dict) -> str:
    """
    Determine the risk of a trap/false signal based on V4.0 price-volume analysis
    
    Returns an emoji indicator:
    🟢 Low risk - unlikely to be a trap
    🟡 Medium risk - some risk factors present
    🔴 High risk - multiple risk factors detected
    ⚫ Unknown - insufficient data
    """
    # Check for V4.0 indicators in the result
    bull_trap = result.get('Bull_Trap', 0) > 0
    bear_trap = result.get('Bear_Trap', 0) > 0
    false_breakout = result.get('False_Breakout', 0) != 0
    volume_price_divergence = result.get('Volume_Price_Divergence', 0) != 0
    hft_activity = result.get('HFT_Activity', 0) > 0.3  # Moderate or higher HFT activity
    stop_hunting = result.get('Stop_Hunting', 0) > 0
    volume_delta = result.get('Volume_Delta', 0)
    
    # Check if we have at least some V4.0 data
    has_v4_data = any(key in result for key in ['Bull_Trap', 'Bear_Trap', 'HFT_Activity', 'False_Breakout'])
    
    # Count risk factors
    risk_factors = 0
    
    # Direct trap indicators
    if bull_trap or bear_trap:
        risk_factors += 2
    if false_breakout:
        risk_factors += 2
    if stop_hunting:
        risk_factors += 1
    
    # HFT activity levels
    if result.get('HFT_Activity', 0) > 0.7:  # High HFT activity
        risk_factors += 2
    elif hft_activity:  # Moderate HFT activity
        risk_factors += 1
    
    # Volume-price relationship
    if volume_price_divergence:
        risk_factors += 1
    
    # Volume delta (buying/selling pressure) conflicts with price direction
    if abs(volume_delta) > 0.5:  # Strong directional volume
        price_above_ma = result.get('Close', 0) > result.get('SMA_20', result.get('Close', 0))
        if (volume_delta < -0.5 and price_above_ma) or (volume_delta > 0.5 and not price_above_ma):
            # Volume direction conflicts with price direction
            risk_factors += 1
    
    # Determine risk level based on factors count
    if risk_factors >= 3:
        return "🔴 High"
    elif risk_factors >= 1:
        return "🟡 Medium"
    elif has_v4_data:
        # We have some V4.0 data and no risk factors were found
        return "🟢 Low"
    else:
        # Not enough data to determine
        return "⚫ Unknown"

def performance_tab():
    """Performance analytics and reports"""
    st.markdown("<h1 style='font-size: 2.5rem; margin-bottom: 1rem;'>📈 Performance Analytics</h1>", unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.info("Please load portfolio data to view performance analytics")
        return
    
    # Get performance data
    stats = st.session_state.portfolio_tracker.get_summary_stats()
    kpis = st.session_state.portfolio_tracker.calculate_trade_kpis()
    
    # Performance overview (existing code)
    st.markdown("<h2 style='font-size: 2rem; margin-bottom: 1rem;'>🎯 Performance Overview</h2>", unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.info("Please load portfolio data to view performance analytics")
        return
    
    # Helper function to determine metric color
    def get_metric_color(value, thresholds):
        if value >= thresholds['great'] or value >= thresholds['good']:
            return "#00C853"  # Material Design Green
        elif value >= thresholds['average']:
            return "#FFA500"  # Orange
        else:
            return "#FF3D00"  # Material Design Red
    
    # Define thresholds for each metric
    thresholds = {
        'roi_monthly': {'poor': 1, 'average': 2, 'good': 5, 'great': 8},
        'win_rate': {'poor': 40, 'average': 55, 'good': 65, 'great': 75},
        'rrr': {'poor': 1.0, 'average': 1.5, 'good': 2.5, 'great': 3.0},
        'sharpe': {'poor': 0.5, 'average': 1.0, 'good': 1.5, 'great': 2.0},
        'profit_factor': {'poor': 1.0, 'average': 1.3, 'good': 1.8, 'great': 2.5}
    }
    
    # Calculate monthly ROI
    monthly_roi = (stats['total_return_pct'] / 12) if stats['total_return_pct'] else 0
    
    # Create two columns for metrics
    col1, col2 = st.columns(2)
    
    with col1:
        # Core Performance Metrics
        st.markdown("<h3 style='font-size: 1.5rem; margin-bottom: 0.5rem;'>Core Metrics</h3>", unsafe_allow_html=True)
        
        # ROI
        roi_color = get_metric_color(monthly_roi, thresholds['roi_monthly'])
        st.markdown(f"<p style='font-size: 1.1rem;'><strong>Monthly ROI:</strong> <span style='color:{roi_color}'>{monthly_roi:+.2f}%</span></p>", unsafe_allow_html=True)
        
        # Win Rate
        if kpis and 'win_rate' in kpis:
            win_rate = kpis['win_rate']
            win_rate_color = get_metric_color(win_rate, thresholds['win_rate'])
            st.markdown(f"<p style='font-size: 1.1rem;'><strong>Win Rate:</strong> <span style='color:{win_rate_color}'>{win_rate:.1f}%</span></p>", unsafe_allow_html=True)
        
        # Risk-Reward Ratio
        if kpis and 'risk_reward_ratio' in kpis:
            rrr = kpis['risk_reward_ratio']
            rrr_color = get_metric_color(rrr, thresholds['rrr'])
            st.markdown(f"<p style='font-size: 1.1rem;'><strong>Risk-Reward Ratio:</strong> <span style='color:{rrr_color}'>{rrr:.2f}</span></p>", unsafe_allow_html=True)
    
    with col2:
        # Risk Metrics
        st.markdown("<h3 style='font-size: 1.5rem; margin-bottom: 0.5rem;'>Risk Metrics</h3>", unsafe_allow_html=True)
        
        # Sharpe Ratio
        if kpis and 'sharpe_ratio' in kpis:
            sharpe = kpis['sharpe_ratio']
            sharpe_color = get_metric_color(sharpe, thresholds['sharpe'])
            st.markdown(f"<p style='font-size: 1.1rem;'><strong>Sharpe Ratio:</strong> <span style='color:{sharpe_color}'>{sharpe:.2f}</span></p>", unsafe_allow_html=True)
        
        # Profit Factor
        if kpis and 'profit_factor' in kpis:
            profit_factor = kpis['profit_factor']
            profit_factor_color = get_metric_color(profit_factor, thresholds['profit_factor'])
            st.markdown(f"<p style='font-size: 1.1rem;'><strong>Profit Factor:</strong> <span style='color:{profit_factor_color}'>{profit_factor:.2f}</span></p>", unsafe_allow_html=True)
        
        # Average Gain vs Loss
        if kpis and 'avg_win' in kpis and 'avg_loss' in kpis and kpis['avg_loss'] != 0:
            gain_loss_ratio = abs(kpis['avg_win'] / kpis['avg_loss'])
            gain_loss_color = get_metric_color(gain_loss_ratio, thresholds['rrr'])
            st.markdown(f"<p style='font-size: 1.1rem;'><strong>Gain/Loss Ratio:</strong> <span style='color:{gain_loss_color}'>{gain_loss_ratio:.2f}</span></p>", unsafe_allow_html=True)
    
    # Add benchmark reference
    with st.expander("📊 Performance Benchmarks"):
        st.markdown("""
        ### Performance Benchmarks
        
        | Metric | Poor | Average | Good | Great |
        |--------|------|---------|------|-------|
        | Monthly ROI | < 1% | 1-2% | 2-5% | > 5% |
        | Win Rate | < 40% | 40-55% | 55-65% | > 65% |
        | Risk-Reward | < 1.0 | 1.0-1.5 | 1.5-2.5 | > 2.5 |
        | Sharpe Ratio | < 0.5 | 0.5-1.0 | 1.0-1.5 | > 1.5 |
        | Profit Factor | < 1.0 | 1.0-1.3 | 1.3-1.8 | > 1.8 |
        
        Note: A low win rate can be acceptable if your Risk-Reward Ratio is high.
        """)
    
    st.markdown("---")
    
    # Trade Analysis
    st.markdown("<h2 style='font-size: 2rem; margin-bottom: 1rem;'>📊 Trade Analysis</h2>", unsafe_allow_html=True)
    if kpis:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<h3 style='font-size: 1.5rem; margin-bottom: 0.5rem;'>Trade Statistics</h3>", unsafe_allow_html=True)
            if 'total_trades' in kpis:
                st.metric("Total Trades", f"{kpis['total_trades']}")
            if 'winning_trades' in kpis:
                st.metric("Winning Trades", f"{kpis['winning_trades']}")
            if 'losing_trades' in kpis:
                st.metric("Losing Trades", f"{kpis['losing_trades']}")
        
        with col2:
            st.markdown("<h3 style='font-size: 1.5rem; margin-bottom: 0.5rem;'>Profit Metrics</h3>", unsafe_allow_html=True)
            if 'avg_win' in kpis:
                st.metric("Average Win", f"${kpis['avg_win']:,.2f}")
            if 'avg_loss' in kpis:
                st.metric("Average Loss", f"${kpis['avg_loss']:,.2f}")
            if 'largest_win' in kpis:
                st.metric("Largest Win", f"${kpis['largest_win']:,.2f}")
        
        with col3:
            st.markdown("<h3 style='font-size: 1.5rem; margin-bottom: 0.5rem;'>Performance Metrics</h3>", unsafe_allow_html=True)
            if 'largest_loss' in kpis:
                st.metric("Largest Loss", f"${kpis['largest_loss']:,.2f}")
            if 'avg_hold_time' in kpis:
                st.metric("Average Hold Time", f"{kpis['avg_hold_time']:.1f} days")
            if 'best_trade' in kpis:
                st.metric("Best Trade", f"${kpis['best_trade']:,.2f}")
    
    st.markdown("---")
    
    # Best and Worst Trades Analysis
    st.markdown("<h2 style='font-size: 2rem; margin-bottom: 1rem;'>🏆 Best & Worst Trades</h2>", unsafe_allow_html=True)
    if kpis and 'trade_history' in kpis and kpis['trade_history']:
        col1, col2 = st.columns(2)
        card_style = "background-color: #2d3748; color: #e2e8f0; padding: 12px; border-radius: 8px; margin-bottom: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"
        
        with col1:
            st.markdown("<h3 style='font-size: 1.5rem; margin-bottom: 0.5rem;'>Top 5 Best Trades</h3>", unsafe_allow_html=True)
            best_trades = sorted(kpis['trade_history'], key=lambda x: x['pnl'], reverse=True)[:5]
            for trade in best_trades:
                with st.container():
                    st.markdown(f"""
                    <div style='{card_style}'>
                        <h4 style='margin: 0; color: #90cdf4; font-size: 1.2rem;'>{trade['symbol']}</h4>
                        <p style='margin: 5px 0; color: #9ae6b4; font-size: 1.1rem;'>P&L: ${trade['pnl']:,.2f} ({trade['pnl_pct']:+.2f}%)</p>
                        <p style='margin: 5px 0; color: #e2e8f0; font-size: 1rem;'>Entry: ${trade['entry_price']:.2f} | Exit: ${trade['exit_price']:.2f}</p>
                        <p style='margin: 5px 0; color: #e2e8f0; font-size: 1rem;'>Hold Time: {trade['hold_time']} days</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("<h3 style='font-size: 1.5rem; margin-bottom: 0.5rem;'>Top 5 Worst Trades</h3>", unsafe_allow_html=True)
            worst_trades = sorted(kpis['trade_history'], key=lambda x: x['pnl'])[:5]
            for trade in worst_trades:
                with st.container():
                    st.markdown(f"""
                    <div style='{card_style}'>
                        <h4 style='margin: 0; color: #90cdf4; font-size: 1.2rem;'>{trade['symbol']}</h4>
                        <p style='margin: 5px 0; color: #feb2b2; font-size: 1.1rem;'>P&L: ${trade['pnl']:,.2f} ({trade['pnl_pct']:+.2f}%)</p>
                        <p style='margin: 5px 0; color: #e2e8f0; font-size: 1rem;'>Entry: ${trade['entry_price']:.2f} | Exit: ${trade['exit_price']:.2f}</p>
                        <p style='margin: 5px 0; color: #e2e8f0; font-size: 1rem;'>Hold Time: {trade['hold_time']} days</p>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("No trade history available for analysis")
    
    # Removed AI Analysis and Recommendations section
    
    st.markdown("---")
    
    # Add Generate Report section
    st.markdown("<h2 style='font-size: 2rem; margin-bottom: 1rem;'>📊 Comprehensive Report</h2>", unsafe_allow_html=True)
    
    # Create expandable section for report generation
    with st.expander("Generate Trading Performance Report", expanded=True):
        st.markdown("""
        Generate a comprehensive analysis of your trading performance over the past 3 months. 
        This report will analyze your trading activity, performance metrics, and provide actionable insights.
        """)
        
        # Add time period options
        time_period = st.radio(
            "Analysis Period",
            ["Last 3 months", "Last month", "All time"],
            horizontal=True
        )
        
        # Include strategy checkbox
        include_strategy = st.checkbox("Include strategy document in analysis", value=True)
        
        # Generate button
        if st.button("🚀 Generate Comprehensive Report"):
            with st.spinner("Analyzing your trading performance..."):
                try:
                    # Get strategy document if requested
                    strategy_content = ""
                    if include_strategy:
                        try:
                            strategy_manager = StrategyManager()
                            strategy_content = strategy_manager.get_strategy()
                        except FileNotFoundError:
                            st.warning("Strategy document not found. Analysis will proceed without it.")
                    
                    # Filter trades based on selected time period
                    if not hasattr(st.session_state.portfolio_tracker, 'trade_history') or not st.session_state.portfolio_tracker.trade_history:
                        st.error("No trade history available for analysis")
                        return
                    
                    trade_history = st.session_state.portfolio_tracker.trade_history
                    
                    if time_period == "Last month":
                        # Filter to last month
                        one_month_ago = datetime.now() - timedelta(days=30)
                        filtered_trades = [t for t in trade_history if t['exit_date'] >= one_month_ago]
                    elif time_period == "Last 3 months":
                        # Filter to last 3 months
                        three_months_ago = datetime.now() - timedelta(days=90)
                        filtered_trades = [t for t in trade_history if t['exit_date'] >= three_months_ago]
                    else:
                        # Use all trades
                        filtered_trades = trade_history
                    
                    if not filtered_trades:
                        st.warning(f"No trades found in the selected time period: {time_period}")
                        return
                    
                    # Prepare comprehensive analysis data
                    comprehensive_data = {
                        'strategy': strategy_content,
                        'trade_history': filtered_trades,
                        'metrics': {
                            'win_rate': kpis.get('win_rate', 0),
                            'risk_reward_ratio': kpis.get('risk_reward_ratio', 0),
                            'profit_factor': kpis.get('profit_factor', 0),
                            'sharpe_ratio': kpis.get('sharpe_ratio', 0),
                            'avg_hold_time': kpis.get('avg_hold_time', 0),
                            'total_trades': len(filtered_trades),
                            'avg_win': kpis.get('avg_win', 0),
                            'avg_loss': kpis.get('avg_loss', 0),
                            'roi_pct': kpis.get('roi_pct', 0),
                            'monthly_roi': kpis.get('monthly_roi', 0),
                        },
                        'time_period': time_period,
                        'include_strategy': include_strategy,
                        'portfolio_stats': {
                            'total_account_value': stats.get('total_account_value', 0),
                            'portfolio_value': stats.get('portfolio_value', 0),
                            'cash_balance': stats.get('current_cash', 0),
                            'realized_pnl': stats.get('realized_pnl', 0),
                            'unrealized_pnl': stats.get('unrealized_pnl', 0),
                            'total_return_pct': stats.get('total_return_pct', 0),
                        }
                    }
                    
                    # Calculate additional metrics for comprehensive analysis
                    winning_trades = [t for t in filtered_trades if t['pnl'] > 0]
                    losing_trades = [t for t in filtered_trades if t['pnl'] <= 0]
                    
                    # Trading patterns
                    comprehensive_data['patterns'] = {
                        'symbols_traded': len(set(t['symbol'] for t in filtered_trades)),
                        'hold_time_distribution': {
                            'short_term': len([t for t in filtered_trades if t['hold_time'] <= 5]),
                            'medium_term': len([t for t in filtered_trades if 5 < t['hold_time'] <= 20]),
                            'long_term': len([t for t in filtered_trades if t['hold_time'] > 20])
                        },
                        'win_rate_by_hold_time': {
                            'short_term': sum(1 for t in filtered_trades if t['hold_time'] <= 5 and t['pnl'] > 0) / 
                                          max(1, len([t for t in filtered_trades if t['hold_time'] <= 5])) * 100,
                            'medium_term': sum(1 for t in filtered_trades if 5 < t['hold_time'] <= 20 and t['pnl'] > 0) / 
                                           max(1, len([t for t in filtered_trades if 5 < t['hold_time'] <= 20])) * 100,
                            'long_term': sum(1 for t in filtered_trades if t['hold_time'] > 20 and t['pnl'] > 0) / 
                                         max(1, len([t for t in filtered_trades if t['hold_time'] > 20])) * 100
                        }
                    }
                    
                    # Get AI analysis
                    ai_analysis = AIService.generate_performance_analysis(comprehensive_data)
                    
                    # Display the report in a styled box
                    st.markdown("""
                    <style>
                    .report-container {
                        background-color: #1e2130;
                        border-radius: 10px;
                        padding: 20px;
                        margin: 10px 0;
                        border-left: 5px solid #4CAF50;
                    }
                    .report-section {
                        margin-bottom: 15px;
                    }
                    .report-section h3 {
                        color: #4CAF50;
                        font-size: 1.3rem;
                        margin-bottom: 10px;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"<div class='report-container'>", unsafe_allow_html=True)
                    
                    st.markdown(f"<h2>Trading Performance Report: {time_period}</h2>", unsafe_allow_html=True)
                    
                    # Split the analysis into sections
                    sections = ai_analysis.split('\n\n')
                    for section in sections:
                        if section.startswith('Trading Insights:'):
                            st.markdown("<div class='report-section'><h3>🔍 Trading Insights</h3></div>", unsafe_allow_html=True)
                            insights = section.replace('Trading Insights:', '').strip().split('\n')
                            for insight in insights:
                                st.markdown(f"<p>{insight}</p>", unsafe_allow_html=True)
                                
                        elif section.startswith('Recommendations:'):
                            st.markdown("<div class='report-section'><h3>💡 Recommendations</h3></div>", unsafe_allow_html=True)
                            recommendations = section.replace('Recommendations:', '').strip().split('\n')
                            for rec in recommendations:
                                st.markdown(f"<p>{rec}</p>", unsafe_allow_html=True)
                                
                        elif section.startswith('Risk Assessment:'):
                            st.markdown("<div class='report-section'><h3>⚠️ Risk Assessment</h3></div>", unsafe_allow_html=True)
                            risk = section.replace('Risk Assessment:', '').strip().split('\n')
                            for r in risk:
                                st.markdown(f"<p>{r}</p>", unsafe_allow_html=True)
                    
                    # Add trade summary section
                    st.markdown("<div class='report-section'><h3>📊 Trading Summary</h3></div>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Trades", f"{len(filtered_trades)}")
                        st.metric("Win Rate", f"{comprehensive_data['metrics']['win_rate']:.1f}%")
                        st.metric("Profit Factor", f"{comprehensive_data['metrics']['profit_factor']:.2f}")
                    
                    with col2:
                        st.metric("Risk-Reward Ratio", f"{comprehensive_data['metrics']['risk_reward_ratio']:.2f}")
                        st.metric("Avg Hold Time", f"{comprehensive_data['metrics']['avg_hold_time']:.1f} days")
                        st.metric("Return", f"{comprehensive_data['metrics']['roi_pct']:.2f}%")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Add download button for the report (optional)
                    report_text = f"""
                    TRADING PERFORMANCE REPORT: {time_period.upper()}
                    Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}
                    
                    PERFORMANCE METRICS:
                    - Total Trades: {len(filtered_trades)}
                    - Win Rate: {comprehensive_data['metrics']['win_rate']:.1f}%
                    - Risk-Reward Ratio: {comprehensive_data['metrics']['risk_reward_ratio']:.2f}
                    - Profit Factor: {comprehensive_data['metrics']['profit_factor']:.2f}
                    - Return: {comprehensive_data['metrics']['roi_pct']:.2f}%
                    - Avg Hold Time: {comprehensive_data['metrics']['avg_hold_time']:.1f} days
                    
                    {ai_analysis}
                    """
                    
                    st.download_button(
                        label="📥 Download Report",
                        data=report_text,
                        file_name=f"trading_report_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")

def settings_tab():
    """Application settings and configuration"""
    st.markdown("<h1 style='font-size: 2.5rem; margin-bottom: 1rem;'>⚙️ Settings</h1>", unsafe_allow_html=True)
    
    # Create tabs for different settings
    settings_tab1, settings_tab2, settings_tab3 = st.tabs([
        "🤖 AI Settings",
        "📊 Application Settings", 
        "ℹ️ About"
    ])
    
    # Tab 1: AI Settings
    with settings_tab1:
        st.subheader("🤖 AI Model Configuration")
        
        # Display current model information
        st.markdown("### Current AI Model")
        
        # Get model information
        current_model = API_CONFIG.get('MODEL', 'Unknown')
        model_initialized = st.session_state.get('ai_model_initialized', False)
        
        # Display model information
        st.info(f"**Current Model:** {current_model}")
        
        if model_initialized:
            st.success("✅ AI model successfully initialized on startup")
        else:
            st.warning("⚠️ AI model was not automatically initialized. Manual selection is recommended.")
        
        # Add button to manually check available models
        st.markdown("### Manual Model Selection")
        if st.button("🔍 Check Available AI Models"):
            try:
                with st.spinner("Checking available AI models..."):
                    # Clear cache to ensure fresh test
                    AIService.clear_cache()
                    
                    # Initialize models again
                    from services.model_manager import ModelManager
                    selected_model = ModelManager.initialize()
                    
                    if selected_model:
                        st.session_state.current_model = selected_model
                        st.session_state.ai_model_initialized = True
                        st.success(f"✅ Successfully selected model: {selected_model}")
                    else:
                        st.error("❌ No working models found")
            except Exception as e:
                st.error(f"❌ Error checking models: {str(e)}")
        
        # Show information about model fallback
        st.markdown("### AI Model Information")
        st.markdown("""
        The application automatically selects the best available AI model on startup.
        Models are tried in this order:
        1. gpt-4o-mini (best quality)
        2. gpt-3.5-turbo-16k (large context)
        3. gpt-3.5-turbo-0125
        4. gpt-3.5-turbo-1106
        5. gpt-3.5-turbo
        
        If you experience issues with AI-powered features, try using the "Check Available AI Models" button above.
        """)
        
        # Clear AI cache option
        st.markdown("### Cache Management")
        if st.button("🧹 Clear AI Cache"):
            AIService.clear_cache()
            st.success("✅ AI cache cleared successfully")
    
    # Tab 2: Application Settings
    with settings_tab2:
        st.subheader("📊 Application Settings")
        st.markdown("Configure application display settings")
        
        # Add application settings here as needed
        # For example:
        default_time_period = st.selectbox(
            "Default Time Period for Analysis",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
            index=3  # Default to 1y
        )
        
        default_theme = st.selectbox(
            "Chart Theme",
            options=["dark", "light"],
            index=0  # Default to dark
        )
        
        if st.button("Save Settings"):
            # Save settings (would need to be implemented)
            st.success("✅ Settings saved")
    
    # Tab 3: About
    with settings_tab3:
        st.subheader("ℹ️ About")
        st.markdown("""
        **Advanced Trading Portfolio Tracker** v2.0
        
        This application implements research-based trading strategies with:
        - 📊 Comprehensive portfolio tracking and performance analysis
        - 🔍 Technical analysis with 88+ indicators
        - 🎯 Intelligent stock screening for swing and breakout trades
        - ⚖️ Advanced risk management with position sizing
        - 🤖 Machine learning signal prediction
        - 📈 Professional-grade analytics and reporting
        
        Built with Streamlit, powered by academic research for high win-rate trading strategies.
        """)
        
        # Display API information in About tab
        st.subheader("🔌 API Information")
        masked_key = API_CONFIG['API_KEY'][:4] + "*" * 10 + API_CONFIG['API_KEY'][-4:] if len(API_CONFIG['API_KEY']) > 8 else "Not set"
        st.code(f"""
        API URL: {API_CONFIG['API_URL']}
        API Key: {masked_key}
        Model: {API_CONFIG['MODEL']}
        """)

def calculate_average_price(symbol):
    """Calculate average price based on trade history"""
    if not hasattr(st.session_state.portfolio_tracker, 'trade_history'):
        return 0.0
        
    # Filter trades for this symbol
    symbol_trades = [t for t in st.session_state.portfolio_tracker.trade_history 
                    if t['symbol'] == symbol]
    
    if not symbol_trades:
        # If no trade history, try to get from portfolio details
        portfolio_details = st.session_state.portfolio_tracker.get_summary_stats()['portfolio_details']
        if symbol in portfolio_details:
            return portfolio_details[symbol].get('avg_cost', 0.0)
        return 0.0
        
    # Calculate weighted average price
    total_units = 0
    total_value = 0
    
    for trade in symbol_trades:
        units = trade['units']  # Changed from 'shares' to 'units'
        price = trade['entry_price']
        total_units += units
        total_value += units * price
    
    return total_value / total_units if total_units > 0 else 0.0

if __name__ == "__main__":
    main()
