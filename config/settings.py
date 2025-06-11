"""
Configuration settings for the Trading Portfolio Tracker
"""

# Application Settings
APP_TITLE = "📈 Advanced Trading Portfolio Tracker"
APP_ICON = "📈"
DEFAULT_STARTING_CASH = 56127.46

# Risk Management Settings
MAX_PORTFOLIO_RISK = 0.05  # 5% maximum portfolio risk
MAX_POSITION_RISK = 0.02   # 2% maximum risk per position
DEFAULT_STOP_LOSS = 0.05   # 5% stop loss

# Technical Analysis Settings
RSI_PERIOD = 14
STOCHASTIC_K_PERIOD = 14
STOCHASTIC_D_PERIOD = 3
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
ADX_PERIOD = 14
ATR_PERIOD = 14
VOLUME_SMA_PERIOD = 10

# Moving Average Settings
SMA_SHORT = 20
SMA_MEDIUM = 50
SMA_LONG = 200
EMA_SHORT = 12
EMA_LONG = 26

# Screening Criteria
MIN_MARKET_CAP = 1e9  # $1B minimum market cap
MIN_VOLUME = 1e6      # $1M minimum daily volume
MIN_PRICE = 5.0       # $5 minimum stock price

# Swing Trading Settings - UPDATED FOR HIGHER WIN RATE
SWING_RSI_OVERBOUGHT = 75  # Increased from 70 for more reliable reversals
SWING_RSI_OVERSOLD = 28    # Adjusted from 30 for better mean-reversion entries
SWING_STOP_LOSS = 0.02     # Tightened from 3% to 2% for better risk management (research shows tighter stops work better)
SWING_SUPPORT_RESISTANCE_PERIOD = 14  # Reduced from 20 to focus on more recent S/R levels
SWING_PRICE_PROXIMITY_THRESHOLD = 0.015  # Tightened from 2% to 1.5% for higher precision entries
SWING_VOLUME_CONFIRMATION = 1.35  # Increased from 1.2 to 1.35 for stronger volume confirmation
SWING_MA_PROXIMITY_THRESHOLD = 0.01  # Maintained at 1% proximity to moving averages
SWING_PATTERN_CONFIRMATION = True  # NEW: Require chart pattern confirmation for higher probability
SWING_MULTI_TIMEFRAME_CONFIRM = True  # NEW: Require confirmation across multiple timeframes
SWING_WIN_RATE_THRESHOLD = 65  # NEW: Minimum historical success rate required for pattern entry
SWING_OPTIMAL_PRICE_RANGE = (15, 80)  # NEW: Optimal price range for swing trades
SWING_TREND_FILTER = True  # NEW: Only take swings in direction of larger trend (higher win rate)

# Breakout Trading Settings - UPDATED BASED ON RESEARCH
BREAKOUT_VOLUME_MULTIPLIER = 1.8  # Increased from 1.5 to 1.8 (80% above average volume)
BREAKOUT_STOP_LOSS = 0.08        # Improved from 10% to 8% for better risk management
BREAKOUT_ADX_THRESHOLD = 22      # Lowered from 25 to 22 to catch more early breakouts
BREAKOUT_CONSOLIDATION_DAYS = 14  # NEW: Look for 2+ weeks of consolidation 
BREAKOUT_PRICE_RANGE_LIMIT = 0.15  # NEW: Consolidation should be tight (<15% range)
BREAKOUT_MIN_VOLATILITY_ADR = 1.5  # NEW: Minimum Average Daily Range %
BREAKOUT_OPTIMAL_PRICE_RANGE = (20, 100)  # NEW: Optimal price range for breakouts ($20-$100)
BREAKOUT_SEASONALITY_BOOST = 1.2  # NEW: Boost score during favorable seasons

# Machine Learning Settings
ML_LOOKBACK_PERIOD = 252  # 1 year of trading days
ML_FEATURES_COUNT = 88    # Total technical indicators as per research
ML_TRAIN_TEST_SPLIT = 0.8
ML_RANDOM_STATE = 42

# Performance Benchmarks
BENCHMARKS = {
    'monthly_roi': {
        'thresholds': [1, 2, 5],
        'ratings': ['Poor', 'Average', 'Good', 'Great'],
        'colors': ['#FF4444', '#FFA500', '#32CD32', '#00FF00']
    },
    'win_rate': {
        'thresholds': [45, 60, 75],  # Updated thresholds based on research (increasing from [40,55,65])
        'ratings': ['Poor', 'Average', 'Good', 'Great'],
        'colors': ['#FF4444', '#FFA500', '#32CD32', '#00FF00']
    },
    'risk_reward_ratio': {
        'thresholds': [1.5, 2.0, 3.0],  # Improved from [1.0,1.5,2.5] for better risk management
        'ratings': ['Poor', 'Average', 'Good', 'Great'],
        'colors': ['#FF4444', '#FFA500', '#32CD32', '#00FF00']
    },
    'sharpe_ratio': {
        'thresholds': [0.5, 1.0, 1.5],
        'ratings': ['Poor', 'Average', 'Good', 'Great'],
        'colors': ['#FF4444', '#FFA500', '#32CD32', '#00FF00']
    },
    'profit_factor': {
        'thresholds': [1.2, 1.5, 2.0],  # Increased from [1.0,1.3,1.8] for higher quality trades
        'ratings': ['Poor', 'Average', 'Good', 'Great'],
        'colors': ['#FF4444', '#FFA500', '#32CD32', '#00FF00']
    }
}

# Chart Settings
CHART_HEIGHT = 500
CHART_TEMPLATE = 'plotly_white'

# File Upload Settings
ALLOWED_FILE_TYPES = ['csv', 'xlsx', 'xls']
MAX_FILE_SIZE = 50  # MB

# Data Sources
STOCK_DATA_SOURCE = 'yfinance'
MARKET_DATA_REFRESH_INTERVAL = 300  # 5 minutes in seconds 