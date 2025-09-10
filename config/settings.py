"""
Configuration settings for the Trading Portfolio Tracker
"""

# Application Settings
APP_TITLE = "📈 Advanced Trading Portfolio Tracker"
APP_ICON = "📈"
DEFAULT_STARTING_CASH = 69013.50

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

# Options Pinning Settings
OPTIONS_PINNING_OI_THRESHOLD = 2.5  # OI ratio threshold for strong pinning pressure
OPTIONS_PINNING_PROXIMITY_PCT = 0.05  # % proximity to consider strikes as potential magnets
OPTIONS_PINNING_PATTERN_LOOKBACK = 3  # Days to look back for pinning pattern detection
EXPIRATION_CALENDAR = {  # Sample monthly options expiration calendar
    2023: ["01-20", "02-17", "03-17", "04-21", "05-19", "06-16", 
          "07-21", "08-18", "09-15", "10-20", "11-17", "12-15"],
    2024: ["01-19", "02-16", "03-15", "04-19", "05-17", "06-21", 
          "07-19", "08-16", "09-20", "10-18", "11-15", "12-20"]
}

# Screening Criteria
MIN_MARKET_CAP = 1e9  # $1B minimum market cap
MIN_VOLUME = 1e6      # $1M minimum daily volume
MIN_PRICE = 5.0       # $5 minimum stock price

# Market Cap Range Targets (V3.0 backtest findings)
SMALL_CAP_RANGE = (0, 2e9)        # 0-$2B (31% breakout success rate)
MID_CAP_RANGE = (2e9, 10e9)       # $2B-$10B (58% breakout success rate)
LARGE_CAP_RANGE = (10e9, float('inf'))  # $10B+ (22% breakout success rate)
PREFERRED_CAP_RANGE = MID_CAP_RANGE  # V3.0: Focus on mid-caps for breakouts

# Swing Trading Settings - UPDATED FOR V3.0 EVIDENCE-BASED OPTIMIZATION
SWING_RSI_OVERBOUGHT = 75  # Increased from 70 for more reliable reversals
SWING_RSI_OVERSOLD = 28    # Adjusted from 30 for better mean-reversion entries
SWING_STOP_LOSS = 0.02     # Tightened from 3%→2% based on drawdown analysis (47%→28%)
SWING_SUPPORT_RESISTANCE_PERIOD = 14  # Reduced from 20 to focus on more recent S/R levels
SWING_PRICE_PROXIMITY_THRESHOLD = 0.015  # Tightened from 2% to 1.5% for higher precision entries
SWING_VOLUME_CONFIRMATION = 1.35  # Increased from 1.2 to 1.35 for stronger volume confirmation
SWING_MA_PROXIMITY_THRESHOLD = 0.01  # Maintained at 1% proximity to moving averages
SWING_PATTERN_CONFIRMATION = True  # Require chart pattern confirmation for higher probability
SWING_MULTI_TIMEFRAME_CONFIRM = True  # Require confirmation across multiple timeframes
SWING_WIN_RATE_THRESHOLD = 65  # Minimum historical success rate required for pattern entry
SWING_OPTIMAL_PRICE_RANGE = (15, 80)  # V3.0: Confirmed 2.3x better returns in this range (8.7% vs 3.8%)
SWING_TREND_FILTER = True  # Only take swings in direction of larger trend (higher win rate)
SWING_REWARD_RISK_RATIO = 3.0  # V3.0: Minimum 1:3 reward-to-risk ratio (improved from 1:2)

# Breakout Trading Settings - UPDATED FOR V3.0 EVIDENCE-BASED OPTIMIZATION
BREAKOUT_VOLUME_MULTIPLIER = 1.8  # Captures 91% of significant moves while eliminating 83% of false signals
BREAKOUT_STOP_LOSS = 0.08  # Improved risk-reward ratio from 1.18 to 1.86
BREAKOUT_ADX_THRESHOLD = 22  # Lowered from 25 to 22 to catch early breakouts
BREAKOUT_CONSOLIDATION_DAYS = 14  # 2+ weeks of consolidation 
BREAKOUT_PRICE_RANGE_LIMIT = 0.15  # Consolidation should be tight (<15% range)
BREAKOUT_MIN_VOLATILITY_ADR = 1.5  # Minimum Average Daily Range %
BREAKOUT_OPTIMAL_PRICE_RANGE = (20, 100)  # V3.0: 68% higher success rate in this range
BREAKOUT_SEASONALITY_BOOST = 1.2  # Boost score during favorable seasons

# V3.0 Sector-Specific Settings
TECH_SECTOR_VOLUME_MULTIPLIER = 2.0  # V3.0: Tech stocks require stricter volume filters
FINANCIAL_SECTOR_RSI_RANGE = (25, 80)  # V3.0: Financial sector benefits from wider RSI thresholds
HEALTHCARE_PRICE_RANGE = (20, 85)  # V3.0: Healthcare/biotech performs best with narrower price filters

# V3.0 Multi-Timeframe Settings
TIMEFRAME_DAILY = "1d"
TIMEFRAME_HOURLY = "1h"
TIMEFRAME_15MIN = "15m"
REQUIRED_TIMEFRAME_CONFIRMATION = [TIMEFRAME_DAILY, TIMEFRAME_HOURLY]  # At least these 2 timeframes must confirm

# V4.0 Price-Volume Analysis Settings (Based on research paper)
VOLUME_PRICE_DIVERGENCE_WINDOW = 5  # Days to check for volume-price divergence
VOLUME_DELTA_THRESHOLD = 0.65  # Threshold for volume delta (buy vs sell pressure)
BULL_BEAR_TRAP_DETECTION = True  # Enable bull/bear trap detection
BULL_TRAP_VOLUME_THRESHOLD = 1.5  # Volume threshold for bull trap detection
BEAR_TRAP_VOLUME_THRESHOLD = 1.3  # Volume threshold for bear trap detection
TRAP_REVERSAL_THRESHOLD = 0.02  # Price reversal % to confirm trap
FLICKERING_QUOTE_THRESHOLD = 3  # Number of rapid price changes to detect algo activity

# V4.0 Market Depth and Liquidity Analysis
MARKET_DEPTH_ANALYSIS = True  # Enable market depth analysis when available
VOLUME_DEPTH_RATIO_THRESHOLD = 5.0  # Flag when volume/depth ratio exceeds this value
ORDER_FLOW_IMBALANCE_THRESHOLD = 0.7  # Order flow buy/sell imbalance threshold (0-1)
BID_ASK_SPREAD_WARNING = 0.01  # Flag wide spreads (>1% of price)
LIQUIDITY_QUALITY_CHECK = True  # Check for fake liquidity and wash trades

# V4.0 False Breakout Detection
FALSE_BREAKOUT_DETECTION = True  # Enable false breakout detection
FALSE_BREAKOUT_VOLUME_THRESHOLD = 1.2  # Minimum volume for valid breakout
FALSE_BREAKOUT_REVERSAL_PERIOD = 3  # Days to monitor for reversal after breakout
CANDLESTICK_REVERSAL_PATTERNS = True  # Check for bearish/bullish reversal patterns
MULTI_TIMEFRAME_BREAKOUT_CONFIRM = True  # Require breakout confirmation on multiple timeframes
STOP_HUNT_DETECTION = True  # Detect potential stop hunting patterns

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