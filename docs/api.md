# API Documentation

## Overview
This document provides detailed information about the Stock Tracker application's API endpoints, data structures, and integration points.

## External APIs

### Yahoo Finance API
The application uses Yahoo Finance API for market data.

#### Endpoints
- Stock Data: `yfinance.Ticker(symbol).history()`
- Company Info: `yfinance.Ticker(symbol).info`
- Real-time Quotes: `yfinance.Ticker(symbol).fast_info`

#### Rate Limits
- Standard rate limits apply
- Implement caching to minimize API calls
- Handle rate limit errors gracefully

#### Example: Yahoo Finance Integration
```python
# utils/yahoo_finance.py
import yfinance as yf
from typing import Dict, Optional
import pandas as pd

class YahooFinanceAPI:
    def __init__(self, cache_duration: int = 300):
        self.cache_duration = cache_duration
        self._cache = {}
        
    def get_stock_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Get historical stock data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data
        except Exception as e:
            raise APIError(f"Error fetching stock data: {str(e)}")
            
    def get_company_info(self, symbol: str) -> Dict:
        """Get company information"""
        try:
            ticker = yf.Ticker(symbol)
            return ticker.info
        except Exception as e:
            raise APIError(f"Error fetching company info: {str(e)}")
            
    def get_realtime_quote(self, symbol: str) -> Dict:
        """Get real-time quote"""
        try:
            ticker = yf.Ticker(symbol)
            return ticker.fast_info
        except Exception as e:
            raise APIError(f"Error fetching real-time quote: {str(e)}")

### Alpha Vantage API (Optional)
```python
# utils/alpha_vantage.py
import requests
from typing import Dict, Optional
import os

class AlphaVantageAPI:
    def __init__(self):
        self.api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.base_url = "https://www.alphavantage.co/query"
        
    def get_technical_indicators(self, symbol: str, indicator: str) -> Dict:
        """Get technical indicators"""
        params = {
            'function': indicator,
            'symbol': symbol,
            'apikey': self.api_key
        }
        response = requests.get(self.base_url, params=params)
        return response.json()

## Internal APIs

### Portfolio Management

#### PortfolioTracker Class
```python
class PortfolioTracker:
    def load_trades(self, trades_df: pd.DataFrame) -> bool
    def refresh_current_prices(self) -> None
    def get_summary_stats(self) -> Dict
    def calculate_trade_kpis(self) -> Dict
```

#### Methods
1. `load_trades(trades_df)`
   - Loads trading history
   - Parameters: DataFrame with trade data
   - Returns: Success status

2. `refresh_current_prices()`
   - Updates current market prices
   - No parameters
   - Returns: None

3. `get_summary_stats()`
   - Gets portfolio statistics
   - No parameters
   - Returns: Dictionary with stats

4. `calculate_trade_kpis()`
   - Calculates trading performance metrics
   - No parameters
   - Returns: Dictionary with KPIs

#### Example: Portfolio Tracker Implementation
```python
# models/portfolio_tracker.py
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime

class PortfolioTracker:
    def __init__(self, starting_cash: float = 100000.0):
        self.starting_cash = starting_cash
        self.trade_history = []
        self.current_positions = {}
        
    def add_trade(self, trade: Dict) -> bool:
        """Add a new trade to history"""
        try:
            # Validate trade data
            self._validate_trade(trade)
            
            # Calculate P&L
            trade['pnl'] = self._calculate_pnl(trade)
            trade['pnl_pct'] = self._calculate_pnl_pct(trade)
            
            # Add to history
            self.trade_history.append(trade)
            
            # Update positions
            self._update_positions(trade)
            
            return True
        except Exception as e:
            raise DataError(f"Error adding trade: {str(e)}")
            
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary"""
        return {
            'total_value': self._calculate_total_value(),
            'cash_balance': self._calculate_cash_balance(),
            'positions': self.current_positions,
            'total_pnl': self._calculate_total_pnl(),
            'total_pnl_pct': self._calculate_total_pnl_pct()
        }

### Technical Analysis

#### TechnicalAnalyzer Class
```python
class TechnicalAnalyzer:
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame
    def get_signal_strength(self, df: pd.DataFrame) -> Dict
    def get_swing_signals(self, df: pd.DataFrame) -> Dict
    def get_breakout_signals(self, df: pd.DataFrame) -> Dict
```

#### Methods
1. `calculate_all_indicators(df)`
   - Calculates technical indicators
   - Parameters: Price DataFrame
   - Returns: DataFrame with indicators

2. `get_signal_strength(df)`
   - Calculates signal strength
   - Parameters: Price DataFrame
   - Returns: Dictionary with strength metrics

3. `get_swing_signals(df)`
   - Gets swing trading signals
   - Parameters: Price DataFrame
   - Returns: Dictionary with signals

4. `get_breakout_signals(df)`
   - Gets breakout signals
   - Parameters: Price DataFrame
   - Returns: Dictionary with signals

#### Example: Technical Analyzer Implementation
```python
# models/technical_analysis.py
import pandas as pd
import numpy as np
from typing import Dict, List

class TechnicalAnalyzer:
    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def calculate_macd(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate MACD indicator"""
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return {
            'macd': macd,
            'signal': signal,
            'histogram': macd - signal
        }
```

### Stock Screening

#### StockScreener Class
```python
class StockScreener:
    def set_stock_universe(self, symbols: List[str] = None) -> None
    def screen_swing_opportunities(self, max_results: int) -> List[Dict]
    def screen_breakout_opportunities(self, max_results: int) -> List[Dict]
```

#### Methods
1. `set_stock_universe(symbols)`
   - Sets stock universe for screening
   - Parameters: List of symbols (optional)
   - Returns: None

2. `screen_swing_opportunities(max_results)`
   - Screens for swing trades
   - Parameters: Maximum results to return
   - Returns: List of opportunities

3. `screen_breakout_opportunities(max_results)`
   - Screens for breakout trades
   - Parameters: Maximum results to return
   - Returns: List of opportunities

#### Example: Stock Screener Implementation
```python
# models/stock_screener.py
from typing import List, Dict
import pandas as pd
from .technical_analysis import TechnicalAnalyzer

class StockScreener:
    def __init__(self):
        self.technical_analyzer = TechnicalAnalyzer()
        self.stock_universe = []
        
    def screen_swing_trades(self, criteria: Dict) -> List[Dict]:
        """Screen for swing trading opportunities"""
        opportunities = []
        for symbol in self.stock_universe:
            try:
                # Get stock data
                data = self._get_stock_data(symbol)
                
                # Calculate indicators
                indicators = self._calculate_indicators(data)
                
                # Check criteria
                if self._meets_criteria(indicators, criteria):
                    opportunities.append({
                        'symbol': symbol,
                        'indicators': indicators,
                        'setup_type': self._identify_setup(indicators)
                    })
            except Exception as e:
                print(f"Error screening {symbol}: {str(e)}")
                
        return opportunities

### AI Service

#### AIService Class
```python
class AIService:
    def generate_trading_suggestions(self, data: Dict) -> str
    def generate_performance_analysis(self, data: Dict) -> str
    def generate_short_term_decision(self, data: Dict) -> Dict
```

#### Methods
1. `generate_trading_suggestions(data)`
   - Generates trading suggestions
   - Parameters: Technical data dictionary
   - Returns: Formatted suggestions string

2. `generate_performance_analysis(data)`
   - Analyzes trading performance
   - Parameters: Performance data dictionary
   - Returns: Analysis string

3. `generate_short_term_decision(data)`
   - Generates short-term trading decision
   - Parameters: Technical data dictionary
   - Returns: Decision dictionary

## Data Structures

### Trade Data
```python
{
    'symbol': str,
    'entry_date': datetime,
    'exit_date': datetime,
    'entry_price': float,
    'exit_price': float,
    'shares': int,
    'pnl': float,
    'pnl_pct': float,
    'hold_time': int,
    'fees': float
}
```

### Portfolio Stats
```python
{
    'total_account_value': float,
    'current_cash': float,
    'portfolio_value': float,
    'total_pnl': float,
    'total_return_pct': float,
    'unrealized_pnl': float,
    'portfolio_details': Dict
}
```

### Technical Indicators
```python
{
    'RSI': float,
    'MACD': float,
    'MACD_Signal': float,
    'Stoch_K': float,
    'Stoch_D': float,
    'ADX': float,
    'BB_Position': float,
    'ATR': float,
    'SMA_20': float,
    'SMA_50': float,
    'SMA_200': float
}
```

### Example: Trade Data Model
```python
# models/trade.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class Trade:
    symbol: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    hold_time: int
    fees: float
    notes: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'entry_date': self.entry_date.isoformat(),
            'exit_date': self.exit_date.isoformat(),
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'shares': self.shares,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'hold_time': self.hold_time,
            'fees': self.fees,
            'notes': self.notes
        }
```

### Example: Portfolio Position Model
```python
# models/position.py
from dataclasses import dataclass
from typing import List
from .trade import Trade

@dataclass
class Position:
    symbol: str
    shares: int
    avg_cost: float
    current_price: float
    trades: List[Trade]
    
    @property
    def market_value(self) -> float:
        """Calculate current market value"""
        return self.shares * self.current_price
        
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L"""
        return self.market_value - (self.shares * self.avg_cost)
```

## Error Handling

### Error Codes
- `API_ERROR`: External API errors
- `DATA_ERROR`: Data processing errors
- `VALIDATION_ERROR`: Input validation errors
- `CALCULATION_ERROR`: Calculation errors

### Error Response Format
```python
{
    'error': str,
    'code': str,
    'message': str,
    'details': Dict
}
```

### Example: API Error Handler
```python
# utils/api_error_handler.py
from functools import wraps
import requests
from typing import Callable, Any

def handle_api_errors(func: Callable) -> Callable:
    """Decorator for handling API errors"""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except requests.exceptions.RequestException as e:
            raise APIError(f"API request failed: {str(e)}")
        except ValueError as e:
            raise ValidationError(f"Invalid data: {str(e)}")
        except Exception as e:
            raise APIError(f"Unexpected error: {str(e)}")
    return wrapper
```

## Rate Limiting

### Internal Rate Limits
- Maximum 100 API calls per minute
- Cache results for 5 minutes
- Implement exponential backoff

### External Rate Limits
- Follow Yahoo Finance API limits
- Implement request queuing
- Use caching strategies

### Example: Rate Limiter Implementation
```python
# utils/rate_limiter.py
from datetime import datetime, timedelta
from typing import Dict, Optional
import time

class RateLimiter:
    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.calls: Dict[str, List[datetime]] = {}
        
    def wait_if_needed(self, key: str) -> None:
        """Wait if rate limit is reached"""
        now = datetime.now()
        if key not in self.calls:
            self.calls[key] = []
            
        # Remove old calls
        self.calls[key] = [t for t in self.calls[key] 
                          if now - t < timedelta(minutes=1)]
        
        # Check if we need to wait
        if len(self.calls[key]) >= self.calls_per_minute:
            sleep_time = (self.calls[key][0] + 
                         timedelta(minutes=1) - now).total_seconds()
            if sleep_time > 0:
                time.sleep(sleep_time)
                
        # Add current call
        self.calls[key].append(now)
```

## Security

### Authentication
- API key required for external APIs
- Session-based authentication for web interface
- Token-based authentication for API access

### Data Protection
- Encrypt sensitive data
- Validate all inputs
- Sanitize outputs
- Implement rate limiting

## Best Practices

### API Usage
1. Use appropriate error handling
2. Implement retry mechanisms
3. Cache responses when possible
4. Monitor API usage

### Data Processing
1. Validate input data
2. Handle missing values
3. Implement data cleaning
4. Use appropriate data types

### Performance
1. Implement caching
2. Use efficient algorithms
3. Optimize database queries
4. Monitor resource usage 