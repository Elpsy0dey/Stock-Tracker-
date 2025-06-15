"""
Options Data Service for Trading Portfolio Tracker

Fetches real options chain data for analyzing options pinning effects
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

def get_options_chain(symbol: str) -> Optional[Dict]:
    """
    Get options chain data for a symbol with open interest
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        Dictionary containing expiry date, strikes and open interest
        or None if data cannot be retrieved
    """
    try:
        ticker = yf.Ticker(symbol)
        
        # Get all available expiration dates
        expirations = ticker.options
        
        if not expirations:
            logger.warning(f"No options data available for {symbol}")
            return None
            
        # Sort expirations by date and get the nearest one
        sorted_expirations = sorted(expirations)
        nearest_expiry = sorted_expirations[0]
        
        # Also find monthly expiration if available (typically 3rd Friday)
        # For simplicity, we'll look for dates with day numbers between 15-21
        monthly_expiry = None
        for exp in sorted_expirations:
            exp_date = datetime.strptime(exp, "%Y-%m-%d")
            if 15 <= exp_date.day <= 21 and exp_date.weekday() == 4:  # Friday between 15th-21st
                monthly_expiry = exp
                break
        
        # If no monthly found or if monthly is further, use nearest
        expiry_to_use = monthly_expiry if monthly_expiry else nearest_expiry
        
        # Get chain data
        chain = ticker.option_chain(expiry_to_use)
        
        # Current price for reference
        current_price = ticker.history(period="1d")["Close"].iloc[-1]
        
        # Combine calls and puts OI at each strike
        strikes = {}
        
        # Process calls
        for opt in chain.calls.itertuples():
            if hasattr(opt, "openInterest") and not pd.isna(opt.openInterest):
                strikes[opt.strike] = opt.openInterest
            else:
                strikes[opt.strike] = 0
        
        # Process puts (add to existing strikes)
        for opt in chain.puts.itertuples():
            if opt.strike in strikes:
                if hasattr(opt, "openInterest") and not pd.isna(opt.openInterest):
                    strikes[opt.strike] += opt.openInterest
            else:
                if hasattr(opt, "openInterest") and not pd.isna(opt.openInterest):
                    strikes[opt.strike] = opt.openInterest
                else:
                    strikes[opt.strike] = 0
        
        # Calculate days to expiration
        expiry_date = datetime.strptime(expiry_to_use, "%Y-%m-%d")
        days_to_expiry = (expiry_date - datetime.now()).days + 1
        
        # Filter out strikes with zero OI
        filtered_strikes = {k: v for k, v in strikes.items() if v > 0}
        
        # If too few strikes with OI, include all strikes
        if len(filtered_strikes) < 5:
            filtered_strikes = strikes
        
        return {
            "symbol": symbol,
            "current_price": current_price,
            "expiry_date": expiry_to_use,
            "days_to_expiry": days_to_expiry,
            "is_monthly": monthly_expiry == expiry_to_use,
            "strikes": list(filtered_strikes.keys()),
            "open_interest": filtered_strikes
        }
        
    except Exception as e:
        logger.error(f"Error fetching options data for {symbol}: {str(e)}")
        return None 