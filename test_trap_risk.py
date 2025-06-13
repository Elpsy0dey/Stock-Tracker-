"""
Test script for the trap risk indicator function
"""

from models.stock_screener import StockScreener
import pandas as pd
import numpy as np

def get_trap_risk_indicator(result):
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

def main():
    # Create a stock screener
    screener = StockScreener()
    
    # Set a small universe for testing
    screener.set_stock_universe(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'])
    
    # Get screening results
    results = screener.screen_swing_opportunities(5)
    
    print(f"Found {len(results)} screening results")
    
    # Test trap risk indicator for each result
    for i, result in enumerate(results):
        print(f"\nStock {i+1}: {result['symbol']}")
        
        # Print V4.0 indicators
        v4_indicators = {k: v for k, v in result.items() if k in [
            'Bull_Trap', 'Bear_Trap', 'False_Breakout', 'Volume_Price_Divergence', 
            'HFT_Activity', 'Stop_Hunting', 'Volume_Delta'
        ]}
        print("V4.0 indicators:", v4_indicators)
        
        # Get trap risk
        trap_risk = get_trap_risk_indicator(result)
        print(f"Trap Risk: {trap_risk}")
        
        # Create test cases with different risk levels
        test_cases = [
            ("No risk factors", {**result, 'Bull_Trap': 0, 'Bear_Trap': 0, 'False_Breakout': 0, 'Volume_Price_Divergence': 0, 'HFT_Activity': 0.0, 'Stop_Hunting': 0}),
            ("Medium risk (HFT)", {**result, 'Bull_Trap': 0, 'Bear_Trap': 0, 'False_Breakout': 0, 'Volume_Price_Divergence': 0, 'HFT_Activity': 0.5, 'Stop_Hunting': 0}),
            ("High risk (Bull trap + HFT)", {**result, 'Bull_Trap': 1, 'Bear_Trap': 0, 'False_Breakout': 0, 'Volume_Price_Divergence': 0, 'HFT_Activity': 0.5, 'Stop_Hunting': 0}),
            ("High risk (multiple factors)", {**result, 'Bull_Trap': 1, 'Bear_Trap': 0, 'False_Breakout': 1, 'Volume_Price_Divergence': 1, 'HFT_Activity': 0.8, 'Stop_Hunting': 0})
        ]
        
        print("\nTest cases:")
        for name, test_case in test_cases:
            print(f"  {name}: {get_trap_risk_indicator(test_case)}")

if __name__ == "__main__":
    main() 