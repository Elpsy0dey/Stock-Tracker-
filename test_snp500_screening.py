#!/usr/bin/env python3
"""
Test S&P 500 Stock Screening Before Deployment
============================================

This script tests the stock screening logic against S&P 500 stocks
to validate that the fixes work correctly before deploying to Streamlit.

Expected Results:
- Should find MSFT, AMZN, TSLA from Quick Test
- Should include JPM, PEP from current results  
- Should have 20-30% success rate
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.stock_screener import StockScreener
from utils.data_utils import get_sp500_tickers

def main():
    print("🧪 S&P 500 SCREENING VALIDATION TEST")
    print("=" * 60)
    
    # Initialize screener with new logic
    screener = StockScreener()
    screener.set_stock_universe()
    
    # Show universe composition
    print(f"📊 Stock Universe: {len(screener.stock_universe)} stocks")
    print(f"📋 First 15: {screener.stock_universe[:15]}")
    
    # Validate priority stocks are included
    priority_check = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'PEP']
    print(f"\n🎯 Priority Stock Validation:")
    for stock in priority_check:
        if stock in screener.stock_universe:
            pos = screener.stock_universe.index(stock) + 1
            print(f"  ✅ {stock} (position #{pos})")
        else:
            print(f"  ❌ {stock} (missing)")
    
    # Run comprehensive test
    print(f"\n🚀 Running comprehensive test on first 20 stocks...")
    test_results = screener.test_snp500_screening(20)
    
    # Display results
    summary = test_results['summary']
    print(f"\n📊 RESULTS:")
    print(f"  • Opportunities found: {summary['opportunities_count']}")
    print(f"  • Success rate: {summary['success_rate']:.1f}%")
    print(f"  • Stocks tested: {test_results['stocks_tested']}")
    
    # Show opportunities
    if test_results['opportunities_found']:
        print(f"\n🎯 OPPORTUNITIES FOUND:")
        for opp in test_results['opportunities_found']:
            signals = [k for k, v in opp['signals'].items() if v]
            print(f"  • {opp['symbol']:<6} ${opp['current_price']:>7.2f} | "
                  f"Strength: {opp['signal_strength']:>3.0f} | "
                  f"RSI: {opp['rsi']:>5.1f} | "
                  f"Signals: {', '.join(signals[:2])}")
    
    # Validation against expected results
    found_symbols = {opp['symbol'] for opp in test_results['opportunities_found']}
    expected_quick_test = {'MSFT', 'AMZN', 'TSLA'}
    expected_current = {'JPM', 'PEP'}
    
    print(f"\n✅ VALIDATION RESULTS:")
    
    # Quick Test validation
    quick_found = found_symbols.intersection(expected_quick_test)
    if quick_found:
        print(f"  ✅ Quick Test stocks found: {', '.join(quick_found)}")
    else:
        print(f"  ❌ No Quick Test stocks found (expected: {', '.join(expected_quick_test)})")
    
    # Current results validation  
    current_found = found_symbols.intersection(expected_current)
    if current_found:
        print(f"  ✅ Current result stocks found: {', '.join(current_found)}")
    else:
        print(f"  ⚠️ Current result stocks: {', '.join(expected_current)} (may not be in first 20)")
    
    # Overall assessment
    print(f"\n🎯 DEPLOYMENT ASSESSMENT:")
    
    criteria_met = 0
    total_criteria = 4
    
    if summary['opportunities_count'] >= 3:
        print(f"  ✅ Sufficient opportunities found ({summary['opportunities_count']} >= 3)")
        criteria_met += 1
    else:
        print(f"  ❌ Low opportunity count ({summary['opportunities_count']} < 3)")
    
    if summary['success_rate'] >= 15:
        print(f"  ✅ Good success rate ({summary['success_rate']:.1f}% >= 15%)")
        criteria_met += 1
    else:
        print(f"  ❌ Low success rate ({summary['success_rate']:.1f}% < 15%)")
    
    if len(quick_found) >= 1:
        print(f"  ✅ Quick Test validation passed ({len(quick_found)}/3 stocks found)")
        criteria_met += 1
    else:
        print(f"  ❌ Quick Test validation failed (0/3 stocks found)")
    
    if test_results['summary']['error_count'] <= 2:
        print(f"  ✅ Low error rate ({test_results['summary']['error_count']} <= 2)")
        criteria_met += 1
    else:
        print(f"  ❌ High error rate ({test_results['summary']['error_count']} > 2)")
    
    # Final verdict
    print(f"\n🚀 FINAL VERDICT:")
    if criteria_met >= 3:
        print(f"✅ READY TO DEPLOY ({criteria_met}/{total_criteria} criteria met)")
        print("   The screening logic should work correctly in Streamlit")
    else:
        print(f"⚠️ NEEDS REVIEW ({criteria_met}/{total_criteria} criteria met)")
        print("   Consider adjusting screening parameters before deployment")
    
    return criteria_met >= 3

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 