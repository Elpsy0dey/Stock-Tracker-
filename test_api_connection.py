"""
Test script to verify AI API connection with the fixed code
"""

import sys
import os
import json
from dotenv import load_dotenv
from config.api_config import API_CONFIG
from services.ai_service import AIService, FALLBACK_MODELS

# Load environment variables
load_dotenv()

def test_api_connection():
    """Test the AI API connection with the updated code"""
    print("\n===== TESTING AI API CONNECTION =====")
    
    # Print configuration
    print("\nCurrent API Configuration:")
    print(f"API URL: {API_CONFIG['API_URL']}")
    masked_key = API_CONFIG['API_KEY'][:4] + "*" * 10 + API_CONFIG['API_KEY'][-4:] if len(API_CONFIG['API_KEY']) > 8 else "***"
    print(f"API KEY: {masked_key}")
    print(f"Primary Model: {API_CONFIG['MODEL']}")
    print(f"Fallback Models: {', '.join(FALLBACK_MODELS)}")
    
    # Create sample technical data for testing
    test_data = {
        "symbol": "AAPL",
        "price": 180.5,
        "rsi": 55.2,
        "macd": {
            "value": 2.1,
            "signal": 1.8,
            "histogram": 0.3
        },
        "patterns": ["Double bottom", "Golden cross"],
        "volume": 45000000,
        "volume_change": 15.2,
        "atr": 3.2,
        "adx": 24.5,
        "bb_position": 0.6
    }
    
    # Clear cache to ensure a fresh test
    print("\nClearing cache for fresh test...")
    AIService.clear_cache()
    
    # Test trading suggestions
    print("\nTesting trading suggestions generation...")
    try:
        results = AIService.generate_trading_suggestions(test_data)
        print("\n--- Trading Suggestions Results ---")
        print(results[:300] + "..." if len(results) > 300 else results)
        print("\n✅ Trading suggestions API test completed successfully!")
    except Exception as e:
        print(f"\n❌ Error in trading suggestions API test: {str(e)}")
    
    # Test performance analysis
    print("\nTesting performance analysis generation...")
    performance_data = {
        "metrics": {
            "total_return": 15.2,
            "win_rate": 65.5,
            "profit_factor": 2.1,
            "max_drawdown": 12.5,
            "sharpe_ratio": 1.2
        },
        "time_period": "3 months"
    }
    
    try:
        results = AIService.generate_performance_analysis(performance_data)
        print("\n--- Performance Analysis Results ---")
        print(results[:300] + "..." if len(results) > 300 else results)
        print("\n✅ Performance analysis API test completed successfully!")
    except Exception as e:
        print(f"\n❌ Error in performance analysis API test: {str(e)}")
    
    print("\n===== API CONNECTION TEST COMPLETE =====")

if __name__ == "__main__":
    test_api_connection() 