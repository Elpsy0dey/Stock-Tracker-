"""
Test script to verify that gpt-4o-mini is being used as the default model
"""
import os
import time
from utils.startup import initialize_application
from config.api_config import API_CONFIG
from services.ai_service import AIService

def test_forced_model():
    """Test that gpt-4o-mini is being used as the default model"""
    print("\n=== Testing Forced Model Configuration ===")
    
    # Step 1: Initialize the application
    print("\nInitializing application...")
    initialize_application()
    
    # Step 2: Check the configured model
    print(f"\nConfigured model after initialization: {API_CONFIG['MODEL']}")
    
    # Step 3: Test a simple API call with AIService
    print("\nTesting AIService with technical data...")
    
    # Create test technical data
    test_data = {
        'price': 150.0,
        'rsi': 45.0,
        'macd': {'value': 2.5, 'signal': 1.8},
        'stochastic': {'k': 65.0, 'd': 60.0},
        'adx': 22.0,
        'volume_ratio': 1.2,
        'bb_position': 0.5,
        'atr': 3.5,
        'patterns': [],
        'sma_20': 148.0,
        'sma_50': 145.0,
        'sma_200': 140.0
    }
    
    # Clear cache to ensure a fresh test
    AIService.clear_cache()
    
    # Generate trading suggestions
    print("Generating trading suggestions...")
    suggestions = AIService.generate_trading_suggestions(test_data)
    
    # Print a preview of the suggestions
    preview = suggestions[:200] + "..." if len(suggestions) > 200 else suggestions
    print(f"\nGenerated suggestions: {preview}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_forced_model() 