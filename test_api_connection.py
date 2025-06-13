"""
Test script to verify API connection and model selection
"""
import os
import time
import requests
import json
from config.api_config import API_CONFIG
from services.ai_service import AIService, FALLBACK_MODELS
from services.model_manager import ModelManager, PREFERRED_MODELS

def test_api_connection():
    """Test the API connection and model selection"""
    print("\n=== API Configuration ===")
    print(f"API URL: {API_CONFIG['API_URL']}")
    print(f"Default Model: {API_CONFIG['MODEL']}")
    print(f"Preferred Models: {', '.join(PREFERRED_MODELS)}")
    print(f"Fallback Models: {', '.join(FALLBACK_MODELS)}")
    
    # Initialize the model manager
    print("\n=== Testing Model Manager ===")
    selected_model = ModelManager.initialize()
    print(f"Selected model: {selected_model}")
    
    # Test a simple API call
    print("\n=== Testing API Call ===")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_CONFIG['API_KEY']}"
    }
    
    payload = {
        "model": API_CONFIG['MODEL'],
        "messages": [
            {"role": "user", "content": "What model are you?"}
        ],
        "max_tokens": 50
    }
    
    try:
        print(f"Making API request to {API_CONFIG['API_URL']} with model {API_CONFIG['MODEL']}")
        response = requests.post(
            API_CONFIG['API_URL'],
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("API call successful!")
            
            # Try to extract model information
            if 'model' in result:
                print(f"Model used: {result['model']}")
            
            # Extract response content
            content = None
            if 'choices' in result and result['choices']:
                if 'message' in result['choices'][0] and 'content' in result['choices'][0]['message']:
                    content = result['choices'][0]['message']['content']
                elif 'content' in result['choices'][0]:
                    content = result['choices'][0]['content']
            
            if content:
                print(f"Response: {content}")
            else:
                print("Could not extract response content")
                print(f"Raw response: {json.dumps(result, indent=2)[:500]}")
        else:
            print(f"API call failed with status {response.status_code}")
            print(f"Response: {response.text[:500]}")
    
    except Exception as e:
        print(f"Error testing API connection: {str(e)}")
    
    # Test AIService
    print("\n=== Testing AIService ===")
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
    
    try:
        # Clear cache to ensure fresh request
        AIService.clear_cache()
        
        # Generate trading suggestions
        print("Generating trading suggestions...")
        suggestions = AIService.generate_trading_suggestions(test_data)
        
        # Print a preview of the suggestions
        preview = suggestions[:200] + "..." if len(suggestions) > 200 else suggestions
        print(f"Generated suggestions: {preview}")
        
    except Exception as e:
        print(f"Error testing AIService: {str(e)}")

if __name__ == "__main__":
    test_api_connection() 