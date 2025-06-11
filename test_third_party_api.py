#!/usr/bin/env python
"""
Test script for the third-party API connection at free.v36.cm
"""
import os
import json
import requests
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_third_party_api():
    """Test the connection to the third-party API at free.v36.cm"""
    print("\n" + "=" * 80)
    print(" THIRD-PARTY API TEST (free.v36.cm) ")
    print("=" * 80)
    
    api_key = os.getenv('OPENAI_API_KEY', '')
    api_url = os.getenv('OPENAI_API_URL', '')
    
    if not api_key:
        print("❌ API_KEY is not set in .env file")
        return False
    else:
        masked_key = api_key[:4] + "*" * 20 + api_key[-4:] if len(api_key) > 8 else "***"
        print(f"✅ API_KEY is set: {masked_key}")
    
    if not api_url:
        print("❌ API_URL is not set in .env file")
        return False
    else:
        print(f"✅ API_URL is set: {api_url}")
    
    # The base URL is returning HTML, let's try the correct endpoints
    print("\n❌ The base URL is returning HTML, not a valid API endpoint")
    print("The correct format for the third-party API endpoint should be:")
    print("https://free.v36.cm/v1/chat/completions")
    print("\nTrying with the correct endpoint format...")
    
    # Test different endpoint variations
    endpoints = [
        "https://free.v36.cm/v1/chat/completions",
        "https://api.free.v36.cm/v1/chat/completions",
        "https://free.v36.cm/api/chat/completions"
    ]
    
    # Standard OpenAI format
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'API test successful' if you can read this message."}
        ],
        "temperature": 0.7,
        "max_tokens": 50
    }
    
    for endpoint in endpoints:
        print(f"\nTrying endpoint: {endpoint}")
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            # Record start time
            start_time = time.time()
            
            # Make request
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Check status code
            print(f"Response status code: {response.status_code}")
            print(f"Response time: {response_time:.2f}s")
            
            if response.status_code == 200:
                print("Response content:")
                print("-" * 40)
                print(response.text[:500])
                print("-" * 40)
                
                try:
                    result = response.json()
                    print("Parsed JSON response:")
                    print("-" * 40)
                    print(json.dumps(result, indent=2)[:500])
                    print("-" * 40)
                    
                    # Try to identify the structure
                    if 'choices' in result and result['choices']:
                        print("✅ Response contains 'choices' field (OpenAI format)")
                        if 'message' in result['choices'][0] and 'content' in result['choices'][0]['message']:
                            print(f"✅ Content: {result['choices'][0]['message']['content']}")
                    elif 'response' in result:
                        print("✅ Response contains 'response' field (alternative format)")
                        print(f"✅ Content: {result['response']}")
                    elif 'content' in result:
                        print("✅ Response contains 'content' field (alternative format)")
                        print(f"✅ Content: {result['content']}")
                    else:
                        print("❌ Could not identify a valid response structure")
                        print("Response keys:", list(result.keys()))
                    
                except json.JSONDecodeError:
                    print("❌ Response is not valid JSON")
            else:
                print(f"❌ API returned error status code {response.status_code}")
                print(f"Error details: {response.text[:200]}")
        
        except requests.exceptions.RequestException as e:
            print(f"❌ API connection failed: {str(e)}")
        
        print("-" * 80)
    
    print("\nRecommendations:")
    print("1. Update your .env file to use the correct API endpoint URL:")
    print("   OPENAI_API_URL=https://free.v36.cm/v1/chat/completions")
    print("2. Make sure your API key is valid for this service")
    print("3. The third-party API should mimic the OpenAI API format")

if __name__ == "__main__":
    test_third_party_api() 