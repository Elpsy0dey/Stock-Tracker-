#!/usr/bin/env python
"""
Debug script for investigating API authentication issues with free.v36.cm
"""
import os
import json
import requests
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def debug_api_auth():
    """Debug API authentication issues with detailed request and response info"""
    print("\n" + "=" * 80)
    print(" API AUTHENTICATION DEBUGGER ")
    print("=" * 80)
    
    # Load API key
    api_key = os.getenv('OPENAI_API_KEY', '')
    if not api_key:
        print("❌ No API key found in .env file")
        return
    
    # Show masked key
    masked_key = api_key[:4] + "*" * 20 + api_key[-4:] if len(api_key) > 8 else "***"
    print(f"API Key: {masked_key}")
    
    # API endpoint
    api_url = "https://free.v36.cm/v1/chat/completions"
    print(f"API URL: {api_url}")
    
    # Test different authorization formats
    auth_formats = [
        {"name": "Bearer token", "auth": f"Bearer {api_key}"},
        {"name": "Key only", "auth": f"{api_key}"},
        {"name": "No Bearer prefix", "auth": f"Token {api_key}"},
        {"name": "sk- prefix check", 
         "auth": f"Bearer {'sk-' + api_key.replace('sk-', '')}" if not api_key.startswith('sk-') else f"Bearer {api_key}"}
    ]
    
    # Standard payload
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'API key is valid' if you can read this message."}
        ],
        "temperature": 0.7,
        "max_tokens": 50
    }
    
    # Try different authorization formats
    print("\nTesting different authorization formats:")
    for auth_format in auth_formats:
        print(f"\n[Testing {auth_format['name']}]")
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": auth_format['auth']
        }
        
        print(f"Headers: {json.dumps(headers)}")
        print(f"Payload: {json.dumps(payload)[:100]}...")
        
        try:
            # Make request and time it
            start_time = time.time()
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response_time = time.time() - start_time
            
            # Print response details
            print(f"Response time: {response_time:.2f}s")
            print(f"Status code: {response.status_code}")
            print(f"Response headers: {json.dumps(dict(response.headers))}")
            
            # Check response content
            print("Response content:")
            print("-" * 40)
            print(response.text[:500])
            print("-" * 40)
            
            # Parse JSON if possible
            try:
                result = response.json()
                if 'choices' in result and result['choices']:
                    content = result['choices'][0]['message']['content']
                    print(f"Content: {content}")
                    if "unauthorized" not in content.lower():
                        print("✅ This authorization format works!")
            except Exception as e:
                print(f"Error parsing JSON: {str(e)}")
        
        except Exception as e:
            print(f"Request failed: {str(e)}")
        
        print("-" * 80)
    
    # Check registration requirements
    print("\nThe free.v36.cm service might require:")
    print("1. Registration on their website")
    print("2. A specific API key format")
    print("3. Additional headers or parameters")
    print("4. IP-based restrictions")
    
    print("\nSuggested next steps:")
    print("1. Visit https://free.v36.cm to check registration requirements")
    print("2. Look for API documentation on their website")
    print("3. Check if they require a different API key format")
    print("4. Consider using the official OpenAI API if this service doesn't work")

def test_response_parsing():
    """Test parsing of different response formats"""
    print("\n" + "=" * 80)
    print(" RESPONSE PARSING TEST ")
    print("=" * 80)
    
    # Sample response formats to test
    sample_responses = [
        # Standard OpenAI format
        {
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "created": 1677858242,
            "model": "gpt-3.5-turbo",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Test message content"
                    },
                    "finish_reason": "stop",
                    "index": 0
                }
            ]
        },
        # Alternative format with response field
        {
            "response": "Test alternative format content",
            "status": "success"
        },
        # Simple content format
        {
            "content": "Test simple content format",
            "status": "success"
        },
        # Format with error
        {
            "error": {
                "message": "Unauthorized request",
                "type": "unauthorized"
            }
        }
    ]
    
    print("Testing response parsing for different formats:")
    for i, response in enumerate(sample_responses):
        print(f"\n[Format #{i+1}]")
        print(f"Response: {json.dumps(response)}")
        
        try:
            # Standard OpenAI format
            if 'choices' in response and response['choices']:
                if 'message' in response['choices'][0] and 'content' in response['choices'][0]['message']:
                    content = response['choices'][0]['message']['content']
                    print(f"✅ Extracted content (OpenAI format): {content}")
                    continue
            
            # Alternative format with response field
            if 'response' in response:
                content = response['response']
                print(f"✅ Extracted content (response field): {content}")
                continue
                
            # Simple content format
            if 'content' in response:
                content = response['content']
                print(f"✅ Extracted content (content field): {content}")
                continue
                
            # Check for error
            if 'error' in response:
                error_msg = response['error'].get('message', 'Unknown error')
                print(f"❌ Error detected: {error_msg}")
                continue
                
            print("❓ Unknown format, couldn't extract content")
            
        except Exception as e:
            print(f"Error parsing: {str(e)}")

def check_api_restrictions():
    """Check if the API has rate limits or other restrictions"""
    print("\n" + "=" * 80)
    print(" API RESTRICTIONS CHECK ")
    print("=" * 80)
    
    api_url = "https://free.v36.cm/v1/chat/completions"
    print(f"Checking restrictions for: {api_url}")
    
    # Make a simple OPTIONS request to check for headers
    try:
        options_response = requests.options(api_url)
        print(f"OPTIONS response code: {options_response.status_code}")
        print(f"Allow header: {options_response.headers.get('Allow', 'Not provided')}")
        print(f"Access-Control-Allow headers: {options_response.headers.get('Access-Control-Allow-Headers', 'Not provided')}")
        print(f"Access-Control-Allow methods: {options_response.headers.get('Access-Control-Allow-Methods', 'Not provided')}")
    except Exception as e:
        print(f"OPTIONS request failed: {str(e)}")
    
    # Try to get the API endpoint documentation or info
    try:
        info_url = "https://free.v36.cm/v1"
        print(f"\nTrying to get API info from: {info_url}")
        info_response = requests.get(info_url)
        print(f"Response code: {info_response.status_code}")
        print(f"Response: {info_response.text[:500]}")
    except Exception as e:
        print(f"Info request failed: {str(e)}")

if __name__ == "__main__":
    debug_api_auth()
    test_response_parsing()
    check_api_restrictions() 