#!/usr/bin/env python
"""
Test script for API configuration and connectivity
"""
import os
import json
import requests
import time
from config.api_config import API_CONFIG
from dotenv import load_dotenv

def test_api_configuration():
    """Test the API configuration and connectivity"""
    print("\n" + "=" * 80)
    print(" API CONFIGURATION TEST ")
    print("=" * 80)
    
    # Check if API_KEY is set
    if not API_CONFIG['API_KEY']:
        print("❌ API_KEY is not set in configuration or environment variables")
        print("   Please set OPENAI_API_KEY in your .env file or environment")
        return False
    else:
        masked_key = API_CONFIG['API_KEY'][:4] + "*" * 20 + API_CONFIG['API_KEY'][-4:] if len(API_CONFIG['API_KEY']) > 8 else "***"
        print(f"✅ API_KEY is set: {masked_key}")
    
    # Check if API_URL is set
    if not API_CONFIG['API_URL']:
        print("❌ API_URL is not set in configuration or environment variables")
        print("   Please set OPENAI_API_URL in your .env file or environment")
        return False
    else:
        print(f"✅ API_URL is set: {API_CONFIG['API_URL']}")
    
    # Check other configuration values
    print(f"✅ MODEL: {API_CONFIG['MODEL']}")
    print(f"✅ MAX_TOKENS: {API_CONFIG['MAX_TOKENS']}")
    print(f"✅ TEMPERATURE: {API_CONFIG['TEMPERATURE']}")
    print(f"✅ REQUEST_TIMEOUT: {API_CONFIG['REQUEST_TIMEOUT']} seconds")
    
    # Check API connectivity
    print("\nTesting API connectivity...")
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_CONFIG['API_KEY']}"
        }
        
        payload = {
            "model": API_CONFIG['MODEL'],
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "Say 'API test successful' if you can read this message."
                }
            ],
            "max_tokens": 20
        }
        
        # Record start time
        start_time = time.time()
        
        # Make request
        response = requests.post(
            API_CONFIG['API_URL'],
            headers=headers,
            json=payload,
            timeout=API_CONFIG['REQUEST_TIMEOUT']
        )
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Check status code
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content'] if 'choices' in result else "No content returned"
            print(f"✅ API connection successful! Response time: {response_time:.2f}s")
            print(f"✅ API response: {content}")
            return True
        else:
            print(f"❌ API returned status code {response.status_code}")
            print(f"Error details: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ API connection failed: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Error testing API: {str(e)}")
        return False

def check_env_file():
    """Check if the .env file exists and has required values"""
    print("\n" + "=" * 80)
    print(" ENV FILE CHECK ")
    print("=" * 80)
    
    env_file_path = '.env'
    if not os.path.exists(env_file_path):
        print("❌ .env file does not exist in the project root.")
        print("Creating a sample .env file template...")
        
        with open(env_file_path, 'w') as f:
            f.write("# OpenAI API Configuration\n")
            f.write("OPENAI_API_KEY=your_api_key_here\n")
            f.write("OPENAI_API_URL=https://api.openai.com/v1/chat/completions\n")
        
        print("✅ Created template .env file. Please edit it with your actual API key.")
        return False
    
    # Load .env file
    load_dotenv()
    
    # Check if variables are set
    openai_key = os.getenv('OPENAI_API_KEY')
    openai_url = os.getenv('OPENAI_API_URL')
    
    if not openai_key or openai_key == 'your_api_key_here':
        print("❌ OPENAI_API_KEY not properly set in .env file")
        return False
    
    if not openai_url:
        print("❌ OPENAI_API_URL not set in .env file")
        return False
    
    print("✅ .env file exists and appears to have required values.")
    return True

def main():
    """Main function to run tests"""
    print("\nTesting API Configuration and Connectivity...")
    
    # First check the .env file
    env_ok = check_env_file()
    if not env_ok:
        print("\n⚠️ Environment file (.env) issues detected. Please fix them before continuing.")
    
    # Test API config
    api_ok = test_api_configuration()
    
    # Summary
    print("\n" + "=" * 80)
    print(" TEST SUMMARY ")
    print("=" * 80)
    print(f"Environment file (.env): {'✅ OK' if env_ok else '❌ Issues found'}")
    print(f"API Configuration/Connectivity: {'✅ OK' if api_ok else '❌ Issues found'}")
    
    # Show next steps if there are issues
    if not env_ok or not api_ok:
        print("\n" + "=" * 80)
        print(" NEXT STEPS ")
        print("=" * 80)
        print("1. Make sure you have an OpenAI API key (https://platform.openai.com/api-keys)")
        print("2. Edit the .env file in the project root with your API key")
        print("3. Ensure OPENAI_API_URL is set to 'https://api.openai.com/v1/chat/completions'")
        print("4. Run this test again")

if __name__ == "__main__":
    main() 