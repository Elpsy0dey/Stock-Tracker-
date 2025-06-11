#!/usr/bin/env python
"""
Helper script to update API configuration settings in .env file
"""
import os
import re
from dotenv import load_dotenv

def update_api_config():
    """Update the API configuration in the .env file"""
    print("\n" + "=" * 80)
    print(" API CONFIGURATION UPDATE ")
    print("=" * 80)
    
    env_file_path = '.env'
    if not os.path.exists(env_file_path):
        print("❌ .env file does not exist. Creating a new one...")
        with open(env_file_path, 'w') as f:
            f.write("# OpenAI API Configuration\n")
            f.write("OPENAI_API_KEY=your_api_key_here\n")
            f.write("OPENAI_API_URL=https://api.openai.com/v1/chat/completions\n")
        print("✅ Created new .env file!")
        
    # Load current env values
    load_dotenv()
    
    current_api_key = os.getenv('OPENAI_API_KEY', '')
    current_api_url = os.getenv('OPENAI_API_URL', '')
    
    print("\nCurrent Configuration:")
    masked_key = current_api_key[:4] + "*" * 20 + current_api_key[-4:] if len(current_api_key) > 8 else "not set"
    print(f"API Key: {masked_key}")
    print(f"API URL: {current_api_url or 'not set'}")
    
    print("\nRecommended Configuration:")
    print("API Key: [your OpenAI API key]")
    print("API URL: https://api.openai.com/v1/chat/completions")
    
    print("\n" + "-" * 80)
    print("INSTRUCTIONS:")
    print("1. Open your .env file in a text editor")
    print("2. Update the API URL to: https://api.openai.com/v1/chat/completions")
    print("3. Ensure your API key is correctly set")
    print("4. Save the file")
    print("-" * 80)
    
    print("\nExample .env file content:")
    print("-" * 40)
    print("# OpenAI API Configuration")
    print("OPENAI_API_KEY=sk-yourapikeygoeshere")
    print("OPENAI_API_URL=https://api.openai.com/v1/chat/completions")
    print("-" * 40)
    
    # Attempt to read and show the issue
    try:
        with open(env_file_path, 'r') as f:
            env_content = f.read()
            
        # Check for incorrect URL
        if "https://free.v36.cm" in env_content:
            print("\n⚠️ ISSUE DETECTED: Your API URL appears to be set to https://free.v36.cm")
            print("This is not the official OpenAI API endpoint and is causing errors.")
            print("Please change it to: https://api.openai.com/v1/chat/completions")
    except Exception:
        print("\nCould not analyze .env file. Please manually check and update it.")
    
    print("\nAfter updating your configuration, run the test_api_config.py script to verify:")
    print("python test_api_config.py")

if __name__ == "__main__":
    update_api_config() 