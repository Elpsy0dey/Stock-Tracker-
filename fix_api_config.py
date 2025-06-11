#!/usr/bin/env python
"""
Script to fix API configuration for the third-party API at free.v36.cm
"""
import os
import re
import sys
from dotenv import load_dotenv

def fix_api_config():
    """Update the API configuration in the .env file to use the correct endpoint"""
    print("\n" + "=" * 80)
    print(" API CONFIGURATION FIX ")
    print("=" * 80)
    
    env_file_path = '.env'
    if not os.path.exists(env_file_path):
        print("❌ .env file does not exist. Creating a new one...")
        with open(env_file_path, 'w') as f:
            f.write("# OpenAI API Configuration\n")
            f.write("OPENAI_API_KEY=your_api_key_here\n")
            f.write("OPENAI_API_URL=https://free.v36.cm/v1/chat/completions\n")
        print("✅ Created new .env file!")
        return True
    
    # Load current env values
    load_dotenv()
    
    current_api_key = os.getenv('OPENAI_API_KEY', '')
    current_api_url = os.getenv('OPENAI_API_URL', '')
    
    print("\nCurrent Configuration:")
    masked_key = current_api_key[:4] + "*" * 20 + current_api_key[-4:] if len(current_api_key) > 8 else "not set"
    print(f"API Key: {masked_key}")
    print(f"API URL: {current_api_url or 'not set'}")
    
    if current_api_url != "https://free.v36.cm/v1/chat/completions":
        try:
            # Read the .env file
            with open(env_file_path, 'r') as f:
                env_content = f.read()
            
            # Update the API URL
            if 'OPENAI_API_URL=' in env_content:
                env_content = re.sub(
                    r'OPENAI_API_URL=.*',
                    'OPENAI_API_URL=https://free.v36.cm/v1/chat/completions',
                    env_content
                )
            else:
                env_content += "\nOPENAI_API_URL=https://free.v36.cm/v1/chat/completions\n"
            
            # Write the updated content back
            with open(env_file_path, 'w') as f:
                f.write(env_content)
            
            print("\n✅ Updated API URL in .env file to: https://free.v36.cm/v1/chat/completions")
            return True
            
        except Exception as e:
            print(f"\n❌ Error updating .env file: {str(e)}")
            print("\nPlease manually update your .env file with:")
            print("OPENAI_API_URL=https://free.v36.cm/v1/chat/completions")
            return False
    else:
        print("\n✅ API URL is already correctly set!")
        return True

def check_auth_requirements():
    """Check the authentication requirements for the third-party API"""
    print("\n" + "=" * 80)
    print(" AUTHENTICATION REQUIREMENTS CHECK ")
    print("=" * 80)
    
    print("\nThe third-party API at free.v36.cm may require:")
    print("1. Registration on their website to get a valid API key")
    print("2. A special authentication format different from standard OpenAI")
    
    print("\nPlease visit https://free.v36.cm to check if you need to:")
    print("- Create an account")
    print("- Obtain a specific API key for this service")
    print("- Follow any special authentication requirements")
    
    print("\nIf you received an 'Unauthorized request' message, you likely need to:")
    print("1. Register on the website")
    print("2. Get a valid API key")
    print("3. Update your .env file with the new API key")

def run_tests():
    """Run API tests to check if the configuration works"""
    print("\n" + "=" * 80)
    print(" RUNNING API TESTS ")
    print("=" * 80)
    
    print("\nRunning test_api_config.py...")
    os.system("python test_api_config.py")
    
    print("\nRunning test_ai_service.py...")
    os.system("python test_ai_service.py")

if __name__ == "__main__":
    print("=" * 80)
    print(" THIRD-PARTY API CONFIGURATION FIX ")
    print("=" * 80)
    print("\nThis script will fix your API configuration to work with the third-party API at free.v36.cm")
    
    # Fix API config
    config_fixed = fix_api_config()
    
    # Check auth requirements
    check_auth_requirements()
    
    # Ask if the user wants to run tests
    if config_fixed:
        print("\nDo you want to run API tests to check if the configuration works? (y/n)")
        choice = input("> ").strip().lower()
        if choice == 'y':
            run_tests()
    
    print("\n" + "=" * 80)
    print(" SUMMARY ")
    print("=" * 80)
    print("✅ Updated API configuration to use the correct endpoint")
    print("✅ The correct URL is: https://free.v36.cm/v1/chat/completions")
    print("⚠️ You may need to register on the website to get a valid API key")
    print("⚠️ Check authentication requirements at https://free.v36.cm") 