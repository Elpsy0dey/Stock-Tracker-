#!/usr/bin/env python
"""
Helper script to register and obtain an API key for the free.v36.cm service
"""
import os
import sys
import webbrowser
import requests
from dotenv import load_dotenv

def check_registration():
    """Check if the user needs to register for the third-party API service"""
    print("\n" + "=" * 80)
    print(" THIRD-PARTY API REGISTRATION CHECK ")
    print("=" * 80)
    
    # Load current API key
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY', '')
    
    if not api_key:
        print("❌ No API key found in your .env file")
    else:
        masked_key = api_key[:4] + "*" * 20 + api_key[-4:] if len(api_key) > 8 else "***"
        print(f"Current API Key: {masked_key}")
    
    print("\nThe 'Unauthorized request' response indicates you need to register on the free.v36.cm service")
    print("and obtain a valid API key specifically for this service.")
    
    # Explain registration process
    print("\n" + "-" * 80)
    print(" HOW TO REGISTER AND GET A VALID API KEY ")
    print("-" * 80)
    print("1. Visit the website https://free.v36.cm")
    print("2. Look for a registration or sign-up link")
    print("3. Create an account with the service")
    print("4. Once registered, find the API key section")
    print("5. Generate or obtain your API key")
    print("6. Update your .env file with the new API key")
    
    # Ask if they want to open the website
    print("\nWould you like to open the free.v36.cm website in your browser? (y/n)")
    choice = input("> ").strip().lower()
    if choice == 'y':
        try:
            webbrowser.open("https://free.v36.cm")
            print("✅ Website opened in your browser")
        except Exception as e:
            print(f"❌ Could not open the website: {str(e)}")
            print("Please visit https://free.v36.cm manually")

def test_api_key():
    """Test if the current API key works with the third-party service"""
    print("\n" + "=" * 80)
    print(" API KEY TEST ")
    print("=" * 80)
    
    # Load current API key
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY', '')
    
    if not api_key:
        print("❌ No API key found. Please update your .env file first.")
        return False
    
    # Ask for a new key if they want to test a different one
    print("\nDo you want to test with a different API key? (y/n)")
    choice = input("> ").strip().lower()
    if choice == 'y':
        print("\nEnter the new API key:")
        api_key = input("> ").strip()
        
        # Ask if they want to save this key
        print("\nDo you want to save this API key to your .env file? (y/n)")
        save_choice = input("> ").strip().lower()
        if save_choice == 'y':
            try:
                # Update .env file
                with open('.env', 'r') as f:
                    env_content = f.read()
                
                # Replace API key
                if 'OPENAI_API_KEY=' in env_content:
                    env_content = env_content.replace(f"OPENAI_API_KEY={os.getenv('OPENAI_API_KEY', '')}", f"OPENAI_API_KEY={api_key}")
                else:
                    env_content += f"\nOPENAI_API_KEY={api_key}\n"
                
                # Write updated content
                with open('.env', 'w') as f:
                    f.write(env_content)
                
                print("✅ API key saved to .env file")
            except Exception as e:
                print(f"❌ Could not update .env file: {str(e)}")
    
    # Test the API key
    print("\nTesting API key...")
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'API key is valid' if you can read this message."}
            ],
            "max_tokens": 20
        }
        
        response = requests.post(
            "https://free.v36.cm/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"Response status code: {response.status_code}")
        
        if response.status_code == 200:
            print("Response content:")
            print("-" * 40)
            print(response.text[:500])
            print("-" * 40)
            
            try:
                result = response.json()
                if 'choices' in result and result['choices']:
                    content = result['choices'][0]['message']['content']
                    if "unauthorized" in content.lower():
                        print("❌ API key is not authorized for this service")
                        return False
                    else:
                        print(f"✅ API key is valid! Response: {content}")
                        return True
            except Exception:
                pass
        
        print("❌ API key test failed")
        return False
    
    except Exception as e:
        print(f"❌ Error testing API key: {str(e)}")
        return False

def main():
    """Main function"""
    print("=" * 80)
    print(" FREE.V36.CM API REGISTRATION HELPER ")
    print("=" * 80)
    
    print("\nThis script will help you register and obtain a valid API key for the free.v36.cm service.")
    
    while True:
        print("\nPlease select an option:")
        print("1. Check registration requirements")
        print("2. Test your API key")
        print("3. Run AI service tests")
        print("4. Exit")
        
        choice = input("\nEnter option number (1-4): ").strip()
        
        if choice == '1':
            check_registration()
        elif choice == '2':
            test_api_key()
        elif choice == '3':
            print("\nRunning test_ai_service.py...")
            os.system("python test_ai_service.py")
        elif choice == '4':
            print("\nExiting program...")
            break
        else:
            print("\n❌ Invalid option. Please try again.")
    
    print("\n" + "=" * 80)
    print(" SUMMARY ")
    print("=" * 80)
    print("Remember to:")
    print("1. Register on https://free.v36.cm")
    print("2. Get a valid API key")
    print("3. Update your .env file with the new API key")
    print("4. Make sure the API URL is set to https://free.v36.cm/v1/chat/completions")

if __name__ == "__main__":
    main() 