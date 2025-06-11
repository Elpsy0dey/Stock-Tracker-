import requests
import json
import time
import os
import re

# API configuration
API_URL = "https://free.v36.cm/v1/chat/completions"
API_KEY = "sk-F8eR7EKmeVkfIERoA8D207A1100a4467Ad97609aA37e6fE7"

# Priority order of models to try (from most preferred to least)
MODELS_TO_TRY = [
    "gpt-4o-mini",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-16k",
    "net-gpt-3.5-turbo"
]

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

def test_model(model_name):
    """Test if a specific model is working"""
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 50
    }
    
    print(f"Testing model: {model_name}")
    try:
        response = requests.post(API_URL, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            print(f"✅ Model {model_name} is working!")
            try:
                # Try to extract model response
                response_data = response.json()
                model_response = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
                print(f"Response: {model_response}")
            except:
                pass
            return True
        else:
            print(f"❌ Model {model_name} failed with status {response.status_code}")
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error testing {model_name}: {str(e)}")
        return False

def find_best_working_model():
    """Find the best working model from the priority list"""
    for model in MODELS_TO_TRY:
        if test_model(model):
            return model
        print("-" * 50)
        time.sleep(1)  # Add delay between requests
    
    return None

def update_code_file(file_path, working_model):
    """Update Python file to use working model"""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to find model references
    patterns = [
        r'model\s*=\s*["\']gpt-3\.5-turbo["\']',
        r'model\s*:\s*["\']gpt-3\.5-turbo["\']',
        r'["\']model["\']\s*:\s*["\']gpt-3\.5-turbo["\']'
    ]
    
    original_content = content
    for pattern in patterns:
        content = re.sub(pattern, f'model="{working_model}"', content)
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Updated {file_path} to use {working_model}")
        return True
    else:
        print(f"❌ No model references found in {file_path}")
        return False

def main():
    print("=== Finding the best working API model ===")
    
    working_model = find_best_working_model()
    
    if working_model:
        print(f"\n✅ Best working model found: {working_model}")
        
        # Ask user for the file to update
        file_to_update = input("\nEnter the path to the file you want to update (or press Enter to skip): ")
        
        if file_to_update:
            update_code_file(file_to_update, working_model)
    else:
        print("\n❌ No working models found. Please check your API connection or try again later.")

if __name__ == "__main__":
    main() 