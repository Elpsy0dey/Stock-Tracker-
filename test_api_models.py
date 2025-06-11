import requests
import time
import json

# API configuration
API_URL = "https://free.v36.cm/v1/chat/completions"
API_KEY = "sk-F8eR7EKmeVkfIERoA8D207A1100a4467Ad97609aA37e6fE7"

# List of models to test
MODELS = [
    "gpt-4o-mini",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo",
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
        
        print(f"Status code: {response.status_code}")
        
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

def main():
    print("=== Testing available API models ===")
    
    results = {}
    working_models = []
    
    for model in MODELS:
        result = test_model(model)
        results[model] = "Working" if result else "Not Working"
        if result:
            working_models.append(model)
        print("-" * 50)
        time.sleep(2)  # Add delay between requests to avoid rate limits
    
    print("\n=== SUMMARY ===")
    for model, status in results.items():
        print(f"{model}: {status}")
    
    print("\n=== WORKING MODELS ===")
    if working_models:
        for model in working_models:
            print(f"- {model}")
    else:
        print("No working models found!")
    
    # Recommend a model to use
    if working_models:
        # Prioritize gpt-4o-mini, then gpt-3.5-turbo-16k
        if "gpt-4o-mini" in working_models:
            recommended = "gpt-4o-mini"
        elif "gpt-3.5-turbo-16k" in working_models:
            recommended = "gpt-3.5-turbo-16k"
        elif "gpt-3.5-turbo-0125" in working_models:
            recommended = "gpt-3.5-turbo-0125"
        else:
            recommended = working_models[0]
        
        print(f"\n=== RECOMMENDATION ===")
        print(f"Replace 'gpt-3.5-turbo' with '{recommended}' in your code")

if __name__ == "__main__":
    main() 