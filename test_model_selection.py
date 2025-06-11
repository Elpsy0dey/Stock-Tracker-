"""
Test script for the automatic AI model selection functionality
"""

import time
import logging
from services.model_manager import ModelManager, PREFERRED_MODELS
from config.api_config import API_CONFIG
from utils.startup import initialize_application

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_model_selection")

def test_model_selection():
    """Test the AI model selection functionality"""
    print("\n===== TESTING AI MODEL AUTO-SELECTION =====\n")
    
    # Print current configuration
    print("Original configuration:")
    print(f"- API URL: {API_CONFIG['API_URL']}")
    masked_key = API_CONFIG['API_KEY'][:4] + "*" * 10 + API_CONFIG['API_KEY'][-4:] if len(API_CONFIG['API_KEY']) > 8 else "***"
    print(f"- API KEY: {masked_key}")
    print(f"- Current Model: {API_CONFIG['MODEL']}")
    
    # Store original model for comparison
    original_model = API_CONFIG['MODEL']
    
    print("\nChecking models in preferred order:")
    for model in PREFERRED_MODELS:
        print(f"- {model}")
    
    print("\nTesting individual model availability:")
    available_models = []
    
    for model in PREFERRED_MODELS:
        print(f"\nChecking model: {model}")
        result = ModelManager.test_model(model)
        status = "✅ AVAILABLE" if result else "❌ NOT AVAILABLE"
        print(f"Status: {status}")
        
        if result:
            available_models.append(model)
        
        # Add a small delay between checks
        time.sleep(1)
    
    print("\nSummarizing available models:")
    if available_models:
        print("Available models:")
        for model in available_models:
            print(f"- {model}")
    else:
        print("No models are currently available")
    
    print("\nRunning complete initialization process:")
    try:
        # Run the full initialization process
        initialize_application()
        
        # Check if model was changed
        if API_CONFIG['MODEL'] != original_model:
            print(f"✅ Model changed from {original_model} to {API_CONFIG['MODEL']}")
        else:
            print(f"Model remained as {API_CONFIG['MODEL']}")
        
        print("\n===== MODEL SELECTION TEST COMPLETED =====")
        print(f"\nSelected model: {API_CONFIG['MODEL']}")
        
        return True
    except Exception as e:
        print(f"❌ Error during initialization: {str(e)}")
        return False

if __name__ == "__main__":
    test_model_selection() 